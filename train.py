"""
Training script — ViT on Tiny ImageNet with the DeiT recipe.

Usage:
  python train.py --data /path/to/tiny-imagenet-200 --output ./runs/exp1
  python train.py --data /path/to/tiny-imagenet-200 --wandb --wandb-run-name my-run
  python train.py --data /path/to/tiny-imagenet-200 --embed-dim 384 --depth 6
"""

import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from config import get_args
from data import apply_mixup_cutmix, build_loaders
from model import build_model

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


def accuracy(output: torch.Tensor, target: torch.Tensor,
             topk=(1, 5)) -> list[float]:
    """Return top-k accuracy values (%) for the given batch."""
    with torch.no_grad():
        maxk = max(topk)
        B = target.size(0)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()                          # (maxk, B)
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [correct[:k].reshape(-1).float().sum(0).item() * 100.0 / B
                for k in topk]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


def warmup_cosine_lr(epoch: int, warmup_epochs: int,
                     total_epochs: int, base_lr: float) -> float:
    """Return the learning rate for a given epoch (linear warmup + cosine decay)."""
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(state: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


# ──────────────────────────────────────────────────────────────────────────────
# W&B helpers (all calls guarded by use_wandb)
# ──────────────────────────────────────────────────────────────────────────────

def wandb_init(args):
    import wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
    )


def wandb_log_step(step: int, loss: float, lr: float):
    import wandb
    wandb.log({"train/loss_step": loss, "train/lr": lr}, step=step)


def wandb_log_epoch(epoch: int, metrics: dict, model: nn.Module, use_grad: bool = True):
    """Log per-epoch metrics plus grad / param histograms."""
    import wandb
    log_dict = {f"{k}": v for k, v in metrics.items()}

    if use_grad:
        for name, param in model.named_parameters():
            if param.data is not None:
                log_dict[f"params/{name}"] = wandb.Histogram(
                    param.data.cpu().float().numpy()
                )
            if param.grad is not None:
                log_dict[f"gradients/{name}"] = wandb.Histogram(
                    param.grad.cpu().float().numpy()
                )

    wandb.log(log_dict, step=epoch)


def wandb_log_confusion(epoch: int, all_preds: list[int], all_labels: list[int],
                        num_classes: int):
    import wandb
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    # Log as a wandb Table (N×N can be heavy; log raw array as artifact)
    wandb.log({
        "val/confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=all_labels,
            preds=all_preds,
            class_names=[str(i) for i in range(num_classes)],
        )
    }, step=epoch)


# ──────────────────────────────────────────────────────────────────────────────
# Training epoch
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    args,
    use_amp: bool,
    use_wandb: bool,
    global_step: int,
) -> tuple[float, float, float, int]:
    """
    Returns (avg_loss, top1, top5, updated_global_step).
    """
    model.train()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d} [train]", leave=False)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        B = images.size(0)

        # ── Mixup / CutMix ────────────────────────────────────────────────────
        if not args.no_mixup_cutmix:
            images, labels_a, labels_b, lam = apply_mixup_cutmix(images, labels, args)
            mixed = True
        else:
            mixed = False

        # ── Forward + loss ────────────────────────────────────────────────────
        with autocast("cuda", enabled=use_amp):
            logits = model(images)
            if mixed:
                loss = lam * criterion(logits, labels_a) + \
                       (1.0 - lam) * criterion(logits, labels_b)
            else:
                loss = criterion(logits, labels)

        # ── Backward ──────────────────────────────────────────────────────────
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # ── Metrics (use hard labels for accuracy even under Mixup) ───────────
        with torch.no_grad():
            hard_labels = labels if not mixed else labels_a
            t1, t5 = accuracy(logits, hard_labels, topk=(1, 5))

        loss_meter.update(loss.item(), B)
        top1_meter.update(t1, B)
        top5_meter.update(t5, B)
        global_step += 1

        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", top1=f"{top1_meter.avg:.2f}%")

        if use_wandb and global_step % args.log_freq == 0:
            wandb_log_step(global_step, loss.item(), get_lr(optimizer))

    return loss_meter.avg, top1_meter.avg, top5_meter.avg, global_step


# ──────────────────────────────────────────────────────────────────────────────
# Validation epoch
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    collect_preds: bool = False,
) -> tuple[float, float, float, list, list]:
    """
    Returns (avg_loss, top1, top5, all_preds, all_labels).
    all_preds / all_labels are populated only when collect_preds=True.
    """
    model.eval()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()
    all_preds: list[int] = []
    all_labels: list[int] = []

    pbar = tqdm(loader, desc="            [val]  ", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        B = images.size(0)

        with autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        t1, t5 = accuracy(logits, labels, topk=(1, 5))
        loss_meter.update(loss.item(), B)
        top1_meter.update(t1, B)
        top5_meter.update(t5, B)

        if collect_preds:
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", top1=f"{top1_meter.avg:.2f}%")

    return loss_meter.avg, top1_meter.avg, top5_meter.avg, all_preds, all_labels


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    use_wandb = args.wandb

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}  |  AMP: {use_amp}  |  W&B: {use_wandb}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader = build_loaders(args)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(args).to(device)

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    # Exclude bias / LayerNorm params from weight decay (DeiT convention)
    decay_params, nodecay_params = [], []
    for name, param in model.named_parameters():
        if param.ndim <= 1 or name.endswith(".bias"):
            nodecay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": args.weight_decay},
         {"params": nodecay_params, "weight_decay": 0.0}],
        lr=args.lr,
    )

    scaler = GradScaler("cuda", enabled=use_amp)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_top1 = 0.0
    global_step = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_top1 = ckpt.get("best_top1", 0.0)
        global_step = ckpt.get("global_step", 0)
        print(f"Resumed from epoch {ckpt['epoch']}  (best top1={best_top1:.2f}%)")

    if args.eval_only:
        assert args.resume, "--eval-only requires --resume"
        val_loss, val_top1, val_top5, _, _ = validate(
            model, val_loader, criterion, device, use_amp
        )
        print(f"Val  loss={val_loss:.4f}  top1={val_top1:.2f}%  top5={val_top5:.2f}%")
        return

    # ── W&B init ──────────────────────────────────────────────────────────────
    if use_wandb:
        wandb_init(args)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        # Cosine LR with linear warmup (set before stepping)
        new_lr = warmup_cosine_lr(epoch, args.warmup_epochs, args.epochs, args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = new_lr

        t0 = time.time()
        tr_loss, tr_top1, tr_top5, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, args, use_amp, use_wandb, global_step,
        )

        # Collect predictions for confusion matrix on last epoch or every 10
        collect = use_wandb and (epoch % 10 == 0 or epoch == args.epochs - 1)
        val_loss, val_top1, val_top5, all_preds, all_labels = validate(
            model, val_loader, criterion, device, use_amp,
            collect_preds=collect,
        )

        elapsed = time.time() - t0
        print(
            f"[{epoch:3d}/{args.epochs}]  "
            f"lr={new_lr:.2e}  "
            f"train loss={tr_loss:.4f}  top1={tr_top1:.2f}%  top5={tr_top5:.2f}%  |  "
            f"val  loss={val_loss:.4f}  top1={val_top1:.2f}%  top5={val_top5:.2f}%  "
            f"({elapsed:.0f}s)"
        )

        # ── W&B logging ───────────────────────────────────────────────────────
        if use_wandb:
            metrics = {
                "train/loss": tr_loss,
                "train/top1": tr_top1,
                "train/top5": tr_top5,
                "val/loss": val_loss,
                "val/top1": val_top1,
                "val/top5": val_top5,
                "train/error": 100.0 - tr_top1,
                "val/error": 100.0 - val_top1,
                "epoch": epoch,
                "lr": new_lr,
            }
            wandb_log_epoch(epoch, metrics, model, use_grad=True)

            if collect:
                wandb_log_confusion(epoch, all_preds, all_labels, args.num_classes)

        # ── Checkpoint ────────────────────────────────────────────────────────
        is_best = val_top1 > best_top1
        best_top1 = max(val_top1, best_top1)

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_top1": best_top1,
            "global_step": global_step,
            "args": vars(args),
        }
        save_checkpoint(state, output_dir / "last.pt")
        if is_best:
            save_checkpoint(state, output_dir / "best.pt")
            print(f"  ↑ New best: {best_top1:.2f}%")

    print(f"\nTraining complete. Best val top-1: {best_top1:.2f}%")

    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
