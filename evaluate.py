"""
Standalone evaluation script.

Usage:
  python evaluate.py --data /path/to/tiny-imagenet-200 --checkpoint runs/exp1/best.pt
  python evaluate.py --data /path/to/tiny-imagenet-200 --checkpoint runs/exp1/best.pt --wandb
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import autocast
from tqdm import tqdm

from data import IMAGENET_MEAN, IMAGENET_STD, build_loaders
from model import VisionTransformer

# ──────────────────────────────────────────────────────────────────────────────
# Args (separate from train.py so evaluate.py can be run standalone)
# ──────────────────────────────────────────────────────────────────────────────

def get_eval_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a ViT checkpoint")
    p.add_argument("--data", required=True,
                   help="Path to tiny-imagenet-200 root")
    p.add_argument("--checkpoint", required=True,
                   help="Path to checkpoint file (.pt)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--wandb", action="store_true",
                   help="Log results to W&B (requires the run to exist or a new one is created)")
    p.add_argument("--wandb-project", default="vibe-vit")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--conf-matrix", action="store_true",
                   help="Print a per-class accuracy breakdown")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def accuracy_topk(output: torch.Tensor, target: torch.Tensor,
                  topk=(1, 5)) -> list[float]:
    with torch.no_grad():
        maxk = max(topk)
        B = target.size(0)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [correct[:k].reshape(-1).float().sum(0).item() * 100.0 / B
                for k in topk]


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device,
             use_amp: bool, num_classes: int) -> dict:
    model.eval()

    total_top1 = 0.0
    total_top5 = 0.0
    total_loss = 0.0
    total_n = 0

    criterion = nn.CrossEntropyLoss()

    all_preds: list[int] = []
    all_labels: list[int] = []
    per_class_correct = torch.zeros(num_classes, dtype=torch.long)
    per_class_total = torch.zeros(num_classes, dtype=torch.long)

    for images, labels in tqdm(loader, desc="Evaluating"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        B = images.size(0)

        with autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        t1, t5 = accuracy_topk(logits, labels, topk=(1, 5))
        total_top1 += t1 * B
        total_top5 += t5 * B
        total_loss += loss.item() * B
        total_n += B

        preds = logits.argmax(dim=1).cpu()
        labels_cpu = labels.cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels_cpu.tolist())

        for c in range(num_classes):
            mask = labels_cpu == c
            per_class_correct[c] += (preds[mask] == labels_cpu[mask]).sum()
            per_class_total[c] += mask.sum()

    assert total_n > 0, "No samples evaluated"

    results = {
        "loss": total_loss / total_n,
        "top1": total_top1 / total_n,
        "top5": total_top5 / total_n,
        "error1": 100.0 - total_top1 / total_n,
        "error5": 100.0 - total_top5 / total_n,
        "all_preds": all_preds,
        "all_labels": all_labels,
        "per_class_acc": (per_class_correct.float() /
                          per_class_total.float().clamp(min=1) * 100).tolist(),
    }
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = get_eval_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"

    ckpt_path = Path(args.checkpoint)
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    saved_args = ckpt.get("args", {})

    # Reconstruct model from saved args; override batch/workers from eval args
    model = VisionTransformer(
        img_size=saved_args.get("img_size", 64),
        patch_size=saved_args.get("patch_size", 8),
        num_classes=saved_args.get("num_classes", 200),
        embed_dim=saved_args.get("embed_dim", 192),
        depth=saved_args.get("depth", 12),
        num_heads=saved_args.get("num_heads", 3),
        mlp_ratio=saved_args.get("mlp_ratio", 4.0),
        drop_rate=0.0,         # no dropout at eval
        attn_drop_rate=0.0,
    ).to(device)

    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}  "
          f"(saved best top-1: {ckpt.get('best_top1', 0.0):.2f}%)")

    # ── Data — reuse build_loaders with a minimal namespace ───────────────────
    import types
    loader_args = types.SimpleNamespace(
        data=args.data,
        img_size=saved_args.get("img_size", 64),
        patch_size=saved_args.get("patch_size", 8),
        num_classes=saved_args.get("num_classes", 200),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        randaug_magnitude=9,
        randaug_num_ops=2,
        random_erasing_prob=0.0,   # no erasing at eval
    )
    _, val_loader = build_loaders(loader_args)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    num_classes = saved_args.get("num_classes", 200)
    results = evaluate(model, val_loader, device, use_amp, num_classes)

    print(f"\n{'─'*50}")
    print(f"  Val loss : {results['loss']:.4f}")
    print(f"  Top-1    : {results['top1']:.2f}%  (error: {results['error1']:.2f}%)")
    print(f"  Top-5    : {results['top5']:.2f}%  (error: {results['error5']:.2f}%)")
    print(f"{'─'*50}\n")

    if args.conf_matrix:
        per_class = results["per_class_acc"]
        print("Per-class accuracy:")
        for i, acc in enumerate(per_class):
            print(f"  Class {i:3d}: {acc:.1f}%")

    # ── W&B ───────────────────────────────────────────────────────────────────
    if args.wandb:
        import wandb
        from sklearn.metrics import confusion_matrix

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"eval-{ckpt_path.stem}",
            config={**saved_args, "eval_checkpoint": str(ckpt_path)},
        )
        wandb.log({
            "val/loss": results["loss"],
            "val/top1": results["top1"],
            "val/top5": results["top5"],
            "val/error1": results["error1"],
            "val/error5": results["error5"],
        })
        wandb.log({
            "val/confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=results["all_labels"],
                preds=results["all_preds"],
                class_names=[str(i) for i in range(num_classes)],
            )
        })
        wandb.finish()


if __name__ == "__main__":
    main()
