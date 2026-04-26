import argparse


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ViT on Tiny ImageNet — DeiT recipe")

    # ── Paths ──────────────────────────────────────────────────────────────────
    p.add_argument("--data", default="./tiny-imagenet-200",
                   help="Path to tiny-imagenet-200 root directory")
    p.add_argument("--output", default="./runs/exp",
                   help="Directory to save checkpoints and logs")

    # ── Architecture ───────────────────────────────────────────────────────────
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--patch-size", type=int, default=8,
                   help="Patch size (8 → 64 tokens for 64×64 images)")
    p.add_argument("--embed-dim", type=int, default=192)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--num-heads", type=int, default=3)
    p.add_argument("--mlp-ratio", type=float, default=4.0)
    p.add_argument("--drop-rate", type=float, default=0.0,
                   help="Dropout on MLP and projection layers")
    p.add_argument("--attn-drop-rate", type=float, default=0.0,
                   help="Dropout on attention weights")
    p.add_argument("--num-classes", type=int, default=200)

    # ── Optimiser ──────────────────────────────────────────────────────────────
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=5)

    # ── Augmentation ───────────────────────────────────────────────────────────
    p.add_argument("--mixup-alpha", type=float, default=0.8)
    p.add_argument("--cutmix-alpha", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--randaug-magnitude", type=int, default=9)
    p.add_argument("--randaug-num-ops", type=int, default=2)
    p.add_argument("--random-erasing-prob", type=float, default=0.25)
    p.add_argument("--no-mixup-cutmix", action="store_true",
                   help="Disable Mixup and CutMix")

    # ── AMP ────────────────────────────────────────────────────────────────────
    p.add_argument("--no-amp", action="store_true",
                   help="Disable automatic mixed precision")

    # ── W&B ────────────────────────────────────────────────────────────────────
    p.add_argument("--wandb", action="store_true",
                   help="Enable Weights & Biases logging")
    p.add_argument("--wandb-project", default="vibe-vit")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--log-freq", type=int, default=50,
                   help="Log train loss/lr every N batches")

    # ── Misc ───────────────────────────────────────────────────────────────────
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training; only evaluate the checkpoint given by --resume")

    args = p.parse_args()

    assert args.img_size % args.patch_size == 0, (
        f"img_size ({args.img_size}) must be divisible by patch_size ({args.patch_size})"
    )
    assert args.embed_dim % args.num_heads == 0, (
        f"embed_dim ({args.embed_dim}) must be divisible by num_heads ({args.num_heads})"
    )

    return args
