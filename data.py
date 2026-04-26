"""
Data loading for Tiny ImageNet.

Tiny ImageNet folder layout (after unzipping):
  tiny-imagenet-200/
    train/
      n01443537/
        images/  *.JPEG
    val/
      images/  *.JPEG
      val_annotations.txt   (tab-separated: filename  class_id  x1 y1 x2 y2)
    wnids.txt               (200 class WordNet IDs, one per line)

Usage:
  train_loader, val_loader = build_loaders(args)

Mixup / CutMix:
  Use apply_mixup_cutmix(images, labels, args) inside the training loop.
  Returns (mixed_images, label_a, label_b, lam) where loss = lam * CE(pred, a)
  + (1 - lam) * CE(pred, b).
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

from timm.data.auto_augment import rand_augment_transform


# ──────────────────────────────────────────────────────────────────────────────
# Tiny ImageNet val dataset (non-standard layout)
# ──────────────────────────────────────────────────────────────────────────────

class TinyImageNetVal(Dataset):
    """
    Validation split of Tiny ImageNet.

    val/val_annotations.txt maps filenames to class WordNet IDs.
    We reuse the class-to-index mapping from the training ImageFolder so that
    label integers are consistent between splits.
    """

    def __init__(self, val_dir: str, class_to_idx: dict[str, int],
                 transform=None):
        self.img_dir = Path(val_dir) / "images"
        self.transform = transform

        anno_path = Path(val_dir) / "val_annotations.txt"
        assert anno_path.exists(), f"Missing {anno_path}"

        self.samples: list[tuple[Path, int]] = []
        with open(anno_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                fname, class_id = parts[0], parts[1]
                assert class_id in class_to_idx, \
                    f"Val class '{class_id}' not found in training classes"
                self.samples.append((self.img_dir / fname, class_to_idx[class_id]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ──────────────────────────────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = (0.4802, 0.4481, 0.3975)   # Tiny ImageNet statistics
IMAGENET_STD  = (0.2770, 0.2691, 0.2821)


def build_transforms(train: bool, args) -> transforms.Compose:
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    if train:
        # RandAugment config string expected by timm
        ra_config = (
            f"rand-m{args.randaug_magnitude}-n{args.randaug_num_ops}-mstd0.5"
        )
        return transforms.Compose([
            transforms.RandomResizedCrop(args.img_size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            rand_augment_transform(ra_config, {}),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=args.random_erasing_prob,
                                     scale=(0.02, 0.33),
                                     ratio=(0.3, 3.3),
                                     value="random"),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            normalize,
        ])


# ──────────────────────────────────────────────────────────────────────────────
# DataLoaders
# ──────────────────────────────────────────────────────────────────────────────

def build_loaders(args) -> tuple[DataLoader, DataLoader]:
    root = Path(args.data)
    assert root.exists(), f"Data root not found: {root}"

    train_dir = root / "train"
    val_dir = root / "val"
    assert train_dir.exists(), f"Missing {train_dir}"
    assert val_dir.exists(), f"Missing {val_dir}"

    train_tf = build_transforms(train=True, args=args)
    val_tf = build_transforms(train=False, args=args)

    train_dataset = ImageFolder(str(train_dir), transform=train_tf)
    val_dataset = TinyImageNetVal(str(val_dir),
                                  class_to_idx=train_dataset.class_to_idx,
                                  transform=val_tf)

    assert len(train_dataset.classes) == args.num_classes, (
        f"Expected {args.num_classes} classes, found {len(train_dataset.classes)}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"Train: {len(train_dataset):,} images  |  "
          f"Val: {len(val_dataset):,} images  |  "
          f"Classes: {args.num_classes}")
    return train_loader, val_loader


# ──────────────────────────────────────────────────────────────────────────────
# Mixup / CutMix — applied per batch in the training loop
# ──────────────────────────────────────────────────────────────────────────────

def _rand_bbox(height: int, width: int, lam: float):
    """Return a random bounding box for CutMix with area ratio (1 - lam)."""
    cut_ratio = (1.0 - lam) ** 0.5
    cut_h = int(height * cut_ratio)
    cut_w = int(width * cut_ratio)

    cx = random.randint(0, width)
    cy = random.randint(0, height)

    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, width)
    y2 = min(cy + cut_h // 2, height)
    return x1, y1, x2, y2


def apply_mixup_cutmix(
    images: torch.Tensor,
    labels: torch.Tensor,
    args,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Randomly apply either Mixup or CutMix with equal probability.

    Returns:
        images  : augmented batch
        labels_a: original labels
        labels_b: shuffled labels
        lam     : mixing coefficient  (loss = lam*CE(a) + (1-lam)*CE(b))
    """
    assert images.ndim == 4, "images must be (B, C, H, W)"
    assert labels.ndim == 1, "labels must be (B,)"

    use_cutmix = random.random() < 0.5
    alpha = args.cutmix_alpha if use_cutmix else args.mixup_alpha

    lam = float(np.random.beta(alpha, alpha))
    B, C, H, W = images.shape
    idx = torch.randperm(B, device=images.device)
    labels_a = labels
    labels_b = labels[idx]

    if use_cutmix:
        x1, y1, x2, y2 = _rand_bbox(H, W, lam)
        images = images.clone()
        images[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
        # Recompute lam based on actual pixel area
        lam = 1.0 - (x2 - x1) * (y2 - y1) / (H * W)
    else:
        images = lam * images + (1.0 - lam) * images[idx]

    return images, labels_a, labels_b, lam
