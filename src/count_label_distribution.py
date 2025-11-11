#!/usr/bin/env python3
import os
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image

from config import Config


def iter_mask_files(root: Path, exts: List[str]):
    for f in sorted(root.iterdir()):
        if f.is_file() and f.suffix.lower() in exts:
            yield f

def main():
    cfg = Config()
    mask_dir = Path(os.path.join(cfg.data_dir, 'masks'))
    classes = list(cfg.class_names_to_labels.values())
    class_names = list(cfg.class_names_to_labels.keys())
    if not mask_dir.is_dir():
        raise SystemExit(f"Not a directory: {mask_dir}")

    extensions = ['.png', '.jpg', '.jpeg']

    # Accumulate counts with 256-long histogram (8-bit masks).
    hist = np.zeros(256, dtype=np.int64)
    total_pixels = 0
    total_images = 0

    for path in iter_mask_files(mask_dir, extensions):
        with Image.open(path) as im:
            im = im.convert("L")  # single-channel 8-bit
            arr = np.array(im, dtype=np.uint8)
        binc = np.bincount(arr.ravel(), minlength=256)
        hist += binc
        total_pixels += arr.size
        total_images += 1

    if total_images == 0:
        raise SystemExit("No mask images found with given extensions.")

    # Extract counts for requested classes
    counts = np.array([hist[c] for c in classes], dtype=np.int64)
    counted_pixels = counts.sum()

    # Percentages over all pixels (recommended for loss weights)
    pct_of_total = counts / total_pixels * 100.0

    # Print report
    print(f"Images: {total_images}")
    print(f"Total pixels (all images): {total_pixels:,}")
    print()
    print(f"{'Class':>15} | {'Count':>14} | {'% of total':>11}")
    print("-" * 48)
    for class_name, label, n, p_all in zip(class_names, classes, counts, pct_of_total):
        print(f"{class_name:>15} | {n:>14,} | {p_all:11.6f}")
    print("-" * 48)
    print(f"{'SUM':>15} | {counted_pixels:>14,} | {counted_pixels/total_pixels*100:11.6f}")
    print()

    # Suggested class weights for CrossEntropyLoss
    # Option A: inverse frequency normalized to mean=1
    #   w_c = (total_pixels / (K * count_c)) ; then renormalize to mean=1 for stability
    K = len(classes)
    inv_freq = (total_pixels / np.maximum(1, counts)) / K
    inv_freq = inv_freq / inv_freq.mean()
    # Option B: median frequency balancing
    med = np.median(counts[counts > 0]) if np.any(counts > 0) else 1.0
    med_freq = med / np.maximum(1, counts)

    print("Weights (inverse-frequency, mean≈1):")
    print("[" + ", ".join(f"{w:.6f}" for w in inv_freq) + "]")
    print("Weights (median-frequency balancing):")
    print("[" + ", ".join(f"{w:.6f}" for w in med_freq) + "]")
    print()
    print("Paste into training as:")
    print("  class_weights = torch.tensor([{}], dtype=torch.float32, device=device)".format(
        ", ".join(f"{w:.6f}" for w in inv_freq)
    ))
    print("  loss = F.cross_entropy(logits, targets, weight=class_weights, ignore_index=-1)")

if __name__ == "__main__":
    main()
