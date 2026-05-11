"""
STEP 2 & 6 — YOLO visualization and bonus detection narrative.

Bounding boxes in YOLO format are stored normalized to image width/height:
- x_center, y_center: centre of the box in relative coordinates (0–1).
- width, height: box size relative to full image.

To draw on pixels: multiply centres and sizes by image W/H, convert to top-left corner:
  x1 = x_c * W - (w * W) / 2,  y1 = y_c * H - (h * H) / 2 (then x2, y2 similarly).

For full YOLO *training* (YOLOv5/v8), you would export this dataset to YOLO folder layout and run
Ultralytics `yolo train` — heavier than patch classification; this project uses labels for crops +
visual validation, which matches common final-year scope.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .data_loader import CLASS_NAMES, DATA_ROOT, build_annotation_manifest, get_image_path_for_label


def yolo_norm_to_pixel_xyxy(
    xc: float,
    yc: float,
    bw: float,
    bh: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """
    Convert YOLO-normalized box (centre + size) to integer pixel corners (x1,y1,x2,y2).

    YOLO stores xc,yc,w,h in [0,1] relative to image width/height respectively.
    """
    xc_px = xc * image_width
    yc_px = yc * image_height
    w_px = bw * image_width
    h_px = bh * image_height
    x1 = int(round(xc_px - w_px / 2))
    y1 = int(round(yc_px - h_px / 2))
    x2 = int(round(xc_px + w_px / 2))
    y2 = int(round(yc_px + h_px / 2))
    return x1, y1, x2, y2


def read_yolo_label_file(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    rows: List[Tuple[int, float, float, float, float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        p = line.split()
        rows.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    return rows


def draw_boxes_on_image(
    image_rgb: np.ndarray,
    yolo_rows: Sequence[Tuple[int, float, float, float, float]],
    class_names: dict[int, str] | None = None,
) -> np.ndarray:
    """Return a copy of RGB image with rectangles and class names drawn."""
    names = class_names or CLASS_NAMES
    out = image_rgb.copy()
    h, w = out.shape[:2]
    for class_id, xc, yc, bw, bh in yolo_rows:
        x1, y1, x2, y2 = yolo_norm_to_pixel_xyxy(xc, yc, bw, bh, w, h)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = names.get(class_id, str(class_id))
        cv2.putText(
            out,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 64, 0),
            2,
            cv2.LINE_AA,
        )
    return out


def visualize_label_file(label_path: Path, save_path: Path | None = None) -> np.ndarray:
    image_path = get_image_path_for_label(label_path)
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rows = read_yolo_label_file(label_path)
    vis = draw_boxes_on_image(rgb, rows)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    return vis


def plot_random_samples(split: str = "train", n: int = 6, seed: int = 42, show: bool = True) -> None:
    rng = np.random.default_rng(seed)
    manifest = build_annotation_manifest(split)
    if not manifest:
        raise RuntimeError(f"Empty manifest for split {split!r}")
    picks = rng.choice(len(manifest), size=min(n, len(manifest)), replace=False)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    for ax, idx in zip(axes, picks):
        rec = manifest[int(idx)]
        label_path = DATA_ROOT / split / "labels" / rec["label_file"]
        vis = visualize_label_file(label_path)
        ax.imshow(vis)
        ax.set_title(CLASS_NAMES[rec["class_id"]])
        ax.axis("off")
    for ax in axes[len(picks) :]:
        ax.axis("off")
    plt.suptitle(f"YOLO annotation check — {split} split")
    plt.tight_layout()
    if show:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize YOLO labels on PCB images.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--label-file", default=None, help="Specific label .txt filename (optional).")
    parser.add_argument("--grid", type=int, default=0, help="If >0, save a grid of N random samples to results/.")
    parser.add_argument("--out", type=Path, default=Path("results") / "yolo_vis")
    args = parser.parse_args()

    label_dir = DATA_ROOT / args.split / "labels"
    if args.label_file:
        vis = visualize_label_file(label_dir / args.label_file, args.out / "single.png")
        plt.figure(figsize=(8, 8))
        plt.imshow(vis)
        plt.axis("off")
        plt.title(args.label_file)
        plt.tight_layout()
        plt.savefig(args.out / "single_preview.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved:", args.out / "single.png")
        return

    if args.grid > 0:
        args.out.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(42)
        manifest = build_annotation_manifest(args.split)
        picks = rng.choice(len(manifest), size=min(args.grid, len(manifest)), replace=False)
        cols = min(3, args.grid)
        rows_n = int(np.ceil(len(picks) / cols))
        fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 4 * rows_n))
        axes = np.atleast_1d(axes).ravel()
        for ax, idx in zip(axes, picks):
            rec = manifest[int(idx)]
            lp = DATA_ROOT / args.split / "labels" / rec["label_file"]
            ax.imshow(visualize_label_file(lp))
            ax.set_title(CLASS_NAMES[rec["class_id"]], fontsize=8)
            ax.axis("off")
        for ax in axes[len(picks) :]:
            ax.axis("off")
        plt.suptitle(f"YOLO samples — {args.split}")
        plt.tight_layout()
        fig.savefig(args.out / f"grid_{args.split}.png", dpi=150)
        plt.close(fig)
        print("Saved:", args.out / f"grid_{args.split}.png")
        return

    first = sorted(label_dir.glob("*.txt"))[0]
    vis = visualize_label_file(first, args.out / "first_label.png")
    plt.figure(figsize=(8, 8))
    plt.imshow(vis)
    plt.axis("off")
    plt.title(first.name)
    plt.tight_layout()
    plt.savefig(args.out / "first_label_preview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", args.out / "first_label.png")


if __name__ == "__main__":
    main()
