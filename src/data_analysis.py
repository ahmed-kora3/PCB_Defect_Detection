"""
Exploratory data analysis — class distribution, imbalance diagnostics, random PCB + YOLO overlays.

Outputs (default under results/visualizations/):
  - class_distribution.png
  - random_pcb_grid_with_boxes.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .data_loader import CLASS_NAMES, DATA_ROOT, build_annotation_manifest, count_unique_images
from .yolo_visualization import visualize_label_file


def count_annotation_instances_per_class(manifest: list) -> dict[str, int]:
    counts: dict[str, int] = {CLASS_NAMES[i]: 0 for i in CLASS_NAMES}
    for m in manifest:
        counts[CLASS_NAMES[m["class_id"]]] += 1
    return counts


def count_unique_images_per_class(manifest: list) -> dict[str, int]:
    """Distinct image files that contain at least one box of that class."""
    per: dict[str, set[str]] = {CLASS_NAMES[i]: set() for i in CLASS_NAMES}
    for m in manifest:
        key = str(m["image_path"])
        per[CLASS_NAMES[m["class_id"]]].add(key)
    return {c: len(per[c]) for c in per}


def analyze_class_imbalance(counts: dict[str, int]) -> tuple[str, dict[str, float]]:
    vals = np.array(list(counts.values()), dtype=np.float64)
    total = float(vals.sum())
    ratios = {c: counts[c] / total for c in counts}
    min_c, max_c = min(counts, key=counts.get), max(counts, key=counts.get)
    imbalance_ratio = counts[max_c] / max(counts[min_c], 1)
    # Normalized entropy of class distribution (1 = perfectly balanced)
    p = vals / max(vals.sum(), 1.0)
    p = p[p > 0]
    entropy = float(-(p * np.log(p + 1e-12)).sum() / np.log(len(counts)))
    lines = [
        "=== Class imbalance analysis ===",
        f"Total defect instances (annotations): {int(total)}",
        f"Majority class: {max_c} ({counts[max_c]} instances)",
        f"Minority class: {min_c} ({counts[min_c]} instances)",
        f"Max/min instance ratio: {imbalance_ratio:.2f}",
        f"Balance score (normalized entropy, 1=balanced): {entropy:.3f}",
        "",
        "If imbalance_ratio is large, use class_weight in training (enabled by default in train.py).",
    ]
    return "\n".join(lines), {"imbalance_ratio": imbalance_ratio, "entropy_balance": entropy, **ratios}


def plot_class_distribution_bar(counts: dict[str, int], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(counts.keys())
    values = [counts[n] for n in names]
    plt.figure(figsize=(10, 5))
    plt.bar(names, values, color="steelblue", edgecolor="black")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Annotation count")
    plt.title("PCB defect class distribution (YOLO instances)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_random_visualization_grid(split: str = "train", n: int = 8, seed: int = 0, save_path: Path | None = None) -> Path:
    rng = np.random.default_rng(seed)
    manifest = build_annotation_manifest(split)
    if not manifest:
        raise RuntimeError("Empty manifest")
    picks = rng.choice(len(manifest), size=min(n, len(manifest)), replace=False)
    cols = 4
    rows_n = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 4 * rows_n))
    axes = np.atleast_1d(axes).ravel()
    for ax, idx in zip(axes, picks):
        rec = manifest[int(idx)]
        lp = DATA_ROOT / split / "labels" / rec["label_file"]
        vis = visualize_label_file(lp)
        ax.imshow(vis)
        ax.set_title(CLASS_NAMES[rec["class_id"]], fontsize=9)
        ax.axis("off")
    for ax in axes[len(picks) :]:
        ax.axis("off")
    plt.suptitle(f"Random PCB samples with YOLO boxes — {split}", fontsize=12)
    plt.tight_layout()
    save_path = save_path or Path("results/visualizations/random_pcb_grid_with_boxes.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def run_full_data_analysis(
    output_viz_dir: Path | None = None,
    split: str = "train",
) -> dict:
    out = output_viz_dir or Path("results/visualizations")
    out.mkdir(parents=True, exist_ok=True)

    train_m = build_annotation_manifest("train")
    val_m = build_annotation_manifest("val")
    test_m = build_annotation_manifest("test")
    combined = train_m + val_m + test_m

    counts = count_annotation_instances_per_class(combined)
    img_per_class = count_unique_images_per_class(combined)
    n_unique_images = count_unique_images(combined)

    plot_class_distribution_bar(counts, out / "class_distribution.png")
    grid_path = save_random_visualization_grid(split=split, n=8, save_path=out / "random_pcb_grid_with_boxes.png")

    text, stats = analyze_class_imbalance(counts)
    print(text)
    print(f"\nUnique images covered by manifest (all splits): {n_unique_images}")
    print("Images-per-class (at least one box of that class):", img_per_class)

    summary_path = out / "data_analysis_summary.txt"
    summary_path.write_text(
        text
        + "\n\nUnique images (all splits): "
        + str(n_unique_images)
        + "\n\nImages per class:\n"
        + "\n".join(f"  {k}: {v}" for k, v in sorted(img_per_class.items()))
        + "\n\nSaved:\n  "
        + str(out / "class_distribution.png")
        + "\n  "
        + str(grid_path),
        encoding="utf-8",
    )

    return {
        "counts": counts,
        "stats": stats,
        "unique_images": n_unique_images,
        "class_distribution_png": str(out / "class_distribution.png"),
        "random_grid_png": str(grid_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA: class distribution + random YOLO visualizations.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--out", type=Path, default=Path("results/visualizations"))
    args = parser.parse_args()
    run_full_data_analysis(output_viz_dir=args.out, split=args.split)
    print("EDA complete. Outputs under:", args.out.resolve())


if __name__ == "__main__":
    main()
