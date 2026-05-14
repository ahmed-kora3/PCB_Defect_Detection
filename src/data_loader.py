"""
Dataset loading — raw Kaggle PCB images + YOLO .txt labels (manual I/O, no built-in CV datasets).

Each YOLO line: class_id x_center y_center width height (normalized 0–1 w.r.t. image W, H).

Corrupted or unreadable images are skipped when building the manifest so training does not fail
on broken files (OpenCV returns None).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

from .config import DATA_ROOT, PROJECT_ROOT

SPLITS = ("train", "val", "test")

FALLBACK_CLASS_NAMES = {
    0: "mouse_bite",
    1: "spur",
    2: "missing_hole",
    3: "short",
    4: "open_circuit",
    5: "spurious_copper",
}


def load_class_names() -> Dict[int, str]:
    """Parse names: block from data.yaml (YOLO-style class index → string label)."""
    yaml_path = DATA_ROOT / "data.yaml"
    if not yaml_path.exists():
        return dict(FALLBACK_CLASS_NAMES)

    result: Dict[int, str] = {}
    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("names:"):
            continue
        if ":" in stripped and stripped and stripped[0].isdigit():
            key, value = stripped.split(":", 1)
            try:
                result[int(key.strip())] = value.strip()
            except ValueError:
                continue
    return result if result else dict(FALLBACK_CLASS_NAMES)


CLASS_NAMES = load_class_names()
NUM_CLASSES = len(CLASS_NAMES)


def parse_label_line(label_line: str) -> Tuple[int, float, float, float, float]:
    parts = label_line.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Invalid YOLO label line: {label_line!r}")
    class_id = int(parts[0])
    coords = [float(x) for x in parts[1:]]
    return class_id, coords[0], coords[1], coords[2], coords[3]


def verify_image_readable(image_path: Path) -> bool:
    """
    Return True if OpenCV can decode a non-empty BGR image.
    Used to skip corrupted downloads or truncated files safely.
    """
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            return False
        if len(img.shape) < 2 or img.shape[0] < 2 or img.shape[1] < 2:
            return False
        return True
    except (OSError, cv2.error):
        return False


def _stem_variants(label_stem: str) -> List[str]:
    stems = {label_stem}
    s = label_stem
    for suffix in ("_256", "_600"):
        if s.endswith(suffix):
            base = s[: -len(suffix)]
            stems.add(base)
            stems.add(base + "_256")
            stems.add(base + "_600")
    if "_256" in s or "_600" in s:
        stems.add(s.replace("_256", "").replace("_600", ""))
    if s.startswith("rotation_"):
        stems.add(s.replace("rotation_", "", 1))
    if s.startswith("l_light_"):
        stems.add(s.replace("l_light_", "light_", 1))
    return list(stems)


def get_image_path_for_label(label_path: Path) -> Path:
    image_dir = label_path.parent.parent / "images"
    for candidate in _stem_variants(label_path.stem):
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
            candidate_path = image_dir / f"{candidate}{ext}"
            if candidate_path.exists():
                return candidate_path
    raise FileNotFoundError(f"No image found for label file: {label_path}")


def find_label_path_for_image(image_path: Path, split: str) -> Path | None:
    """
    Given an image path (or any Path whose `.stem` matches the image filename stem),
    return the corresponding YOLO label path under `DATA_ROOT/<split>/labels/` if found.

    The Kaggle release sometimes uses `_256`/`_600` suffixes or other stem variants; this
    helper mirrors `get_image_path_for_label()` by trying common stem variants.
    """
    if split not in SPLITS:
        raise ValueError(f"Unsupported split {split!r}. Use one of {SPLITS}")
    label_dir = DATA_ROOT / split / "labels"
    if not label_dir.exists():
        return None
    for candidate in _stem_variants(image_path.stem):
        p = label_dir / f"{candidate}.txt"
        if p.exists():
            return p
    return None


def build_annotation_manifest(split: str, skip_unreadable_images: bool = True) -> List[Dict]:
    """
    Build one manifest row per YOLO line (one defect instance).
    Optionally drops label files whose matched image fails verify_image_readable().
    """
    if split not in SPLITS:
        raise ValueError(f"Unsupported split {split!r}. Use one of {SPLITS}")

    label_dir = DATA_ROOT / split / "labels"
    if not label_dir.exists():
        raise FileNotFoundError(f"Missing label directory: {label_dir}")

    manifest: List[Dict] = []
    for label_file in sorted(label_dir.glob("*.txt")):
        try:
            image_path = get_image_path_for_label(label_file)
        except FileNotFoundError:
            continue
        if skip_unreadable_images and not verify_image_readable(image_path):
            continue
        try:
            with label_file.open("r", encoding="utf-8") as handle:
                lines = [ln for ln in handle if ln.strip()]
        except OSError:
            continue
        for line in lines:
            try:
                class_id, xc, yc, bw, bh = parse_label_line(line)
            except ValueError:
                continue
            manifest.append(
                {
                    "split": split,
                    "image_path": image_path,
                    "class_id": class_id,
                    "bbox": [xc, yc, bw, bh],
                    "label_file": label_file.name,
                }
            )
    return manifest


def get_class_counts(manifest: List[Dict]) -> Dict[str, int]:
    counts = {CLASS_NAMES[i]: 0 for i in CLASS_NAMES}
    for item in manifest:
        counts[CLASS_NAMES[item["class_id"]]] += 1
    return counts


def compute_class_weight_dict(manifest: List[Dict]) -> Dict[int, float]:
    """
    Balanced class weights for Keras fit(class_weight=...).
    Mitigates moderate imbalance without resampling the raw files.
    """
    from sklearn.utils.class_weight import compute_class_weight

    y = np.array([m["class_id"] for m in manifest], dtype=np.int32)
    classes = np.arange(NUM_CLASSES)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def save_manifest_csv(manifest: List[Dict], output_path: Path) -> None:
    rows = []
    for row in manifest:
        img = row["image_path"]
        rows.append(
            {
                "split": row["split"],
                "image_path": str(img),
                "class_id": row["class_id"],
                "class_name": CLASS_NAMES[row["class_id"]],
                "bbox_xc": row["bbox"][0],
                "bbox_yc": row["bbox"][1],
                "bbox_w": row["bbox"][2],
                "bbox_h": row["bbox"][3],
                "label_file": row["label_file"],
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def count_unique_images(manifest: List[Dict]) -> int:
    return len({str(m["image_path"]) for m in manifest})
