"""
Real-time and batch inference — load best_model.keras, preprocess consistently with training
(read hyperparameters.json for advanced_preprocessing if present next to the model).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from .data_loader import CLASS_NAMES, NUM_CLASSES
from .preprocess import crop_and_resize


def load_prep_flags(model_dir: Path) -> dict:
    hp = model_dir.parent / "hyperparameters.json"
    if not hp.exists():
        return {"advanced_preprocessing": False, "image_size": 224}
    return json.loads(hp.read_text(encoding="utf-8"))


def predict_image(
    model: tf.keras.Model,
    bgr: np.ndarray,
    yolo_box_norm: list[float],
    image_size: int = 224,
    advanced: bool = False,
) -> tuple[str, float, np.ndarray]:
    """
    yolo_box_norm: [xc, yc, w, h] in 0-1 for full image. For single full-frame demo use e.g. [0.5,0.5,0.4,0.4].
    Returns class name, confidence, patch float32.
    """
    patch = crop_and_resize(bgr, yolo_box_norm, (image_size, image_size), advanced_preprocessing=advanced)
    batch = np.expand_dims(patch, axis=0)
    p = model.predict(batch, verbose=0)[0]
    idx = int(np.argmax(p))
    return CLASS_NAMES[idx], float(p[idx]), patch


def predict_folder(model_path: Path, input_dir: Path, output_csv: Path | None, box: list[float]) -> None:
    model_dir = model_path.parent.parent
    flags = load_prep_flags(model_dir)
    adv = flags.get("advanced_preprocessing", False)
    size = int(flags.get("image_size", [224, 224])[0]) if isinstance(flags.get("image_size"), list) else 224

    model = tf.keras.models.load_model(str(model_path))
    rows = []
    for p in sorted(input_dir.glob("*")):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        name, conf, _ = predict_image(model, bgr, box, image_size=size, advanced=adv)
        rows.append({"file": p.name, "predicted_class": name, "confidence": conf})

    import pandas as pd

    df = pd.DataFrame(rows)
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print("Wrote", output_csv)
    print(df.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch or single-image PCB defect prediction.")
    parser.add_argument("--model", type=Path, required=True, help="Path to best_model.keras")
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--folder", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument(
        "--box",
        type=float,
        nargs=4,
        default=[0.5, 0.5, 0.45, 0.45],
        metavar=("XC", "YC", "W", "H"),
        help="Normalized YOLO-style box on full image if no label file.",
    )
    args = parser.parse_args()

    if args.image:
        flags = load_prep_flags(args.model.parent.parent)
        adv = flags.get("advanced_preprocessing", False)
        size = int(flags.get("image_size", [224, 224])[0]) if isinstance(flags.get("image_size"), list) else 224
        model = tf.keras.models.load_model(str(args.model))
        bgr = cv2.imread(str(args.image))
        if bgr is None:
            sys.exit("Could not read image")
        name, conf, _ = predict_image(model, bgr, list(args.box), image_size=size, advanced=adv)
        print(f"{name}\t{conf:.4f}")
        return

    if args.folder:
        predict_folder(args.model, args.folder, args.out_csv, list(args.box))
        return

    parser.error("Provide --image or --folder")


if __name__ == "__main__":
    main()
