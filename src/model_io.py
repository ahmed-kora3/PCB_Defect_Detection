"""
Model I/O helpers.

Why this exists:
- Older transfer-learning checkpoints in this repo were saved with `Lambda` layers that
  captured Python callables (e.g., `preprocess_input`) in a way that is not reliably
  deserializable in TF/Keras 2.15+.
- Keras also blocks deserializing Python lambdas by default (safe_mode=True).

This module provides a robust loader that:
1) Tries normal `load_model(..., safe_mode=False)`, and if that fails,
2) Rebuilds the architecture from `hyperparameters.json` and loads weights from the
   `.keras` archive (`model.weights.h5`) without using the serialized graph config.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import tensorflow as tf

from .model import build_scratch_cnn, build_transfer_model


def _read_hyperparams(run_dir: Path) -> dict[str, Any]:
    hp_path = Path(run_dir) / "hyperparameters.json"
    if not hp_path.exists():
        return {}
    return json.loads(hp_path.read_text(encoding="utf-8"))


def _extract_weights_h5_from_keras(keras_path: Path, out_path: Path) -> Path:
    """
    Extract `model.weights.h5` from a `.keras` archive into `out_path`.
    """
    keras_path = Path(keras_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(keras_path, "r") as zf:
        with zf.open("model.weights.h5", "r") as src, open(out_path, "wb") as dst:
            dst.write(src.read())
    return out_path


def _image_size_from_hp(hp: dict[str, Any], default: int = 224) -> int:
    v = hp.get("image_size", default)
    if isinstance(v, (list, tuple)) and v:
        return int(v[0])
    return int(v) if isinstance(v, int) else default


def load_model_for_inference(
    model_path: Path,
    *,
    cache_dir: Path | None = None,
) -> tf.keras.Model:
    """
    Load a `.keras` model checkpoint for inference.

    - Prefers `tf.keras.models.load_model(..., safe_mode=False)` for simplicity.
    - If deserialization fails (common with older Lambda-based transfer models),
      rebuilds the model and loads weights from the `.keras` zip.
    """
    model_path = Path(model_path)
    run_dir = model_path.parent.parent
    hp = _read_hyperparams(run_dir)
    model_kind = str(hp.get("model_kind", "")).lower()
    backbone = str(hp.get("backbone", "MobileNetV2"))
    num_classes = int(hp.get("num_classes", 6))
    size = _image_size_from_hp(hp, default=224)

    try:
        return tf.keras.models.load_model(str(model_path), safe_mode=False)
    except Exception:
        pass

    if model_kind == "scratch":
        model = build_scratch_cnn(input_shape=(size, size, 3), num_classes=num_classes)
    else:
        model = build_transfer_model(input_shape=(size, size, 3), num_classes=num_classes, backbone=backbone)

    cache_root = Path(cache_dir) if cache_dir is not None else (run_dir / ".cache")
    weights_path = cache_root / f"{run_dir.name}__{model_path.stem}.weights.h5"
    if not weights_path.exists():
        _extract_weights_h5_from_keras(model_path, weights_path)

    try:
        model.load_weights(str(weights_path))
    except Exception:
        # HDF5 fallback: be tolerant to minor naming differences.
        model.load_weights(str(weights_path), by_name=True, skip_mismatch=True)

    return model

