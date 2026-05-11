"""
Central configuration — paths, GPU/CPU policy, and backbone-specific fine-tune defaults.

Production deployments should set PCB_DATA_ROOT via environment variable if the dataset is not
under the default relative path.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Primary dataset location (Kaggle PCB layout). Alias `datasets/pcb-defect-dataset` optional via symlink/copy.
DEFAULT_DATA_ROOT = PROJECT_ROOT / "dataset" / "pcb-defect-dataset"
DATA_ROOT = Path(os.environ.get("PCB_DATA_ROOT", DEFAULT_DATA_ROOT))

MODELS_REGISTRY = PROJECT_ROOT / "models" / "registry"
RESULTS_ROOT = PROJECT_ROOT / "results"

# Default layer index from which to unfreeze during fine-tuning (backbone-dependent).
FINETUNE_LAYER_DEFAULTS = {
    "MobileNetV2": 100,
    "ResNet50": 140,
    "EfficientNetB0": 200,
}


def configure_runtime(
    mixed_precision: bool = False,
    disable_gpu_growth: bool = False,
) -> None:
    """
    GPU: enable memory growth to avoid OOM; optional mixed precision for faster inference on GPU.
    CPU: TensorFlow falls back automatically when no CUDA device is visible.
    """
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            if not disable_gpu_growth:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    if mixed_precision and gpus:
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
        except Exception:
            pass
