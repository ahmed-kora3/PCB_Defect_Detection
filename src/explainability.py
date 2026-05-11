"""
Explainability — Grad-CAM overlays and first-layer filter visualization (scratch CNN).

Transfer backbones use DepthwiseConv2D / SeparableConv2D; we pick the last spatial conv-like layer.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def _last_spatial_conv(model: tf.keras.Model) -> tf.keras.layers.Layer:
    scan_models = [model]
    if hasattr(model, "backbone"):
        scan_models.insert(0, model.backbone)

    conv_types = (
        tf.keras.layers.Conv2D,
        tf.keras.layers.DepthwiseConv2D,
        tf.keras.layers.SeparableConv2D,
    )
    for m in scan_models:
        for layer in reversed(m.layers):
            if isinstance(layer, conv_types):
                return layer
    raise ValueError("No conv-like layer found for Grad-CAM.")


def make_gradcam_heatmap(
    model: tf.keras.Model,
    img_array: np.ndarray,
    pred_index: int | None = None,
    conv_layer: tf.keras.layers.Layer | None = None,
) -> np.ndarray:
    """
    img_array: float32 batch (1, H, W, 3) in [0,1] RGB as used at inference.
    Returns 2D heatmap normalized to [0, 1].
    """
    layer = conv_layer or _last_spatial_conv(model)
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array, training=False)
        if pred_index is None:
            pred_index = int(tf.argmax(preds[0]))
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_out), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


def overlay_gradcam_on_image(
    img_rgb_uint8: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    import cv2

    h, w = img_rgb_uint8.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    jet = plt.cm.jet(heatmap_uint8 / 255.0)[..., :3]
    jet = np.uint8(255 * jet)
    superimposed = jet * alpha + img_rgb_uint8 * (1 - alpha)
    return np.clip(superimposed, 0, 255).astype(np.uint8)


def save_gradcam(
    model: tf.keras.Model,
    img_batch: np.ndarray,
    out_path: Path,
) -> None:
    """img_batch: (1,224,224,3) float32 [0,1] RGB."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hm = make_gradcam_heatmap(model, img_batch)
    rgb_uint8 = np.clip(img_batch[0] * 255.0, 0, 255).astype(np.uint8)
    overlay = overlay_gradcam_on_image(rgb_uint8, hm)
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.axis("off")
    plt.title("Grad-CAM")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_first_layer_filters(
    model: tf.keras.Model,
    out_path: Path,
    max_filters: int = 32,
) -> None:
    """Save a grid of first Conv2D filters (scratch CNN explainability)."""
    conv = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            conv = layer
            break
    if conv is None:
        return
    w = conv.get_weights()[0]
    k = min(max_filters, w.shape[-1])
    side = int(np.ceil(np.sqrt(k)))
    fig, axes = plt.subplots(side, side, figsize=(side * 1.2, side * 1.2))
    axes = np.atleast_1d(axes).ravel()
    for i in range(side * side):
        ax = axes[i]
        if i < k:
            fmap = np.mean(np.abs(w[..., i]), axis=-1)
            ax.imshow(fmap, cmap="viridis")
        ax.axis("off")
    plt.suptitle("First conv kernel magnitude (mean abs across input channels)")
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
