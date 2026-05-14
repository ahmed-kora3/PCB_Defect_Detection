"""
Models — CNN from scratch (explicit Functional API, layer-by-layer) and transfer learning
(MobileNetV2 / EfficientNetB0 / ResNet50, frozen backbone + custom head).

Scratch inputs are RGB patches in [0, 1]. Transfer path scales to 0–255 then applies the
official Keras preprocess_input for the chosen backbone (ImageNet normalization only — the
classification dataset remains raw PCB crops, not tf.keras.datasets).

Architecture diagram: save_architecture_diagram() uses keras.utils.plot_model when graphviz/pydot
are installed (optional on Windows).
"""
from __future__ import annotations

from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models


def build_scratch_cnn(input_shape=(224, 224, 3), num_classes: int = 6) -> models.Model:
    """
    Deep CNN built explicitly with the Functional API (no Sequential container).

    Stack: repeated [Conv → BN → ReLU → MaxPool] blocks, then Flatten → Dropout → Dense → ReLU
    → Dropout → softmax classifier. Depth/width chosen for strong patch-level defect features.
    """
    inputs = layers.Input(shape=input_shape, name="input_rgb_patch")

    x = layers.Conv2D(32, 3, padding="same", name="conv2d_1")(inputs)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.ReLU(name="relu_1")(x)
    x = layers.MaxPooling2D(2, name="pool_1")(x)

    x = layers.Conv2D(64, 3, padding="same", name="conv2d_2")(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.ReLU(name="relu_2")(x)
    x = layers.MaxPooling2D(2, name="pool_2")(x)

    x = layers.Conv2D(96, 3, padding="same", name="conv2d_3")(x)
    x = layers.BatchNormalization(name="bn_3")(x)
    x = layers.ReLU(name="relu_3")(x)
    x = layers.MaxPooling2D(2, name="pool_3")(x)

    x = layers.Conv2D(128, 3, padding="same", name="conv2d_4")(x)
    x = layers.BatchNormalization(name="bn_4")(x)
    x = layers.ReLU(name="relu_4")(x)
    x = layers.MaxPooling2D(2, name="pool_4")(x)

    x = layers.Conv2D(256, 3, padding="same", name="conv2d_5")(x)
    x = layers.BatchNormalization(name="bn_5")(x)
    x = layers.ReLU(name="relu_5")(x)
    x = layers.MaxPooling2D(2, name="pool_5")(x)

    x = layers.Conv2D(256, 3, padding="same", name="conv2d_6")(x)
    x = layers.BatchNormalization(name="bn_6")(x)
    x = layers.ReLU(name="relu_6")(x)
    x = layers.MaxPooling2D(2, name="pool_6")(x)

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dropout(0.45, name="dropout_1")(x)
    x = layers.Dense(256, name="dense_1")(x)
    x = layers.ReLU(name="relu_dense_1")(x)
    x = layers.Dropout(0.35, name="dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="softmax_logits")(x)

    model = models.Model(inputs, outputs, name="pcb_scratch_cnn_functional")
    return model


def build_transfer_model(
    input_shape=(224, 224, 3),
    num_classes: int = 6,
    backbone: str = "MobileNetV2",
) -> models.Model:
    """
    Manual transfer learning: frozen ImageNet backbone (include_top=False) + custom head.

    Head (spec): GlobalAveragePooling2D → Dense(128, relu) → Dropout → Dense(softmax).
    """
    if backbone == "MobileNetV2":
        base = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    elif backbone == "ResNet50":
        base = tf.keras.applications.ResNet50(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
        preprocess = tf.keras.applications.resnet50.preprocess_input
    elif backbone == "EfficientNetB0":
        base = tf.keras.applications.EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
        preprocess = tf.keras.applications.efficientnet.preprocess_input
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    base.trainable = False

    inputs = layers.Input(shape=input_shape, name="input_rgb_patch")
    scaled = layers.Lambda(lambda t: t * 255.0, name="scale_to_255")(inputs)
    x = layers.Lambda(lambda t: preprocess(t), name="imagenet_preprocess")(scaled)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu", name="head_dense_128")(x)
    x = layers.Dropout(0.4, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="softmax_logits")(x)

    model = models.Model(inputs, outputs, name=f"pcb_transfer_{backbone}")
    model.backbone = base
    return model


def fine_tune_unfreeze_top_layers(
    model: models.Model,
    fine_tune_at: int,
) -> None:
    """
    Academic note (fine-tuning):
    Lower layers of a CNN pretrained on natural images tend to encode generic edges/textures,
    while deeper layers become more task-specific. For PCB defect patches, we first train only
    the classification head on frozen features, then unfreeze only the *top* backbone layers
    (indices >= fine_tune_at) with a *smaller* learning rate. This adapts high-level filters to
    copper/solder patterns while avoiding catastrophic forgetting in early layers.
    """
    if not hasattr(model, "backbone"):
        raise ValueError("Model has no .backbone attribute (not a transfer model).")
    model.backbone.trainable = True
    for i, layer in enumerate(model.backbone.layers):
        layer.trainable = i >= fine_tune_at


def save_architecture_diagram(model: models.Model, output_path: Path, dpi: int = 150) -> bool:
    """Write a PNG topology diagram if graphviz + pydot are available."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        tf.keras.utils.plot_model(
            model,
            to_file=str(output_path),
            show_shapes=True,
            show_layer_names=True,
            dpi=dpi,
        )
        return True
    except Exception:
        return False


def print_model_summary_and_params(model: models.Model) -> int:
    model.summary()
    n = int(model.count_params())
    print(f"Total trainable + non-trainable parameters: {n:,}")
    return n
