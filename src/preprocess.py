"""
Preprocessing — manual OpenCV read, BGR→RGB, YOLO-box crop, resize 224×224, normalize [0,1],
TensorFlow tf.data with prefetch; optional in-memory cache for val/test; augmentation on train.

Augmentation (train): horizontal/vertical flip, 90° rotation family, brightness, contrast, optional zoom.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import tensorflow as tf

from .data_loader import CLASS_NAMES, build_annotation_manifest


def enhance_bgr_patch(patch_bgr: np.ndarray, denoise: bool = True, clahe_contrast: bool = True) -> np.ndarray:
    """
    Production-style enhancement on raw BGR crop before resize:
    - fastNlMeans denoising (mild) to suppress sensor noise
    - CLAHE on L channel (LAB) for local contrast on copper/solder
    """
    if patch_bgr is None or patch_bgr.size == 0:
        return patch_bgr
    if denoise and patch_bgr.shape[0] >= 4 and patch_bgr.shape[1] >= 4:
        patch_bgr = cv2.fastNlMeansDenoisingColored(
            patch_bgr, None, h=3, hColor=3, templateWindowSize=7, searchWindowSize=15
        )
    if clahe_contrast:
        lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        patch_bgr = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)
    return patch_bgr


def crop_and_resize(
    image: np.ndarray,
    bbox: List[float],
    target_size: Tuple[int, int],
    advanced_preprocessing: bool = False,
) -> np.ndarray:
    """
    Crop normalized YOLO box from BGR uint8 image; invalid boxes fall back to full frame.
    Output: RGB float32 in [0, 1], shape target_size.
    """
    h, w = image.shape[:2]
    x_center, y_center, width, height = bbox
    x_center *= w
    y_center *= h
    crop_w = width * w
    crop_h = height * h
    x1 = int(round(x_center - crop_w / 2))
    y1 = int(round(y_center - crop_h / 2))
    x2 = int(round(x_center + crop_w / 2))
    y2 = int(round(y_center + crop_h / 2))

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if y1 >= y2 or x1 >= x2:
        patch = image
    else:
        patch = image[y1:y2, x1:x2]

    if advanced_preprocessing:
        patch = enhance_bgr_patch(patch, denoise=True, clahe_contrast=True)

    patch = cv2.resize(patch, target_size, interpolation=cv2.INTER_AREA)
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    return patch.astype("float32") / 255.0


def augment_image(image: tf.Tensor) -> tf.Tensor:
    """
    Train-time stochastic transforms (image in [0,1], channels-last, 224×224).
    Includes optional mild zoom via resize + crop back to 224 (fixed spatial size).
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    image = tf.image.rot90(image, k=k)
    image = tf.image.random_brightness(image, max_delta=0.18)
    image = tf.image.random_contrast(image, lower=0.82, upper=1.18)
    scale = tf.random.uniform([], minval=0.94, maxval=1.08, dtype=tf.float32)
    side = tf.cast(tf.maximum(tf.round(224.0 * scale), 1.0), tf.int32)
    image = tf.image.resize(image, [side, side], method=tf.image.ResizeMethod.BILINEAR)
    image = tf.image.resize_with_crop_or_pad(image, 224, 224)
    return tf.clip_by_value(image, 0.0, 1.0)


def _sample_generator(
    manifest: Iterable[dict],
    image_size: Tuple[int, int],
    num_classes: int,
    advanced_preprocessing: bool,
):
    for record in manifest:
        image_path = Path(record["image_path"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            continue
        try:
            patch = crop_and_resize(
                image,
                record["bbox"],
                image_size,
                advanced_preprocessing=advanced_preprocessing,
            )
        except cv2.error:
            continue
        y = tf.keras.utils.to_categorical(int(record["class_id"]), num_classes=num_classes)
        yield patch, y


def create_tf_dataset(
    manifest: List[dict],
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    shuffle: bool = True,
    augment: bool = False,
    num_classes: int | None = None,
    cache_in_memory: bool = False,
    advanced_preprocessing: bool = False,
) -> tf.data.Dataset:
    n = num_classes if num_classes is not None else len(CLASS_NAMES)
    output_signature = (
        tf.TensorSpec(shape=(image_size[0], image_size[1], 3), dtype=tf.float32),
        tf.TensorSpec(shape=(n,), dtype=tf.float32),
    )
    dataset = tf.data.Dataset.from_generator(
        lambda: _sample_generator(manifest, image_size, n, advanced_preprocessing),
        output_signature=output_signature,
    )
    if shuffle:
        buf = min(4096, max(256, len(manifest)))
        dataset = dataset.shuffle(buffer_size=buf, reshuffle_each_iteration=True)
    if augment:
        dataset = dataset.map(
            lambda x, y: (augment_image(x), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    if cache_in_memory:
        dataset = dataset.cache()
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def build_datasets(
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    augment: bool = False,
    cache_val_test: bool = True,
    cache_train: bool = False,
    advanced_preprocessing: bool = False,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    train_manifest = build_annotation_manifest("train")
    val_manifest = build_annotation_manifest("val")
    test_manifest = build_annotation_manifest("test")
    n = len(CLASS_NAMES)

    train_ds = create_tf_dataset(
        train_manifest,
        image_size,
        batch_size,
        shuffle=True,
        augment=augment,
        num_classes=n,
        cache_in_memory=cache_train,
        advanced_preprocessing=advanced_preprocessing,
    )
    val_ds = create_tf_dataset(
        val_manifest,
        image_size,
        batch_size,
        shuffle=False,
        augment=False,
        num_classes=n,
        cache_in_memory=cache_val_test,
        advanced_preprocessing=advanced_preprocessing,
    )
    test_ds = create_tf_dataset(
        test_manifest,
        image_size,
        batch_size,
        shuffle=False,
        augment=False,
        num_classes=n,
        cache_in_memory=cache_val_test,
        advanced_preprocessing=advanced_preprocessing,
    )
    return train_ds, val_ds, test_ds
