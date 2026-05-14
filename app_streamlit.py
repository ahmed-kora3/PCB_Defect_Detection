"""
Streamlit deployment — upload a PCB image, run patch-level defect classification.

Training uses YOLO boxes to crop defect patches; this demo can get boxes from:
- the dataset YOLO label file (GT boxes), or
- a center crop fallback when no label file is provided.

Note: true *detection* (predicting boxes for unseen images) would require training and running an
object detector (e.g., YOLO) at inference time. This app focuses on classification of cropped patches.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

from src.data_loader import CLASS_NAMES, NUM_CLASSES, find_label_path_for_image
from src.config import DATA_ROOT
from src.model_io import load_model_for_inference
from src.preprocess import crop_and_resize

PROJECT = Path(__file__).resolve().parent


def parse_yolo_text(text: str) -> list[tuple[int, float, float, float, float]]:
    rows: list[tuple[int, float, float, float, float]] = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) != 5:
            continue
        try:
            rows.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
        except ValueError:
            continue
    return rows


def yolo_norm_to_pixel_xyxy(
    xc: float,
    yc: float,
    bw: float,
    bh: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    xc_px = xc * image_width
    yc_px = yc * image_height
    w_px = bw * image_width
    h_px = bh * image_height
    x1 = int(round(xc_px - w_px / 2))
    y1 = int(round(yc_px - h_px / 2))
    x2 = int(round(xc_px + w_px / 2))
    y2 = int(round(yc_px + h_px / 2))
    x1 = max(0, min(image_width - 1, x1))
    x2 = max(0, min(image_width - 1, x2))
    y1 = max(0, min(image_height - 1, y1))
    y2 = max(0, min(image_height - 1, y2))
    return x1, y1, x2, y2


def draw_predicted_boxes(
    bgr: np.ndarray,
    yolo_rows: list[tuple[int, float, float, float, float]],
    pred_names: list[str] | None = None,
    pred_confs: list[float] | None = None,
) -> np.ndarray:
    out = bgr.copy()
    h, w = out.shape[:2]
    for i, (gt_id, xc, yc, bw, bh) in enumerate(yolo_rows):
        x1, y1, x2, y2 = yolo_norm_to_pixel_xyxy(xc, yc, bw, bh, w, h)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)

        gt = CLASS_NAMES.get(int(gt_id), str(gt_id))
        if pred_names is not None and pred_confs is not None and i < len(pred_names):
            label = f"pred:{pred_names[i]} ({pred_confs[i]*100:.1f}%) | gt:{gt}"
        else:
            label = f"gt:{gt}"

        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(0, y1 - 6)
        cv2.rectangle(out, (x1, max(0, y_text - th - base)), (x1 + tw + 6, y_text + base), (0, 0, 0), -1)
        cv2.putText(
            out,
            label,
            (x1 + 3, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


@st.cache_data
def list_dataset_images(split: str) -> list[str]:
    img_dir = DATA_ROOT / split / "images"
    if not img_dir.exists():
        return []
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    files: list[Path] = []
    for pat in exts:
        files.extend(img_dir.glob(pat))
    return [str(p) for p in sorted(files)]


def load_run_hyperparams(model_path: Path) -> dict:
    run_dir = model_path.parent.parent
    hp = run_dir / "hyperparameters.json"
    if hp.exists():
        return json.loads(hp.read_text(encoding="utf-8"))
    return {"advanced_preprocessing": False, "image_size": [224, 224]}


def find_latest_model() -> Path | None:
    candidates = sorted(
        PROJECT.glob("results/**/models/best_model.keras"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    candidates = sorted(
        PROJECT.glob("results/**/models/*.keras"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


@st.cache_resource
def load_model_cached(path_str: str):
    # Robust loader: supports legacy Lambda-based checkpoints by rebuilding the model
    # and loading `model.weights.h5` from the `.keras` archive if needed.
    return load_model_for_inference(Path(path_str))


def main() -> None:
    st.set_page_config(page_title="PCB Defect AI", layout="wide", initial_sidebar_state="expanded")

    st.markdown(
        """
        <style>
        .main-title { font-size: 2rem; font-weight: 700; color: #1a237e; }
        .metric-card { background: #f5f7ff; padding: 1rem; border-radius: 8px; border: 1px solid #c5cae9; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<p class="main-title">PCB Defect Detection & Classification</p>', unsafe_allow_html=True)
    st.caption(
        "Deep learning on raw Kaggle PCB data — YOLO labels drive **training crops**. "
        "For the demo: use **dataset mode** (GT labels) or upload a YOLO `.txt` label; otherwise it falls back to a **center crop**."
    )

    with st.sidebar:
        st.header("Model")
        model_path = find_latest_model()
        if model_path is None:
            st.error("No `.keras` weights under `results/`. Train first:")
            st.code("python train.py --model transfer --epochs 20 --fine-tune", language="bash")
            return
        hp0 = load_run_hyperparams(model_path)
        adv0 = bool(hp0.get("advanced_preprocessing", False))
        st.success(f"Loaded\n`{model_path.relative_to(PROJECT)}`")
        st.markdown("---")
        st.markdown("**YOLO-assisted preprocessing (training)**")
        st.markdown(
            "- Parse `class x y w h` (normalized)\n"
            "- Crop defect patch → RGB [0,1] (size from training)\n"
            "- Optional denoise + CLAHE if trained with `--advanced-prep`"
        )
        if adv0:
            st.warning("This checkpoint uses **advanced prep** (denoise + CLAHE).")

    model = load_model_cached(str(model_path))
    hp = load_run_hyperparams(model_path)
    adv = bool(hp.get("advanced_preprocessing", False))
    isize = hp.get("image_size", [224, 224])
    img_side = int(isize[0]) if isinstance(isize, list) else 224

    c1, c2 = st.columns([1, 1])

    has_dataset = (DATA_ROOT / "train").exists() or (DATA_ROOT / "val").exists() or (DATA_ROOT / "test").exists()
    with st.sidebar:
        st.header("Input")
        if has_dataset:
            input_source = st.radio("Source", ["Upload image", "Pick from dataset"], index=0)
        else:
            input_source = "Upload image"
            st.info("Dataset not found under `dataset/pcb-defect-dataset/`. Upload an image manually.")

    bgr = None
    yolo_rows: list[tuple[int, float, float, float, float]] | None = None

    if input_source == "Pick from dataset":
        with st.sidebar:
            split = st.selectbox("Split", ["test", "val", "train"], index=0)
            files = list_dataset_images(split)
            if not files:
                st.error(f"No images found under `{(DATA_ROOT / split / 'images').as_posix()}`")
                return
            st.caption(f"Found {len(files)} images")
            idx = st.number_input("Image index", min_value=0, max_value=len(files) - 1, value=0, step=1)
            image_path = Path(files[int(idx)])
            st.success(f"Selected\n`{image_path.relative_to(PROJECT) if PROJECT in image_path.parents else image_path}`")

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            st.error("Could not read the selected dataset image.")
            return
        label_path = find_label_path_for_image(image_path, split)
        if label_path is not None and label_path.exists():
            yolo_rows = parse_yolo_text(label_path.read_text(encoding="utf-8"))
        else:
            yolo_rows = None

    else:
        with c1:
            uploaded = st.file_uploader("Upload PCB image (JPG / PNG)", type=["jpg", "jpeg", "png"])
            label_upload = st.file_uploader(
                "Optional: upload YOLO label (.txt) to use real boxes (detection-style demo)",
                type=["txt"],
            )
        if not uploaded:
            st.info("Upload an image to see prediction and confidence.")
            return

        data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            st.error("Could not decode the image file.")
            return

        if label_upload is not None:
            try:
                yolo_rows = parse_yolo_text(label_upload.getvalue().decode("utf-8", errors="ignore"))
            except Exception:
                yolo_rows = None
        else:
            # If the user uploaded an image from the dataset folders, we can try to auto-locate its label by filename.
            stem = Path(uploaded.name).stem
            for sp in ("test", "val", "train"):
                lp = find_label_path_for_image(Path(stem), sp)
                if lp is not None and lp.exists():
                    yolo_rows = parse_yolo_text(lp.read_text(encoding="utf-8"))
                    break

    h, w = bgr.shape[:2]

    # If we have YOLO rows, treat this as "detection-style" (GT boxes) and classify each defect patch.
    # Otherwise, fall back to a center crop demo.
    if yolo_rows:
        patches: list[np.ndarray] = []
        for _, xc, yc, bw, bh in yolo_rows:
            patches.append(crop_and_resize(bgr, [xc, yc, bw, bh], (img_side, img_side), advanced_preprocessing=adv))
        batch = np.stack(patches, axis=0) if patches else np.empty((0, img_side, img_side, 3), dtype=np.float32)
        probs_all = model.predict(batch, verbose=0) if len(patches) else np.empty((0, NUM_CLASSES), dtype=np.float32)
        pred_ids = np.argmax(probs_all, axis=-1).astype(int).tolist() if len(patches) else []
        pred_names = [CLASS_NAMES[i] for i in pred_ids]
        pred_confs = [float(probs_all[i, pred_ids[i]]) for i in range(len(pred_ids))] if len(pred_ids) else []
        selected = 0
        if len(patches) > 1:
            selected = int(st.selectbox("Select defect instance", list(range(len(patches))), index=0))
        patch = patches[selected] if patches else None
        probs_sel = probs_all[selected] if len(patches) else None
        pred_id = pred_ids[selected] if pred_ids else 0
        confidence = pred_confs[selected] if pred_confs else 0.0

        vis_bgr = draw_predicted_boxes(bgr, yolo_rows, pred_names=pred_names, pred_confs=pred_confs)
    else:
        box_w = min(0.45, img_side / max(w, 1))
        box_h = min(0.45, img_side / max(h, 1))
        box = [0.5, 0.5, box_w, box_h]
        patch = crop_and_resize(bgr, box, (img_side, img_side), advanced_preprocessing=adv)
        batch = np.expand_dims(patch, axis=0)
        probs = model.predict(batch, verbose=0)[0]
        pred_id = int(np.argmax(probs))
        confidence = float(probs[pred_id])
        vis_bgr = bgr

    with c2:
        st.subheader("Uploaded (full frame)")
        st.image(cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

    if yolo_rows:
        st.subheader("Model input (YOLO GT crop)")
        if patch is not None:
            st.image(patch, use_container_width=True)
    else:
        st.subheader("Model input (center crop, YOLO-style box)")
        st.image(patch, use_container_width=True)

    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Predicted defect class", CLASS_NAMES[pred_id])
    st.metric("Confidence (max softmax probability)", f"{confidence * 100:.2f}%")
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Class probabilities")
    if yolo_rows:
        # Show probabilities for the selected instance.
        if probs_sel is not None:
            st.bar_chart({CLASS_NAMES[i]: float(probs_sel[i]) for i in range(NUM_CLASSES)})
    else:
        st.bar_chart({CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)})


if __name__ == "__main__":
    main()
