"""
Streamlit deployment — upload a PCB image, run patch-level defect classification.

Training uses YOLO boxes to crop defect patches; this demo uses the same crop_and_resize logic
with a **YOLO-style normalized box** estimated from the image center so the pipeline matches
the academic preprocessing story (full detector would require a trained YOLO inference pass).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

from src.data_loader import CLASS_NAMES, NUM_CLASSES
from src.preprocess import crop_and_resize

PROJECT = Path(__file__).resolve().parent


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
    return tf.keras.models.load_model(path_str)


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
        "Deep learning on raw Kaggle PCB data — YOLO labels drive **training crops**; "
        "this app applies the **same crop → 224×224 → normalize** pipeline on an uploaded image."
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
    with c1:
        uploaded = st.file_uploader("Upload PCB image (JPG / PNG)", type=["jpg", "jpeg", "png"])
    if not uploaded:
        st.info("Upload an image to see prediction and confidence.")
        return

    data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("Could not decode the image file.")
        return

    h, w = bgr.shape[:2]
    box_w = min(0.45, img_side / max(w, 1))
    box_h = min(0.45, img_side / max(h, 1))
    box = [0.5, 0.5, box_w, box_h]
    patch = crop_and_resize(bgr, box, (img_side, img_side), advanced_preprocessing=adv)
    batch = np.expand_dims(patch, axis=0)
    probs = model.predict(batch, verbose=0)[0]
    pred_id = int(np.argmax(probs))
    confidence = float(probs[pred_id])

    with c2:
        st.subheader("Uploaded (full frame)")
        st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.subheader("Model input (center crop, YOLO-style box)")
    st.image(patch, use_container_width=True)

    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Predicted defect class", CLASS_NAMES[pred_id])
    st.metric("Confidence (max softmax probability)", f"{confidence * 100:.2f}%")
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Class probabilities")
    st.bar_chart({CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)})


if __name__ == "__main__":
    main()
