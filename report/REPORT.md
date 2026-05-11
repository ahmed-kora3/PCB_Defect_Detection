# PCB Defect Detection and Classification using Deep Learning

**Authors:** [Your Name]  
**Course:** Image Processing & Computer Vision — Graduation Project  
**Date:** 2026  

---

## 1. Introduction

Printed Circuit Boards (PCBs) are central to electronics manufacturing. Automatic visual inspection reduces human error and throughput bottlenecks. This project implements an **industrial-style pipeline**: raw images and **YOLO-format** bounding-box labels from a real **Kaggle** PCB defect dataset are processed manually (OpenCV + file parsing), defect patches are classified with a **CNN built from scratch** and a **transfer learning** model (MobileNetV2 / ResNet50), evaluated with standard metrics, and packaged with a **Streamlit** demo.

---

## 2. Problem Statement

Given a PCB photograph and weak supervision in the form of **axis-aligned defect boxes**, the system must:

1. Reliably **associate** each label file with its image despite filename stem mismatches (`_256`, `_600`, etc.).
2. **Crop** each defect region, normalize it, and **classify** it into one of six defect categories.
3. **Compare** a scratch CNN against transfer learning under the same preprocessing and metrics.

---

## 3. Dataset Description

- **Source:** Kaggle — Printed Circuit Board (PCB) Defects (real images + raw `.txt` labels).
- **Scale:** Thousands of images and >10,000 annotation instances (train/val/test splits provided by the dataset).
- **Layout:**

```
dataset/pcb-defect-dataset/
  data.yaml
  train/images/  train/labels/
  val/images/     val/labels/
  test/images/    test/labels/
```

- **Classes** (from `data.yaml`): mouse_bite, spur, missing_hole, short, open_circuit, spurious_copper.

**Figures (generate with `python -m src.data_analysis`):**

- Class distribution: `../results/visualizations/class_distribution.png`
- Random PCB + YOLO overlays: `../results/visualizations/random_pcb_grid_with_boxes.png`

---

## 4. YOLO Annotation Format

Each line in a label file:

`class_id x_center y_center width height`

All coordinates are **normalized** to \([0,1]\) relative to image width and height. The implementation converts them to pixel corners for drawing and cropping (see `src/yolo_visualization.py`, function `yolo_norm_to_pixel_xyxy`).

---

## 5. Data Preprocessing

1. **Read** BGR image with OpenCV; **skip** unreadable files when building the manifest (`verify_image_readable`).
2. **Crop** using the YOLO box; if the box is degenerate, **fallback** to the full image then resize.
3. **Resize** to **224×224**, **BGR→RGB**, divide by 255 → **float32 in [0,1]**.
4. **Augmentation (train):** flips, 90° rotations, brightness, contrast, mild zoom-resize-crop.
5. **`tf.data`:** shuffle, optional cache, **batch**, **prefetch** (`AUTOTUNE`).

Code: `src/preprocess.py`.

---

## 6. CNN Architecture

A **deep functional CNN** is defined **layer-by-layer** (no `Sequential` stack): repeated blocks of **Conv2D → BatchNormalization → ReLU → MaxPooling2D**, followed by **Flatten → Dropout → Dense → ReLU → Dropout → Softmax**.

Code: `build_scratch_cnn()` in `src/model.py`.  
Optional topology figure after training: `../results/scratch_cnn/architecture_diagram.png` (requires Graphviz/Pydot).

---

## 7. Transfer Learning

A frozen **MobileNetV2** or **ResNet50** backbone (`include_top=False`, ImageNet weights) extracts features. Inputs in \([0,1]\) are scaled to **0–255** and passed through the official **`preprocess_input`**. The custom head is:

**GlobalAveragePooling2D → Dense(128, ReLU) → Dropout → Dense(softmax).**

The backbone is attached as **`model.backbone`** for fine-tuning.

Code: `build_transfer_model()` in `src/model.py`.

---

## 8. Fine-Tuning

After the head converges with a frozen backbone, **upper backbone layers** (indices \(\geq\) `fine_tune_at`, default 100) are **unfrozen** and trained with a **lower learning rate** (default `1e-5`). Early layers keep generic low-level filters; deeper layers adapt to copper, pads, and solder patterns.

Code: `fine_tune_unfreeze_top_layers()` in `src/model.py`; CLI: `python train.py --fine-tune --fine-tune-at 100 --fine-tune-lr 1e-5`.

---

## 9. Evaluation Metrics

On the held-out **test** split:

- Accuracy  
- Precision / Recall / **F1** (macro + weighted)  
- **Confusion matrix** heatmap  
- **Classification report** (per-class)  
- **Bonus:** multi-class ROC (OvR) and AUC summary (`roc_multiclass.png`, `roc_auc_summary.json`)

Code: `src/evaluate.py` (also invoked at the end of each training run).

---

## 10. Results

After training, each run directory (e.g. `results/scratch_cnn/`, `results/transfer_MobileNetV2/`) contains:

| Artifact | Description |
|----------|-------------|
| `history.json` | Loss & accuracy per epoch |
| `accuracy_vs_epoch.png` | Curves |
| `loss_vs_epoch.png` | Curves |
| `confusion_matrix_heatmap.png` | Test confusion matrix |
| `metrics.json` | Aggregated test metrics |
| `classification_report.txt` | sklearn report |
| `models/best_model.keras` | Best `val_accuracy` checkpoint |
| `models/final_model.keras` | Weights after full training schedule |

**Note on “99%” targets:** reported accuracy depends on data noise, class similarity, and train duration. The pipeline applies **class weights**, **augmentation**, **ReduceLROnPlateau**, **EarlyStopping**, and **fine-tuning** to approach the **maximum achievable** performance; exact numbers are filled in **your** `metrics.json` after you run training on your machine.

---

## 11. Model Comparison

Run both models:

```bash
python train.py --model all --backbone MobileNetV2 --fine-tune
```

Outputs:

- `../results/model_comparison.csv`
- `../results/model_comparison.md`

Columns include **test accuracy**, **best validation accuracy**, **F1 (weighted)**, **parameter count**, and **training time**.

---

## 12. Conclusion

The project satisfies end-to-end requirements: **manual** handling of raw files, **YOLO** parsing and visualization, **patch-based** preprocessing, **CNN from scratch** (functional), **transfer learning** with **fine-tuning**, rich **evaluation**, optional **ROC/AUC**, and **Streamlit** deployment.

---

## 13. Future Work

- Train a full **YOLOv8** detector on the same folders (Ultralytics) for joint detection + classification.
- Semi-supervised or **active learning** for rare defect types.
- **TensorRT / ONNX** export for factory-edge deployment.
- **Grad-CAM** for explainability in the viva.

---

## Appendix: Reproducibility

```bash
pip install -r requirements.txt
python -m src.data_analysis
python train.py --model all --epochs 25 --fine-tune --tensorboard
streamlit run app_streamlit.py
python -m src.evaluate --run-dir results/scratch_cnn
```

Replace epochs and flags to match your hardware and deadline.
