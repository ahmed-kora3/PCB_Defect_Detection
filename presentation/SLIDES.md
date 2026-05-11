---
marp: true
theme: default
paginate: true
header: PCB Defect Detection — Graduation Project
footer: Image Processing & Computer Vision
---

# PCB Defect Detection and Classification

Deep Learning · YOLO label processing · CNN from scratch · Transfer learning · Deployment

---

## Problem overview

- **Goal:** Automate visual inspection of PCBs for manufacturing defects.
- **Input:** Raw PCB images + **YOLO** `.txt` annotations (`class xc yc w h`, normalized).
- **Output:** Defect **class** per cropped region (6 classes).

---

## Dataset samples

- Real **Kaggle** PCB defect dataset; **>1000** images; train / val / test splits.
- **No** MNIST / CIFAR / built-in toy loaders — only manual disk I/O.

*(Insert figure: `results/visualizations/random_pcb_grid_with_boxes.png` after running EDA.)*

---

## YOLO visualization

- Normalized coordinates → **pixel** corners.
- Draw **boxes** + **class names** on RGB images.
- Random **grid** export for the report.

*(Insert: `results/yolo_vis/grid_train.png` or EDA grid.)*

---

## Preprocessing pipeline

1. Match image path to label (`_256` / `_600` stem variants).
2. Skip **corrupted** images when building manifest.
3. **Crop** YOLO box → **224×224** RGB, **[0,1]** float32.
4. **Augment:** flip, rotate, brightness, contrast, mild zoom.
5. **`tf.data`:** shuffle · optional cache · batch · prefetch.

---

## CNN architecture (scratch)

- **Functional API**, layer-by-layer:  
  **Conv → BN → ReLU → MaxPool** × multiple blocks → **Flatten** → **Dropout** → **Dense** → **Softmax**.
- Deep enough for patch texture; regularized with **Dropout** + augmentation.

*(Insert: `results/scratch_cnn/architecture_diagram.png` if Graphviz installed.)*

---

## Transfer learning architecture

- **MobileNetV2** or **ResNet50**, `include_top=False`, frozen initially.
- Scale **[0,1] → [0,255]** + correct **`preprocess_input`**.
- Head: **GAP → Dense(128) → Dropout → Softmax**.

---

## Fine-tuning

- Unfreeze **top** backbone layers only (`fine_tune_at`).
- **Smaller LR** (`1e-5`) after head training.
- **Why:** adapt deep features to PCB appearance without destroying low-level filters.

---

## Training curves

- **Loss vs epoch** (train / val).
- **Accuracy vs epoch**.
- Callbacks: **EarlyStopping**, **ModelCheckpoint**, **ReduceLROnPlateau**, optional **TensorBoard**.

*(Insert: `results/transfer_MobileNetV2/loss_vs_epoch.png` and `accuracy_vs_epoch.png`.)*

---

## Confusion matrix

- Test-set **heatmap** for multi-class confusion.
- Diagonal dominance → good separation.

*(Insert: `results/transfer_MobileNetV2/confusion_matrix_heatmap.png`.)*

---

## Accuracy & F1 results

- Report **test accuracy**, **macro / weighted F1** from `metrics.json`.
- Optional **ROC / AUC** (OvR) for bonus analysis.

*(Insert table from `classification_report.txt` or screenshot of Streamlit metrics.)*

---

## Final model comparison

| Model | Test Acc | Best Val Acc | F1 (w) | Params | Time (s) |
|-------|----------|--------------|--------|--------|----------|
| Scratch CNN | … | … | … | … | … |
| Transfer | … | … | … | … | … |

*(Fill from `results/model_comparison.csv` after `python train.py --model all`.)*

---

## Streamlit deployment

- Upload PCB image → **center YOLO-style crop** → predict class.
- Show **confidence** (max softmax) + probability bar chart.
- Professional layout + sidebar explanation.

*(Insert screenshots of the running app.)*

---

## Thank you

**Q&A — viva**  
Repository: `README.md` · Code: `src/` · Report: `report/REPORT.md`
