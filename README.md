# PCB Defect Detection & Classification

End-to-end **production-style** PCB defect AI: multi-class classification on **YOLO-cropped** patches,
**custom Functional CNN**, **transfer learning** (MobileNetV2, **EfficientNetB0**, ResNet50),
**fine-tuning**, strong **augmentation**, optional **denoise + CLAHE**, **Grad-CAM** explainability,
CSV **training logs**, **TensorBoard**, evaluation + ROC, **Streamlit** demo app, and auto-generated report.

> Raw Kaggle images + labels only. ImageNet **weights** initialize backbones only.

---

## Repository layout

```
PCB_Defect_Detection/
├── dataset/                 # PCB dataset root (NOT in repo — must be added manually)
├── datasets/                # README only
├── models/registry/         # Place exported checkpoints here
├── src/
│   ├── config.py            # Paths, GPU setup, fine-tune defaults
│   ├── data_loader.py       # YOLO manifest, class weights
│   ├── data_analysis.py     # EDA plots
│   ├── preprocess.py        # Crop, denoise/CLAHE, augment, tf.data
│   ├── model.py             # Scratch CNN + Transfer Learning models
│   ├── train.py             # Full training pipeline
│   ├── evaluate.py          # Metrics, curves, ROC
│   ├── explainability.py    # Grad-CAM, first-layer filters
│   ├── predict.py           # Single-image / folder inference
│   └── report_generator.py  # Generates final report
├── scripts/
│   └── draw_boxes.py        # Visualize YOLO bounding boxes
├── notebooks/
├── results/                 # Training outputs (auto-created per run)
├── report/
├── presentation/
├── app_streamlit.py         # Streamlit web app
├── train.py                 # Main training entry point
├── requirements.txt
└── README.md
```

---

## How to Run the Project (Step by Step)

### Step 1 - Clone the repository

```bash
git clone https://github.com/ahmed-kora3/PCB_Defect_Detection.git
cd PCB_Defect_Detection
```

---

### Step 2 - Create a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### Step 3 - Install dependencies

```bash
pip install -r requirements.txt
```

> This installs TensorFlow 2.15.1 (pinned for Windows compatibility).
> Do NOT upgrade TensorFlow manually.

---

### Step 4 - Add the dataset

Download the **PCB Defects** dataset from Kaggle and place it in this exact structure:

```
dataset/
└── pcb-defect-dataset/
    ├── data.yaml
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

> The dataset folder is NOT included in the repository (too large).
> Ask a team member to share it or download from Kaggle.

---

### Step 5 - Train the model

```bash
# Recommended: EfficientNetB0 with fine-tuning
python train.py --model transfer --backbone EfficientNetB0 --epochs 30 --fine-tune

# Scratch CNN only
python train.py --model scratch --epochs 25

# Train all backbones and generate comparison table
python train.py --train-all-backbones --epochs 25 --fine-tune
```

Training outputs (model weights, metrics, logs) are saved inside `results/`.

---

### Step 6 - Launch the Streamlit app

```bash
streamlit run app_streamlit.py
```

Then open your browser at **http://localhost:8501** and upload a PCB image.

> You must train the model first (Step 5) before running the app.
> The app looks for `.keras` model files inside `results/`.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `DLL load failed` on Windows | Install VC++ Redistributable x64 from Microsoft |
| TensorFlow import error | `pip uninstall -y tensorflow keras` then `pip install -r requirements.txt` |
| `No .keras weights under results/` | Train the model first using Step 5 above |
| `No module named streamlit` | Make sure venv is activated and requirements are installed |
| oneDNN warning messages | Set env var `TF_ENABLE_ONEDNN_OPTS=0` (cosmetic only) |

---

## Other useful commands

```bash
# Exploratory data analysis
python -m src.data_analysis

# Evaluate a trained model
python -m src.evaluate --run-dir results/transfer_EfficientNetB0

# Predict on a single image
python -m src.predict --model results/transfer_EfficientNetB0/models/best_model.keras --image sample.jpg

# Generate final report
python -m src.report_generator --results-dir results

# Visualize YOLO bounding boxes on dataset images
python scripts/draw_boxes.py --split train
```

---

## Training flags

| Flag | Meaning |
|------|---------|
| `--model` | `scratch`, `transfer`, or `all` |
| `--backbone` | `MobileNetV2`, `ResNet50`, `EfficientNetB0` |
| `--epochs` | Number of training epochs |
| `--fine-tune` | Unfreeze top backbone layers after initial training |
| `--advanced-prep` | Enable denoise + CLAHE on crops |
| `--tensorboard` | Save TensorBoard logs |
| `--gradcam` | Save Grad-CAM figures after training |
| `--train-all-backbones` | Train all 3 backbones and generate comparison table |

---

## Notes

- 6 defect classes: `mouse_bite`, `spur`, `missing_hole`, `short`, `open_circuit`, `spurious_copper`
- The `venv/` and `dataset/` folders are excluded from the repo via `.gitignore`
- Training results and model weights are saved per-run in `results/<run_name>/`
