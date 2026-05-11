# PCB Defect Detection & Classification

End-to-end **production-style** PCB defect AI: multi-class classification on **YOLO-cropped** patches, **custom Functional CNN**, **transfer learning** (MobileNetV2, **EfficientNetB0**, ResNet50), **fine-tuning**, strong **augmentation**, optional **denoise + CLAHE**, **Grad-CAM** explainability, **GPU memory growth** + optional **mixed precision**, CSV **training logs**, **TensorBoard**, evaluation + ROC, **batch/real-time inference**, **Streamlit**, and auto-generated **final report**.

> Raw Kaggle images + labels only (not MNIST/CIFAR loaders). ImageNet **weights** initialize backbones only.

> **Accuracy:** Requirements such as “always >99% validation accuracy” are **not guaranteed by code** — they depend on data splits, noise, and compute. Your recorded **best_val_accuracy** in `training_summary.json` is the ground truth for grading/viva.

## Repository layout

```
PCB_Defect_Detection/
├── dataset/                 # Default PCB dataset root (see also datasets/README.md)
├── datasets/                # README + optional symlink/copy location
├── models/registry/         # Place exported checkpoints for deployment tracking
├── src/
│   ├── config.py            # Paths, GPU setup, fine-tune defaults per backbone
│   ├── data_loader.py       # YAML, YOLO manifest, class weights
│   ├── data_analysis.py     # EDA plots
│   ├── preprocess.py        # Crop, optional denoise/CLAHE, augment, tf.data
│   ├── model.py             # Scratch CNN + MobileNet / EfficientNet / ResNet
│   ├── train.py             # Full training pipeline + logs + optional Grad-CAM
│   ├── evaluate.py          # Metrics, curves, ROC
│   ├── explainability.py    # Grad-CAM, first-layer filters (scratch)
│   ├── predict.py           # Single-image / folder batch inference
│   └── report_generator.py  # Assembles report/FINAL_PROJECT_REPORT.md
├── scripts/                 # CLI wrappers
├── notebooks/
├── results/                 # Training outputs (per run subdirectory)
├── report/                  # REPORT.md + generated FINAL_PROJECT_REPORT.md
├── presentation/
├── app_streamlit.py
├── train.py
├── requirements.txt
└── README.md
```

## Installation (Windows)

```powershell
cd path\to\PCB_Defect_Detection
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional (architecture PNG): install [Graphviz](https://graphviz.org/download/) and `pip install pydot`.

### If TensorFlow fails to load (`DLL load failed`, `DllMain returned false`)

1. **`requirements.txt` pins TensorFlow 2.15.1** — reinstall:  
   `pip uninstall -y tensorflow keras` then `pip install -r requirements.txt`
2. Install **[VC++ Redistributable x64](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)** (2015–2022).
3. Very old CPUs without **AVX** may not run official TensorFlow wheels; use **WSL2** (Ubuntu) or another machine for training.
4. Optional: `set TF_ENABLE_ONEDNN_OPTS=0` before running if oneDNN causes issues.

## Dataset

Place the Kaggle **Printed Circuit Board (PCB) Defects** export under:

`dataset/pcb-defect-dataset/` with `data.yaml`, `train|val|test/{images,labels}/`.

## Quick pipeline

```powershell
# 1) EDA
python -m src.data_analysis

# 2) Train transfer + fine-tune + TensorBoard + CSV logs (+ optional advanced prep)
python train.py --model transfer --backbone EfficientNetB0 --epochs 30 --fine-tune --tensorboard --advanced-prep

# 3) Train scratch + one backbone + comparison table
python train.py --model all --backbone MobileNetV2 --fine-tune --tensorboard --gradcam

# 4) Train all three transfer backbones (long run) → model_comparison.md
python train.py --train-all-backbones --epochs 25 --fine-tune --tensorboard

# 5) GPU FP16 speed (NVIDIA): 
python train.py --model transfer --mixed-precision --fine-tune --tensorboard

# 6) Evaluate / predict / report
python -m src.evaluate --run-dir results/transfer_EfficientNetB0
python -m src.predict --model results/transfer_EfficientNetB0/models/best_model.keras --image sample.jpg
python -m src.report_generator --results-dir results

# 7) Streamlit (reads hyperparameters.json for crop size & advanced prep)
streamlit run app_streamlit.py
```

### Useful flags

| Flag | Meaning |
|------|---------|
| `--advanced-prep` | Denoise + CLAHE before resize (train/eval must match) |
| `--train-all-backbones` | MobileNetV2, ResNet50, EfficientNetB0 sequentially |
| `--gradcam` | Save explainability figures after training |
| `--no-class-weights` | Disable balanced class weights |
| `--cache-train` | In-memory `tf.data` cache for train (high RAM) |
| `--save-manifests` | CSV manifests under `results/manifests/` |

## Training outputs (per run directory)

- `models/best_model.keras`, `models/final_model.keras`
- `hyperparameters.json`, `history.json`, `training_summary.json`, `metrics.json`
- `logs/training_log.csv` (CSVLogger), `tensorboard/` (if `--tensorboard`)
- Curves, confusion matrix, classification report, optional ROC
- `explainability/` when using `--gradcam`

## Evaluation

`src/evaluate.py` recomputes test metrics and plots; accepts `--run-dir` or `--model-path`.

## Report & slides

- **Report:** `report/REPORT.md` — all required sections; embed your `results/` screenshots after training.  
- **Slides:** `presentation/SLIDES.md` — [Marp](https://marp.app/) (`npm i -g @marp-team/marp-cli` → `marp SLIDES.md -o slides.pdf`).

## Accuracy expectations

The code applies **strong augmentation**, **class weights**, **ReduceLROnPlateau**, **early stopping**, and **fine-tuning** to maximize accuracy on your hardware. **Exact** val/test numbers (e.g. 99%) are **data- and run-dependent**; cite the values from **your** `metrics.json` in the report/viva.

## Academic integrity

All preprocessing and label parsing are **implemented explicitly** in this repository (OpenCV + NumPy + TensorFlow), not hidden inside a black-box dataset API for PCB data.
