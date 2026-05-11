# Datasets

Place the **Kaggle PCB Defect** release here as:

`pcb-defect-dataset/` (with `data.yaml`, `train/val/test` each containing `images/` and `labels/`).

**Default project path:** `../dataset/pcb-defect-dataset/` (sibling folder name `dataset`, not `datasets`).

To use this folder instead, either:

- Copy or symlink the dataset into `datasets/pcb-defect-dataset/`, **or**
- Set environment variable `PCB_DATA_ROOT` to the absolute path of your `pcb-defect-dataset` directory.

```powershell
set PCB_DATA_ROOT=C:\path\to\pcb-defect-dataset
python train.py --model transfer --fine-tune
```
