"""
Offline evaluation — test metrics, confusion matrix heatmap, training curves from saved history,
optional multi-class ROC / macro AUC (one-vs-rest).

Run after training:
  python -m src.evaluate --run-dir results/scratch_cnn
Or with explicit model path:
  python -m src.evaluate --model-path results/scratch_cnn/models/best_model.keras --output results/eval_scratch
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from .data_loader import CLASS_NAMES
from .preprocess import build_datasets


def _ordered_class_ids():
    return sorted(CLASS_NAMES.keys())


def save_training_curves(history: dict[str, list[float]], output_dir: Path, prefix: str = "") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    p = f"{prefix}_" if prefix else ""

    loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])
    if loss and val_loss:
        plt.figure(figsize=(10, 4))
        plt.plot(loss, label="train_loss")
        plt.plot(val_loss, label="val_loss")
        plt.title("Loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{p}loss_vs_epoch.png", dpi=150)
        plt.close()

    acc = history.get("accuracy", [])
    val_acc = history.get("val_accuracy", [])
    if acc and val_acc:
        plt.figure(figsize=(10, 4))
        plt.plot(acc, label="train_accuracy")
        plt.plot(val_acc, label="val_accuracy")
        plt.title("Accuracy vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{p}accuracy_vs_epoch.png", dpi=150)
        plt.close()


def save_confusion_matrix_heatmap(y_true, y_pred, output_path: Path) -> np.ndarray:
    labels = _ordered_class_ids()
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    names = [CLASS_NAMES[i] for i in labels]
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=names, yticklabels=names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return cm


def collect_predictions_proba(
    model: tf.keras.Model, dataset: tf.data.Dataset
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true: list[int] = []
    y_pred: list[int] = []
    probas: list[np.ndarray] = []
    for x_batch, y_batch in dataset:
        preds = model.predict(x_batch, verbose=0)
        probas.append(preds)
        y_true.extend(np.argmax(y_batch.numpy(), axis=-1).tolist())
        y_pred.extend(np.argmax(preds, axis=-1).tolist())
    y_score = np.vstack(probas)
    return np.array(y_true), np.array(y_pred), y_score


def plot_multiclass_roc_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    output_dir: Path,
) -> dict[str, float]:
    """
    One-vs-rest ROC per class + micro-average curve; reports macro AUC (OvR).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = _ordered_class_ids()
    y_bin = label_binarize(y_true, classes=labels)

    plt.figure(figsize=(9, 7))
    aucs = {}
    for i, cid in enumerate(labels):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        aucs[CLASS_NAMES[cid]] = float(roc_auc)
        plt.plot(fpr, tpr, lw=2, label=f"{CLASS_NAMES[cid]} (AUC={roc_auc:.3f})")

    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_score.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    plt.plot(fpr_micro, tpr_micro, linestyle="--", color="black", lw=2, label=f"micro (AUC={auc_micro:.3f})")
    plt.plot([0, 1], [0, 1], "k:", lw=1)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-class ROC (one-vs-rest)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_multiclass.png", dpi=150)
    plt.close()

    try:
        macro_auc = float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro"))
    except ValueError:
        macro_auc = float("nan")

    with open(output_dir / "roc_auc_summary.json", "w", encoding="utf-8") as f:
        json.dump({"per_class_auc": aucs, "macro_auc_ovr": macro_auc, "micro_auc": float(auc_micro)}, f, indent=2)
    return {"macro_auc_ovr": macro_auc, "micro_auc": float(auc_micro)}


def run_full_evaluation(
    model: tf.keras.Model,
    output_dir: Path,
    history: dict[str, list[float]] | None = None,
    plot_roc: bool = True,
    advanced_preprocessing: bool = False,
) -> dict[str, Any]:
    """
    Evaluate on held-out test set; write metrics, report, confusion matrix, optional ROC.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _, _, test_ds = build_datasets(
        augment=False,
        cache_val_test=True,
        cache_train=False,
        advanced_preprocessing=advanced_preprocessing,
    )

    y_true, y_pred, y_score = collect_predictions_proba(model, test_ds)
    labels = _ordered_class_ids()

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=[CLASS_NAMES[i] for i in labels],
        digits=4,
        zero_division=0,
    )
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    save_confusion_matrix_heatmap(y_true, y_pred, output_dir / "confusion_matrix_heatmap.png")

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if history:
        save_training_curves(history, output_dir)

    roc_extra: dict[str, float] = {}
    if plot_roc and len(CLASS_NAMES) >= 2:
        roc_extra = plot_multiclass_roc_auc(y_true, y_score, output_dir)
        metrics.update(roc_extra)

    return metrics


def load_history_json(path: Path) -> dict[str, list[float]] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: [float(x) for x in v] for k, v in data.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved PCB defect classifier.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Training output folder (uses models/best_model.keras)")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-roc", action="store_true")
    parser.add_argument(
        "--advanced-prep",
        action="store_true",
        help="Match training with denoise+CLAHE (or read hyperparameters.json from --run-dir).",
    )
    args = parser.parse_args()

    if args.model_path:
        model_path = args.model_path
    elif args.run_dir:
        model_path = args.run_dir / "models" / "best_model.keras"
    else:
        raise SystemExit("Provide --run-dir or --model-path")

    out = args.output or (args.run_dir or model_path.parent.parent) / "evaluation_rerun"
    out.mkdir(parents=True, exist_ok=True)

    adv = args.advanced_prep
    hp_path = (args.run_dir or model_path.parent.parent) / "hyperparameters.json"
    if hp_path.exists():
        hp = json.loads(hp_path.read_text(encoding="utf-8"))
        adv = bool(hp.get("advanced_preprocessing", adv))

    model = tf.keras.models.load_model(str(model_path), safe_mode=False)
    hist_path = (args.run_dir or model_path.parent.parent) / "history.json"
    history = load_history_json(hist_path)

    run_full_evaluation(
        model,
        out,
        history=history,
        plot_roc=not args.no_roc,
        advanced_preprocessing=adv,
    )
    print("Wrote evaluation to:", out.resolve())


if __name__ == "__main__":
    main()
