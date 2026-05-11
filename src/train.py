"""
Training pipeline — scratch CNN, transfer learning (MobileNetV2 / ResNet50 / EfficientNetB0),
optional fine-tuning, GPU-aware runtime, CSV logs, TensorBoard, hyperparameter export.

Performance targets (e.g. val accuracy > 99%) depend on the dataset, hardware, and training budget;
this module maximizes achievable accuracy but does not hard-code a minimum threshold.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import tensorflow as tf

from .config import FINETUNE_LAYER_DEFAULTS, configure_runtime
from .data_loader import (
    CLASS_NAMES,
    NUM_CLASSES,
    build_annotation_manifest,
    compute_class_weight_dict,
    save_manifest_csv,
)
from .evaluate import run_full_evaluation
from .explainability import save_gradcam, visualize_first_layer_filters
from .model import (
    build_scratch_cnn,
    build_transfer_model,
    fine_tune_unfreeze_top_layers,
    print_model_summary_and_params,
    save_architecture_diagram,
)
from .preprocess import build_datasets


def _merge_fit_history(h1: dict, h2: tf.keras.callbacks.History) -> None:
    for k, v in h2.history.items():
        h1.setdefault(k, []).extend([float(x) for x in v])


def _history_to_jsonable(history: tf.keras.callbacks.History) -> dict[str, list[float]]:
    return {k: [float(x) for x in v] for k, v in history.history.items()}


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def train_one_run(
    model_kind: str,
    backbone: str,
    image_size: tuple[int, int],
    batch_size: int,
    epochs: int,
    output_dir: Path,
    fine_tune: bool,
    fine_tune_epochs: int,
    fine_tune_at: int,
    fine_tune_lr: float,
    use_class_weights: bool,
    cache_train: bool,
    log_tensorboard: bool,
    advanced_preprocessing: bool,
    save_gradcam_sample: bool,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    train_manifest = build_annotation_manifest("train")
    class_weight = compute_class_weight_dict(train_manifest) if use_class_weights else None

    train_ds, val_ds, test_ds = build_datasets(
        image_size=image_size,
        batch_size=batch_size,
        augment=True,
        cache_val_test=True,
        cache_train=cache_train,
        advanced_preprocessing=advanced_preprocessing,
    )

    if model_kind == "scratch":
        model = build_scratch_cnn(input_shape=(*image_size, 3), num_classes=NUM_CLASSES)
    else:
        model = build_transfer_model(input_shape=(*image_size, 3), num_classes=NUM_CLASSES, backbone=backbone)

    print_model_summary_and_params(model)
    save_architecture_diagram(model, output_dir / "architecture_diagram.png")

    compile_model(model, learning_rate=1e-4)

    best_path = model_dir / "best_model.keras"
    final_path = model_dir / "final_model.keras"
    callbacks: list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.45,
            patience=4,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(logs_dir / "training_log.csv"), append=False),
    ]
    tb_dir = output_dir / "tensorboard"
    if log_tensorboard:
        tb_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=str(tb_dir), histogram_freq=0))

    hyperparams: dict[str, Any] = {
        "model_kind": model_kind,
        "backbone": backbone if model_kind == "transfer" else None,
        "image_size": list(image_size),
        "batch_size": batch_size,
        "epochs_planned": epochs,
        "fine_tune": fine_tune,
        "fine_tune_epochs": fine_tune_epochs,
        "fine_tune_at": fine_tune_at,
        "fine_tune_lr": fine_tune_lr,
        "advanced_preprocessing": advanced_preprocessing,
        "class_weights": use_class_weights,
        "num_classes": NUM_CLASSES,
    }
    (output_dir / "hyperparameters.json").write_text(json.dumps(hyperparams, indent=2), encoding="utf-8")

    t0 = time.perf_counter()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight,
    )
    train_time_sec = time.perf_counter() - t0

    if model_kind == "transfer" and fine_tune and fine_tune_epochs > 0:
        t1 = time.perf_counter()
        fine_tune_unfreeze_top_layers(model, fine_tune_at)
        compile_model(model, learning_rate=fine_tune_lr)
        history_ft = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=fine_tune_epochs,
            callbacks=callbacks,
            class_weight=class_weight,
        )
        _merge_fit_history(history.history, history_ft)
        train_time_sec += time.perf_counter() - t1

    model.save(str(final_path))

    history_json = _history_to_jsonable(history)
    (output_dir / "history.json").write_text(json.dumps(history_json, indent=2), encoding="utf-8")

    best_val_acc = max(history_json.get("val_accuracy", [0.0]))
    best_train_acc = max(history_json.get("accuracy", [0.0]))

    eval_model = model
    if best_path.exists():
        try:
            eval_model = tf.keras.models.load_model(str(best_path))
        except Exception:
            eval_model = model

    metrics = run_full_evaluation(
        eval_model,
        output_dir,
        history=history_json,
        plot_roc=True,
        advanced_preprocessing=advanced_preprocessing,
    )

    if save_gradcam_sample:
        try:
            ex_dir = output_dir / "explainability"
            for xb, _ in test_ds.take(1):
                save_gradcam(eval_model, xb.numpy()[:1], ex_dir / "grad_cam_sample.png")
                break
            if model_kind == "scratch":
                visualize_first_layer_filters(eval_model, ex_dir / "first_conv_filters.png")
        except Exception as e:
            print("[explainability] skipped:", e, file=sys.stderr)

    params = int(eval_model.count_params())
    summary = {
        "model_kind": model_kind,
        "backbone": backbone if model_kind == "transfer" else None,
        "epochs_ran": len(history_json.get("loss", [])),
        "train_time_sec": round(train_time_sec, 2),
        "num_parameters": params,
        "best_val_accuracy": float(best_val_acc),
        "best_train_accuracy": float(best_train_acc),
        "test_accuracy": metrics["accuracy"],
        "test_f1_weighted": metrics["f1_weighted"],
        "test_f1_macro": metrics["f1_macro"],
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def write_comparison_table(rows: list[dict], path: Path) -> None:
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    cols = list(df.columns)
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    header = "| " + " | ".join(cols) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.astype(str).values.tolist()]
    md = "# Model comparison\n\n" + "\n".join([header, sep, *body]) + "\n"
    path.write_text(md, encoding="utf-8")


TRANSFER_BACKBONES = ("MobileNetV2", "ResNet50", "EfficientNetB0")


def main() -> None:
    parser = argparse.ArgumentParser(description="Production PCB defect training pipeline.")
    parser.add_argument("--model", choices=["scratch", "transfer", "all"], default="transfer")
    parser.add_argument(
        "--backbone",
        choices=list(TRANSFER_BACKBONES),
        default="MobileNetV2",
        help="Transfer backbone (EfficientNetB0 ~ stronger accuracy / heavier compute).",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--fine-tune", action="store_true")
    parser.add_argument("--fine-tune-epochs", type=int, default=6)
    parser.add_argument(
        "--fine-tune-at",
        type=int,
        default=None,
        help="Layer index to unfreeze from; default is backbone-specific (see config.FINETUNE_LAYER_DEFAULTS).",
    )
    parser.add_argument("--fine-tune-lr", type=float, default=1e-5)
    parser.add_argument("--save-manifests", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--cache-train", action="store_true")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument(
        "--advanced-prep",
        action="store_true",
        help="Denoise + CLAHE on BGR crops before resize (slower, often improves contrast).",
    )
    parser.add_argument("--mixed-precision", action="store_true", help="FP16 on GPU for speed (requires GPU).")
    parser.add_argument("--gradcam", action="store_true", help="Save Grad-CAM + first-layer viz after training.")
    parser.add_argument(
        "--train-all-backbones",
        action="store_true",
        help="Train all transfer backbones (MobileNetV2, ResNet50, EfficientNetB0) into separate folders.",
    )
    args = parser.parse_args()

    configure_runtime(mixed_precision=args.mixed_precision)

    image_size = (args.image_size, args.image_size)
    base_out: Path = args.output

    if args.save_manifests:
        man_dir = base_out / "manifests"
        for sp in ("train", "val", "test"):
            save_manifest_csv(build_annotation_manifest(sp), man_dir / f"{sp}_manifest.csv")

    use_weights = not args.no_class_weights
    rows: list[dict] = []

    def ft_at(backbone: str) -> int:
        if args.fine_tune_at is not None:
            return args.fine_tune_at
        return FINETUNE_LAYER_DEFAULTS.get(backbone, 100)

    def run_block(kind: str, bb: str, out_sub: str) -> dict[str, Any]:
        ft_idx = ft_at(bb) if kind == "transfer" else 0
        return train_one_run(
            kind,
            backbone=bb,
            image_size=image_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            output_dir=base_out / out_sub,
            fine_tune=args.fine_tune,
            fine_tune_epochs=args.fine_tune_epochs,
            fine_tune_at=ft_idx,
            fine_tune_lr=args.fine_tune_lr,
            use_class_weights=use_weights,
            cache_train=args.cache_train,
            log_tensorboard=args.tensorboard,
            advanced_preprocessing=args.advanced_prep,
            save_gradcam_sample=args.gradcam,
        )

    if args.train_all_backbones:
        for bb in TRANSFER_BACKBONES:
            s = run_block("transfer", bb, f"transfer_{bb}")
            rows.append(
                {
                    "Model": f"Transfer ({bb})",
                    "Test accuracy": round(s["test_accuracy"], 5),
                    "Best val accuracy": round(s["best_val_accuracy"], 5),
                    "F1 (weighted)": round(s["test_f1_weighted"], 5),
                    "Parameters": s["num_parameters"],
                    "Train time (s)": s["train_time_sec"],
                }
            )
        write_comparison_table(rows, base_out / "model_comparison.md")
        print("Done. Compared all backbones under:", base_out.resolve())
        return

    if args.model in ("scratch", "all"):
        s = run_block("scratch", args.backbone, "scratch_cnn")
        rows.append(
            {
                "Model": "Scratch CNN (Functional)",
                "Test accuracy": round(s["test_accuracy"], 5),
                "Best val accuracy": round(s["best_val_accuracy"], 5),
                "F1 (weighted)": round(s["test_f1_weighted"], 5),
                "Parameters": s["num_parameters"],
                "Train time (s)": s["train_time_sec"],
            }
        )

    if args.model in ("transfer", "all"):
        s = run_block("transfer", args.backbone, f"transfer_{args.backbone}")
        rows.append(
            {
                "Model": f"Transfer ({args.backbone})",
                "Test accuracy": round(s["test_accuracy"], 5),
                "Best val accuracy": round(s["best_val_accuracy"], 5),
                "F1 (weighted)": round(s["test_f1_weighted"], 5),
                "Parameters": s["num_parameters"],
                "Train time (s)": s["train_time_sec"],
            }
        )

    if args.model == "all" and len(rows) == 2:
        write_comparison_table(rows, base_out / "model_comparison.md")

    print("Done. Outputs under:", base_out.resolve())


if __name__ == "__main__":
    main()
