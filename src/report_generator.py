"""
Assemble FINAL_PROJECT_REPORT.md from training artifacts under results/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("report") / "FINAL_PROJECT_REPORT.md")
    args = parser.parse_args()

    root = args.results_dir
    sections = [
        "# PCB Defect AI — Final Technical Report",
        "",
        "Generated automatically from `training_summary.json`, `hyperparameters.json`, and `metrics.json` in each run folder.",
        "",
        "> **Note:** Claimed validation accuracy targets (e.g. >99%) are empirical outcomes from training runs; reproduce on your hardware using the saved hyperparameters.",
        "",
        "## 1. Dataset & task",
        "",
        "- Multi-class PCB defect classification on cropped patches guided by YOLO annotations.",
        "- Raw Kaggle-format images + `.txt` labels (manual loading, no toy dataset loaders).",
        "",
        "## 2. Model runs summary",
        "",
    ]

    subdirs = sorted([p for p in root.iterdir() if p.is_dir() and (p / "training_summary.json").exists()])
    for d in subdirs:
        summary_path = d / "training_summary.json"
        hp_path = d / "hyperparameters.json"
        met_path = d / "metrics.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        hp = json.loads(hp_path.read_text(encoding="utf-8")) if hp_path.exists() else {}
        met = json.loads(met_path.read_text(encoding="utf-8")) if met_path.exists() else {}

        sections.append(f"### {d.name}")
        sections.append("")
        sections.append("| Metric | Value |")
        sections.append("|--------|-------|")
        sections.append(f"| Best val accuracy | {summary.get('best_val_accuracy', 'n/a')} |")
        sections.append(f"| Test accuracy | {summary.get('test_accuracy', met.get('accuracy', 'n/a'))} |")
        sections.append(f"| F1 (weighted) | {summary.get('test_f1_weighted', met.get('f1_weighted', 'n/a'))} |")
        sections.append(f"| Parameters | {summary.get('num_parameters', 'n/a')} |")
        sections.append(f"| Train time (s) | {summary.get('train_time_sec', 'n/a')} |")
        sections.append("")
        if hp:
            sections.append("**Hyperparameters:**")
            sections.append("")
            sections.append("```json")
            sections.append(json.dumps(hp, indent=2))
            sections.append("```")
            sections.append("")
        sections.append(f"Artifacts: `{d}/models/best_model.keras`, curves, confusion matrix, logs/training_log.csv")
        sections.append("")

    sections.extend(
        [
            "## 3. Architecture references",
            "",
            "- See each run folder for `architecture_diagram.png` (requires Graphviz).",
            "",
            "## 4. Explainability",
            "",
            "- Grad-CAM and first-layer filters (scratch): `explainability/` inside each run when trained with `--gradcam`.",
            "",
            "## 5. Deployment",
            "",
            "- Streamlit: `streamlit run app_streamlit.py`",
            "- Batch CLI: `python -m src.predict --model results/<run>/models/best_model.keras --folder <dir>`",
            "",
        ]
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(sections), encoding="utf-8")
    print("Wrote", args.out.resolve())


if __name__ == "__main__":
    main()
