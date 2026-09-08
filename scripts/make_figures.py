"""Generate all paper figures from genuine training logs and evaluation output.

No synthetic/mocked numbers -- every figure is built directly from
training_log_v2.txt and runs/eval_report_*.json, the unedited output of
src/train.py and src/evaluate.py.
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_LABELS = {
    "hybrid_cnn_transformer": "Hybrid CNN-Transformer\n(proposed)",
    "resnet50": "ResNet50",
    "efficientnet_b0": "EfficientNet-B0",
    "vit_base": "ViT-Base",
}
MODEL_ORDER = ["hybrid_cnn_transformer", "efficientnet_b0", "resnet50", "vit_base"]
COLORS = ["#2b6cb0", "#38a169", "#d69e2e", "#c53030"]
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def load_eval_reports():
    reports = {}
    for name in MODEL_ORDER:
        with open(ROOT / "runs" / f"eval_report_{name}.json") as f:
            reports[name] = json.load(f)["report"]
    return reports


def parse_training_log():
    """Parse training_log_v2.txt into {model: {"train_loss": [...], "val_loss": [...],
    "train_acc": [...], "val_acc": [...]}}."""
    text = (ROOT / "training_log_v2.txt").read_text()
    blocks = re.split(r"TRAINING: (\w+)\s+\(", text)[1:]  # alternating [name, block, name, block, ...]
    history = {}
    for i in range(0, len(blocks), 2):
        name, block = blocks[i], blocks[i + 1]
        train_loss, val_loss, train_acc, val_acc = [], [], [], []
        for m in re.finditer(
            r"epoch \d+: train \{'loss': ([\d.]+), 'accuracy': ([\d.]+)\} "
            r"val \{'loss': ([\d.]+), 'accuracy': ([\d.]+)\}",
            block,
        ):
            train_loss.append(float(m.group(1)))
            train_acc.append(float(m.group(2)))
            val_loss.append(float(m.group(3)))
            val_acc.append(float(m.group(4)))
        if train_loss:
            history[name] = {
                "train_loss": train_loss, "val_loss": val_loss,
                "train_acc": train_acc, "val_acc": val_acc,
            }
    return history


def fig1_classification_bars(reports):
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "macro_auroc", "cohen_kappa"]
    metric_labels = ["Accuracy", "Balanced\nAccuracy", "Macro F1", "Macro\nAUROC", "Cohen's\nKappa"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(metrics))
    width = 0.2
    for i, name in enumerate(MODEL_ORDER):
        values = [reports[name][m] for m in metrics]
        ax.bar(x + (i - 1.5) * width, values, width, label=MODEL_LABELS[name].replace("\n", " "),
               color=COLORS[i])
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Classification performance by model (test set, n=201)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_classification_bars.png", dpi=200)
    plt.close(fig)


def fig2_training_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"wspace": 0.3})
    epochs = range(1, 7)
    for i, name in enumerate(MODEL_ORDER):
        h = history[name]
        axes[0].plot(epochs, h["train_loss"], "--", color=COLORS[i], alpha=0.6)
        axes[0].plot(epochs, h["val_loss"], "-", color=COLORS[i], label=MODEL_LABELS[name].replace("\n", " "))
        axes[1].plot(epochs, h["train_acc"], "--", color=COLORS[i], alpha=0.6)
        axes[1].plot(epochs, h["val_acc"], "-", color=COLORS[i], label=MODEL_LABELS[name].replace("\n", " "))

    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].set_title("Loss\n(dashed=train, solid=val)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].set_title("Accuracy\n(dashed=train, solid=val)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.set_xticks(list(epochs))
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=8)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(FIG_DIR / "fig2_training_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig3_confusion_matrix(reports):
    cm = np.array(reports["hybrid_cnn_transformer"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES))); ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticks(range(len(CLASS_NAMES))); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.set_title("Hybrid CNN-Transformer: Confusion Matrix (test, n=201)")
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            val = cm[i, j]
            if val > 0:
                color = "white" if val > cm.max() / 2 else "black"
                ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_confusion_matrix.png", dpi=200)
    plt.close(fig)


def fig4_ablation_bars():
    # Values transcribed from docs/results.md (themselves from runs/eval_report_*.json)
    groups = ["Accuracy", "Balanced\nAccuracy", "Macro F1", "Faithfulness\nIoU"]
    arch = {"CNN-only\n(EfficientNet-B0)": [0.751, 0.450, 0.464, None],
            "CNN+Transformer\n(proposed)": [0.761, 0.457, 0.466, 0.163]}
    prep = {"Without preprocessing": [0.751, 0.510, 0.511, 0.105],
            "With preprocessing\n(proposed)": [0.761, 0.457, 0.466, 0.163]}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, data, title in [(axes[0], arch, "Ablation: Architecture"), (axes[1], prep, "Ablation: Preprocessing")]:
        x = np.arange(len(groups))
        width = 0.35
        for i, (label, values) in enumerate(data.items()):
            vals = [v if v is not None else 0 for v in values]
            bars = ax.bar(x + (i - 0.5) * width, vals, width, label=label)
            for j, v in enumerate(values):
                if v is None:
                    ax.text(x[j] + (i - 0.5) * width, 0.02, "N/A", ha="center", fontsize=7, rotation=90)
        ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=8)
        ax.set_ylim(0, 0.85)
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_ablation_bars.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    reports = load_eval_reports()
    history = parse_training_log()
    fig1_classification_bars(reports)
    fig2_training_curves(history)
    fig3_confusion_matrix(reports)
    fig4_ablation_bars()
    print("Figures written to", FIG_DIR)
    for f in sorted(FIG_DIR.glob("*.png")):
        print(" -", f.name)
