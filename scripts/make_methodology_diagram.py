"""Generate the end-to-end methodology/architecture diagram for the paper: preprocessing ->
hybrid CNN-Transformer -> classification, with the parallel XAI / uncertainty / Trust Score
branches. Pure schematic (no data plotted) built with matplotlib patches for a clean,
publication-style flowchart.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BLUE = "#DCE6F5"
BLUE_EDGE = "#2b6cb0"
GREEN = "#E1F0E5"
GREEN_EDGE = "#38a169"
ORANGE = "#FCEED8"
ORANGE_EDGE = "#d69e2e"
RED = "#F8DEDE"
RED_EDGE = "#c53030"
GREY = "#EDEDED"
GREY_EDGE = "#555555"


def box(ax, xy, w, h, text, face, edge, fontsize=9.5):
    x, y = xy
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                        facecolor=face, edgecolor=edge, linewidth=1.4, zorder=2)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
             zorder=3, wrap=True)
    return (x, y, w, h)


def arrow(ax, b1, b2, side1="right", side2="left", style="-|>", color="#333333", lw=1.4,
          connectionstyle=None):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    points = {
        "right": (x1 + w1, y1 + h1 / 2), "left": (x1, y1 + h1 / 2),
        "top": (x1 + w1 / 2, y1 + h1), "bottom": (x1 + w1 / 2, y1),
    }
    points2 = {
        "right": (x2 + w2, y2 + h2 / 2), "left": (x2, y2 + h2 / 2),
        "top": (x2 + w2 / 2, y2 + h2), "bottom": (x2 + w2 / 2, y2),
    }
    p1, p2 = points[side1], points2[side2]
    cs = connectionstyle or "arc3,rad=0.0"
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=12, color=color, lw=lw,
                         connectionstyle=cs, zorder=1)
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(12, 7.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    # --- Row 1: input -> preprocessing -> backbone -> transformer -> prediction ---
    b_input = box(ax, (0.2, 5.6), 1.5, 1.0, "Dermoscopic\nImage $X$", GREY, GREY_EDGE)
    b_prep = box(ax, (2.1, 5.5), 1.9, 1.2,
                 "Preprocessing\nHair removal (DullRazor)\nGray-world + CLAHE\nAugmentation",
                 GREY, GREY_EDGE, fontsize=8.5)
    b_cnn = box(ax, (4.4, 5.5), 1.7, 1.2, "CNN Stem\nEfficientNet-B0\n$F=\\mathrm{CNN}_\\theta(X)$",
                BLUE, BLUE_EDGE, fontsize=8.8)
    b_tok = box(ax, (6.5, 5.5), 1.8, 1.2, "Tokenize + CLS\n+ Positional\nEmbedding",
                BLUE, BLUE_EDGE, fontsize=8.8)
    b_trans = box(ax, (8.7, 5.5), 1.7, 1.2, "Transformer\nEncoder ($L$=4)\nMulti-Head\nSelf-Attention",
                  BLUE, BLUE_EDGE, fontsize=8.5)
    b_pred = box(ax, (10.7, 5.6), 1.1, 1.0, "$\\hat{y}$=\nsoftmax\n(class)", BLUE, BLUE_EDGE, fontsize=8.5)

    arrow(ax, b_input, b_prep)
    arrow(ax, b_prep, b_cnn)
    arrow(ax, b_cnn, b_tok)
    arrow(ax, b_tok, b_trans)
    arrow(ax, b_trans, b_pred)

    # dropout note under transformer
    ax.text(9.55, 5.35, "Dropout active at\ninference (MC-Dropout)", ha="center", va="top",
            fontsize=7.5, style="italic", color="#555555")

    # --- Row 2: baselines (for comparison) ---
    b_base = box(ax, (4.4, 3.9), 4.0, 0.7,
                 "Baselines (same preprocessing/training): ResNet50, EfficientNet-B0, ViT-Base",
                 GREY, GREY_EDGE, fontsize=8.5)
    arrow(ax, b_prep, b_base, side1="bottom", side2="left", connectionstyle="arc3,rad=-0.2")

    # --- Row 3: from CNN + Transformer down to XAI methods ---
    b_gradcam = box(ax, (3.6, 2.5), 1.7, 0.85, "Grad-CAM++\n(CNN stem)", ORANGE, ORANGE_EDGE, fontsize=8.5)
    b_rollout = box(ax, (5.5, 2.5), 1.9, 0.85, "Attention Rollout\n(Transformer)", ORANGE, ORANGE_EDGE, fontsize=8.5)
    b_mcdrop = box(ax, (9.2, 2.5), 2.0, 0.85, "MC-Dropout\n($T$=10 passes)\n$\\mu,\\ \\sigma^2$", RED, RED_EDGE, fontsize=8.5)

    arrow(ax, b_cnn, b_gradcam, side1="bottom", side2="top", connectionstyle="arc3,rad=0.15")
    arrow(ax, b_trans, b_rollout, side1="bottom", side2="top", connectionstyle="arc3,rad=-0.15")
    arrow(ax, b_trans, b_mcdrop, side1="bottom", side2="top", connectionstyle="arc3,rad=0.2")

    # --- Row 4: agreement + faithfulness ---
    b_agree = box(ax, (3.6, 1.2) , 3.8, 0.85,
                  "Cross-Method Agreement\nIoU + Spearman $\\rho$", ORANGE, ORANGE_EDGE, fontsize=8.5)
    b_faith = box(ax, (0.2, 1.2), 3.0, 0.85,
                  "Faithfulness vs.\nGround-Truth Lesion Mask\nIoU + Pointing Game", GREEN, GREEN_EDGE, fontsize=8)

    arrow(ax, b_gradcam, b_agree, side1="bottom", side2="top", connectionstyle="arc3,rad=0.15")
    arrow(ax, b_rollout, b_agree, side1="bottom", side2="top", connectionstyle="arc3,rad=-0.15")
    arrow(ax, b_gradcam, b_faith, side1="left", side2="top", connectionstyle="arc3,rad=-0.3")

    b_mask = box(ax, (0.2, 2.5), 1.9, 0.85, "Ground-Truth\nSegmentation Mask", GREEN, GREEN_EDGE, fontsize=8.5)
    arrow(ax, b_mask, b_faith, side1="bottom", side2="top")

    # --- Row 5: Trust Score fusion ---
    b_trust = box(ax, (3.6, 0.05), 3.8, 0.85,
                  "Trust Score $T = w_1(1-\\sigma^2_{norm}) + w_2\\cdot$Faith$+ w_3\\cdot$Agree",
                  RED, RED_EDGE, fontsize=8.3)
    arrow(ax, b_faith, b_trust, side1="right", side2="left", connectionstyle="arc3,rad=-0.2")
    arrow(ax, b_agree, b_trust, side1="bottom", side2="top")
    arrow(ax, b_mcdrop, b_trust, side1="bottom", side2="right", connectionstyle="arc3,rad=0.3")

    # Legend
    legend_elems = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=BLUE, markeredgecolor=BLUE_EDGE, markersize=12, label="Classification pathway"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=ORANGE, markeredgecolor=ORANGE_EDGE, markersize=12, label="Explainability (XAI)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=GREEN, markeredgecolor=GREEN_EDGE, markersize=12, label="Faithfulness evaluation"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=RED, markeredgecolor=RED_EDGE, markersize=12, label="Uncertainty / Trust Score"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=GREY, markeredgecolor=GREY_EDGE, markersize=12, label="Data / preprocessing / baselines"),
    ]
    ax.legend(handles=legend_elems, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=3, fontsize=8, frameon=False)

    ax.set_title("End-to-end methodology: hybrid CNN-Transformer classification with\n"
                 "multi-method XAI, mask-anchored faithfulness, and uncertainty-fused Trust Score",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig0_methodology_diagram.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig0_methodology_diagram.png")


if __name__ == "__main__":
    main()
