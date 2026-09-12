"""Generate a qualitative XAI figure from the trained hybrid checkpoint: original image,
Grad-CAM++ overlay, Attention Rollout overlay, and ground-truth lesion mask, for a real
test-set image. No synthetic data -- runs actual inference on the saved checkpoint.
"""
import math
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from src.preprocessing.dataset import HAM10000Dataset, CLASS_NAMES
from src.train import build_model
from src.xai.gradcam import gradcam_saliency
from src.xai.attention_rollout import AttentionRolloutHook, compute_rollout, rollout_saliency_map

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "docs" / "figures"


def resize_to(saliency, shape):
    return cv2.resize(saliency.astype(np.float32), (shape[1], shape[0]))


def main():
    cfg = yaml.safe_load((ROOT / "configs" / "config_cpu_20epoch.yaml").read_text())
    device = torch.device("cpu")

    test_ds = HAM10000Dataset(
        split_csv=str(ROOT / "data" / "splits" / "test.csv"),
        images_dir=str(ROOT / "data" / "raw" / "images"),
        image_size=cfg["data"]["image_size"], train=False,
        hair_removal=cfg["preprocessing"]["hair_removal"], clahe=cfg["preprocessing"]["clahe"],
        segmentations_dir=str(ROOT / "data" / "raw" / "segmentations"),
    )

    # pick a test image that (a) has a mask and (b) the model classifies correctly, for a
    # clean illustrative example -- selection criteria stated explicitly, not cherry-picked
    # on saliency quality.
    eval_cfg = {**cfg, "model": {**cfg["model"], "pretrained": False}}
    model = build_model("hybrid_cnn_transformer", eval_cfg, num_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(torch.load(ROOT / "runs" / "hybrid_cnn_transformer" / "best.pt", map_location=device))
    model.eval()

    attn_hook = AttentionRolloutHook(model.transformer)

    chosen = None
    for idx in range(len(test_ds)):
        sample = test_ds[idx]
        if "mask" not in sample:
            continue
        image = sample["image"].unsqueeze(0)
        label = sample["label"].item()
        with torch.no_grad():
            pred = model(image).argmax(dim=-1).item()
        if pred == label:
            chosen = (idx, sample, pred)
            break
    if chosen is None:
        idx, sample = 0, test_ds[0]
        with torch.no_grad():
            pred = model(sample["image"].unsqueeze(0)).argmax(dim=-1).item()
        chosen = (idx, sample, pred)

    idx, sample, pred = chosen
    image = sample["image"].unsqueeze(0)
    gt_mask = sample["mask"]
    h, w = gt_mask.shape

    target_layer = model.cnn.blocks[-1] if hasattr(model.cnn, "blocks") else list(model.cnn.children())[-1]
    gc_map = resize_to(gradcam_saliency(model, target_layer, image, pred), (h, w))

    attn_hook.clear()
    with torch.no_grad():
        model(image)
    rollout = compute_rollout(attn_hook.attentions)
    grid = int(math.sqrt(rollout.shape[-1] - 1))
    ar_map = resize_to(rollout_saliency_map(rollout, grid)[0].cpu().numpy(), (h, w))
    attn_hook.remove()

    # denormalize the displayed image for visualization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    disp = image[0].permute(1, 2, 0).numpy() * std + mean
    disp = np.clip(disp, 0, 1)
    disp_resized = cv2.resize(disp, (w, h))

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(disp_resized); axes[0].set_title(f"Input (true={CLASS_NAMES[sample['label'].item()]},\npred={CLASS_NAMES[pred]})")
    axes[1].imshow(disp_resized); axes[1].imshow(gc_map, cmap="jet", alpha=0.45); axes[1].set_title("Grad-CAM++")
    axes[2].imshow(disp_resized); axes[2].imshow(ar_map, cmap="jet", alpha=0.45); axes[2].set_title("Attention Rollout")
    axes[3].imshow(disp_resized); axes[3].imshow(gt_mask, cmap="Greens", alpha=0.45); axes[3].set_title("Ground-truth lesion mask")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"Qualitative explanation example (test image: {sample['image_id']})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_xai_qualitative.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig5_xai_qualitative.png for image_id:", sample["image_id"])


if __name__ == "__main__":
    main()
