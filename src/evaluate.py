"""Evaluation entry point: classification metrics, XAI faithfulness, uncertainty, Trust Score.

Runs on the test split. For each image with an available ground-truth segmentation mask, also
computes Grad-CAM++ / Attention Rollout / SHAP saliency maps, their cross-method agreement, and
faithfulness against the mask, then fuses everything into a per-image Trust Score.
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.faithfulness import pointing_game_hit, saliency_vs_mask_iou
from src.evaluation.metrics import classification_report_dict, expected_calibration_error
from src.evaluation.trust_score import TrustScoreWeights, compute_trust_score
from src.preprocessing.dataset import CLASS_NAMES, HAM10000Dataset
from src.train import build_model
from src.uncertainty.mc_dropout import mc_dropout_predict
from src.xai.agreement import cross_method_agreement
from src.xai.attention_rollout import AttentionRolloutHook, compute_rollout, rollout_saliency_map
from src.xai.gradcam import gradcam_saliency


def resize_to(saliency: np.ndarray, shape) -> np.ndarray:
    return cv2.resize(saliency.astype(np.float32), (shape[1], shape[0]))


def evaluate(cfg: dict, model_name: str, checkpoint: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg, prep_cfg = cfg["data"], cfg["preprocessing"]
    num_classes = len(CLASS_NAMES)

    test_ds = HAM10000Dataset(
        split_csv=f"{data_cfg['splits_dir']}/test.csv", images_dir=data_cfg["images_dir"],
        image_size=data_cfg["image_size"], train=False,
        hair_removal=prep_cfg["hair_removal"], clahe=prep_cfg["clahe"],
        segmentations_dir=data_cfg.get("segmentations_dir"),
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    # pretrained=False: the loaded checkpoint overwrites weights immediately below, so
    # downloading pretrained ImageNet weights here would be wasted bandwidth/time.
    eval_cfg = {**cfg, "model": {**cfg["model"], "pretrained": False}}
    model = build_model(model_name, eval_cfg, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    is_hybrid = model_name == "hybrid_cnn_transformer"
    attn_hook = AttentionRolloutHook(model.transformer) if is_hybrid else None

    y_true, y_pred, y_proba = [], [], []
    confidences, correctness = [], []
    faithfulness_records, trust_records = [], []
    max_entropy = math.log(num_classes)
    weights = TrustScoreWeights(**cfg["trust_score"]["weights"])

    for batch in tqdm(test_loader, desc="evaluate"):
        image = batch["image"].to(device)
        label = batch["label"].item()

        mean_probs, variance, entropy = mc_dropout_predict(model, image, cfg["model"]["mc_dropout_passes"])
        pred_class = int(mean_probs.argmax(dim=-1).item())
        confidence = float(mean_probs.max(dim=-1).values.item())

        y_true.append(label)
        y_pred.append(pred_class)
        y_proba.append(mean_probs.detach().cpu().numpy()[0])
        confidences.append(confidence)
        correctness.append(int(pred_class == label))

        if "mask" in batch and is_hybrid:
            gt_mask = batch["mask"][0].numpy()
            h, w = gt_mask.shape

            target_layer = model.cnn.blocks[-1] if hasattr(model.cnn, "blocks") else list(model.cnn.children())[-1]
            gc_map = resize_to(gradcam_saliency(model, target_layer, image, pred_class), (h, w))

            attn_hook.clear()
            with torch.no_grad():
                model(image)
            rollout = compute_rollout(attn_hook.attentions)
            grid = int(math.sqrt(rollout.shape[-1] - 1))
            ar_map = resize_to(rollout_saliency_map(rollout, grid)[0].cpu().numpy(), (h, w))

            saliency_maps = {"gradcam++": gc_map, "attention_rollout": ar_map}
            agreement = cross_method_agreement(saliency_maps, threshold=cfg["xai"]["saliency_threshold"])

            faithfulness = saliency_vs_mask_iou(gc_map, gt_mask, cfg["xai"]["saliency_threshold"])
            hit = pointing_game_hit(gc_map, gt_mask)
            faithfulness_records.append({
                "image_id": batch["image_id"][0], "iou": faithfulness, "pointing_game_hit": hit,
                "mean_iou_agreement": agreement["mean_iou"], "mean_spearman_agreement": agreement["mean_spearman"],
            })

            norm_uncertainty = min(float(entropy.item()) / max_entropy, 1.0)
            trust = compute_trust_score(norm_uncertainty, faithfulness, agreement["mean_iou"], weights)
            trust_records.append({"image_id": batch["image_id"][0], "trust_score": trust})

    if attn_hook is not None:
        attn_hook.remove()

    y_true, y_pred, y_proba = np.array(y_true), np.array(y_pred), np.array(y_proba)
    report = classification_report_dict(y_true, y_pred, y_proba, CLASS_NAMES)
    report["ece"] = expected_calibration_error(np.array(confidences), np.array(correctness))

    if faithfulness_records:
        report["mean_faithfulness_iou"] = float(np.mean([r["iou"] for r in faithfulness_records]))
        report["pointing_game_accuracy"] = float(np.mean([r["pointing_game_hit"] for r in faithfulness_records]))
    if trust_records:
        report["mean_trust_score"] = float(np.mean([r["trust_score"] for r in trust_records]))

    return report, faithfulness_records, trust_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default="hybrid_cnn_transformer")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="runs/eval_report.json")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    report, faithfulness_records, trust_records = evaluate(cfg, args.model, args.checkpoint)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "report": report,
        "faithfulness_records": faithfulness_records,
        "trust_records": trust_records,
    }, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
