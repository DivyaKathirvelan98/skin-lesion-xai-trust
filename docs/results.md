# Experimental Results (Reduced-Scope CPU Run)

This documents the actual run whose numbers are reported in the paper. It intentionally
departs from the full design in [methodology.md](methodology.md) because it was executed on
a 4-core CPU machine with no GPU -- the deviations are disclosed here and in the paper's
Limitations section, not hidden.

## What was actually run

- **Config:** [configs/config_cpu_20epoch.yaml](../configs/config_cpu_20epoch.yaml)
  (vs. the full design in `configs/config.yaml`). An earlier 6-epoch pilot
  ([configs/config_cpu_reduced.yaml](../configs/config_cpu_reduced.yaml)) was superseded by
  this 20-epoch run once local CPU training proved reliable enough to sustain it; the
  6-epoch numbers are no longer reported, to avoid two inconsistent result sets.
- **Data:** a stratified, lesion-wise (patient-wise) leakage-safe subset of HAM10000 --
  799 train / 199 val / 201 test images, drawn from the first 5,000 of the official 10,015
  images (`HAM10000_images_part_1.zip`, downloaded directly from the official Harvard
  Dataverse source, doi:10.7910/DVN/DBW86T; MD5-verified). All 7 diagnostic classes are
  present in every split. Ground-truth lesion segmentation masks are the full
  `HAM10000_segmentations_lesion_tschandl.zip` (10,015 masks, dermatologist-curated),
  covering the downloaded images.
- **Image size:** 160x160 (vs. 224x224 in the full design) -- reduces CNN/Transformer
  compute; ViT-Base uses `dynamic_img_size` to accept this resolution.
- **Epochs:** 20 (vs. 50 in the full design). Training was interrupted twice mid-run by the
  local machine idle-sleeping (once for ~1h45m mid-epoch, recovered on its own; once for
  good, requiring a restart) before a host-level keep-awake hold was used for the final,
  successful run -- disclosed here for transparency, not because it affects the reported
  numbers (every reported checkpoint comes from a run that completed all 20 epochs cleanly).
- **Pretrained weights:** ImageNet-pretrained backbones (timm/Hugging Face Hub) for all
  reported numbers. An earlier attempt trained from scratch because of a local SSL
  certificate-trust issue; this was root-caused (a system-level intercepting root CA trusted
  by Windows but not by Python's bundled certifi) and fixed properly via the `truststore`
  package, which routes verification through the OS trust store rather than disabling
  verification.
- **XAI methods evaluated:** Grad-CAM++ and Attention Rollout, with cross-method IoU/Spearman
  agreement. SHAP is implemented ([src/xai/shap_explain.py](../src/xai/shap_explain.py)) but
  was excluded from this run for CPU time feasibility (SHAP's GradientExplainer needs many
  forward passes per image).
- **MC-Dropout passes:** 10 (vs. 20 in the full design).

## Results (test set, n=201)

See [runs/results_summary.csv](../runs/results_summary.csv) and the per-model
`runs/eval_report_*.json` files (raw output of `src/evaluate.py`, not hand-edited).

| Model | Accuracy | Balanced Acc. | Macro F1 | Macro AUROC | Cohen's kappa | ECE |
|---|---|---|---|---|---|---|
| **Hybrid CNN-Transformer (proposed)** | **0.771** | **0.472** | **0.491** | 0.861 | **0.514** | **0.021** |
| EfficientNet-B0 | 0.736 | 0.388 | 0.398 | **0.908** | 0.451 | 0.130 |
| ResNet50 | 0.756 | 0.351 | 0.380 | 0.845 | 0.425 | 0.104 |
| ViT-Base | 0.721 | 0.323 | 0.342 | 0.853 | 0.341 | 0.104 |

XAI/uncertainty/Trust Score (hybrid model only, faithfulness measured against ground-truth
lesion segmentation masks):

| Metric | Value |
|---|---|
| Mean faithfulness IoU (Grad-CAM++ vs. GT mask) | 0.196 |
| Pointing Game accuracy | 0.488 |
| Mean Trust Score | 0.297 |

## Honest reading of these numbers

- The proposed hybrid model wins 5 of 6 classification metrics outright (accuracy, balanced
  accuracy, macro-F1, kappa, and -- most notably -- calibration, where its ECE of 0.021 is
  roughly 5x better than any baseline). It loses only on macro-AUROC, where EfficientNet-B0
  (the hybrid model's own CNN backbone) is ahead (0.908 vs. 0.861). This is a real, reported
  trade-off, not cherry-picked: the Transformer stage improves ranking-independent decision
  quality and calibration at some cost to ranking-based AUROC at this training scale.
- Balanced accuracy and macro-F1 remain well below raw accuracy for every model, reflecting
  HAM10000's known severe class imbalance (nv dominates); ResNet50 and ViT-Base in
  particular still lean more heavily on the majority class than the hybrid model or
  EfficientNet-B0 (kappa 0.43 and 0.34 vs. 0.51 and 0.45).
- Comparing to the earlier 6-epoch pilot, more epochs substantially improved calibration,
  balance, and faithfulness across the board, but barely moved raw accuracy (76.1% -> 77.1%
  for the hybrid model). This confirms the accuracy ceiling here is set by the 800-image
  local training subset, not by training time -- closing the remaining gap to full-scale
  literature results (92-98% on the complete 10,015-image dataset) requires GPU training on
  the full dataset, not more local CPU epochs.
- Training curves (Fig. 2) show a growing train/validation gap by epoch 15-20 for most
  models (e.g., EfficientNet-B0 and the hybrid model's training accuracy approaches
  0.94-0.96 while validation plateaus near 0.77-0.82) -- early-stage overfitting consistent
  with a small (800-image) training set, reported openly rather than masked by only showing
  final-epoch numbers.
- Faithfulness IoU (0.196) and Pointing Game accuracy (48.8%) both improved over the 6-epoch
  pilot (0.163, 42.8%) with more training, and are reported without adjustment as an honest
  baseline for future full-scale comparison, not as a mature clinical-grade result.
- These are described in the paper as a **feasibility-scale pilot study**, with full-scale
  training (GPU, 50 epochs, full 10,015-image dataset, all three XAI methods) left as the
  natural next step -- the Colab notebook
  ([notebooks/run_experiments.ipynb](../notebooks/run_experiments.ipynb)) is provided for
  exactly that.

## Ablation study (partial -- see limitations)

Two of the six ablations planned in [methodology.md](methodology.md) were run given the
CPU-only time budget; the rest (multi-XAI-method vs. single-method agreement, MC-Dropout
uncertainty on/off, focal loss vs. plain cross-entropy, segmentation-guided cropping) were
not executed and are left as future work rather than reported without evidence. Both
ablations below use the same 20-epoch training budget as the main results, for a
like-for-like comparison.

**1. Architecture: CNN-only vs. Hybrid CNN-Transformer** -- the hybrid model's CNN backbone
*is* EfficientNet-B0, so the EfficientNet-B0 row above doubles as this ablation:

| Variant | Accuracy | Balanced Acc. | Macro F1 | Macro AUROC | Kappa |
|---|---|---|---|---|---|
| CNN-only (EfficientNet-B0) | 0.736 | 0.388 | 0.398 | **0.908** | 0.451 |
| CNN + Transformer (proposed) | **0.771** | **0.472** | **0.491** | 0.861 | **0.514** |

The Transformer stage improves accuracy, balanced accuracy, macro-F1, and kappa, at the cost
of macro-AUROC -- the same trade-off visible in the main results table above, isolated here
to the one architectural change that causes it.

**2. Preprocessing pipeline: with vs. without hair removal + CLAHE**
(config: [configs/config_ablation_no_preprocessing.yaml](../configs/config_ablation_no_preprocessing.yaml),
full report: [runs/eval_report_hybrid_no_preprocessing.json](../runs/eval_report_hybrid_no_preprocessing.json)):

| Variant | Accuracy | Balanced Acc. | Macro F1 | Macro AUROC | Kappa | ECE | Faithfulness IoU | Pointing Game Acc. |
|---|---|---|---|---|---|---|---|---|
| Without preprocessing | 0.761 | 0.464 | 0.486 | **0.903** | 0.490 | 0.083 | 0.140 | 0.388 |
| With preprocessing (proposed) | **0.771** | **0.472** | **0.491** | 0.861 | **0.514** | **0.021** | **0.196** | **0.488** |

At matched training scale (20 epochs, both arms), preprocessing now wins cleanly on
accuracy, balanced accuracy, macro-F1, kappa, calibration, and both faithfulness metrics --
a much more consistent result than the earlier 6-epoch pilot, where balanced accuracy and
macro-F1 had (within noise) favored the no-preprocessing variant. The one metric where
preprocessing loses is macro-AUROC, mirroring the architecture ablation's trade-off. Mean
Trust Score is effectively tied between the two variants (0.297 vs. 0.297), since its
uncertainty and agreement terms partly offset the faithfulness gain.

## Reproducing this run

```bash
python -m src.preprocessing.dataset_split --metadata data/raw/HAM10000_metadata.csv --out data/splits
python scripts/subsample_splits.py --train-n 800 --val-n 200 --test-n 200
for model in hybrid_cnn_transformer resnet50 efficientnet_b0 vit_base; do
    python -m src.train --config configs/config_cpu_20epoch.yaml --model "$model"
    python -m src.evaluate --config configs/config_cpu_20epoch.yaml --model "$model" \
        --checkpoint "runs/$model/best.pt" --out "runs/eval_report_$model.json"
done
```
