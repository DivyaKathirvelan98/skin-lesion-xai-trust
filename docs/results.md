# Experimental Results (Reduced-Scope CPU Run)

This documents the actual run whose numbers are reported in the paper. It intentionally
departs from the full design in [methodology.md](methodology.md) because it was executed on
a 4-core CPU machine with no GPU -- the deviations are disclosed here and in the paper's
Limitations section, not hidden.

## What was actually run

- **Config:** [configs/config_cpu_reduced.yaml](../configs/config_cpu_reduced.yaml)
  (vs. the full design in `configs/config.yaml`).
- **Data:** a stratified, lesion-wise (patient-wise) leakage-safe subset of HAM10000 --
  799 train / 199 val / 201 test images, drawn from the first 5,000 of the official 10,015
  images (`HAM10000_images_part_1.zip`, downloaded directly from the official Harvard
  Dataverse source, doi:10.7910/DVN/DBW86T; MD5-verified). All 7 diagnostic classes are
  present in every split. Ground-truth lesion segmentation masks are the full
  `HAM10000_segmentations_lesion_tschandl.zip` (10,015 masks, dermatologist-curated),
  covering the downloaded images.
- **Image size:** 160x160 (vs. 224x224 in the full design) -- reduces CNN/Transformer
  compute; ViT-Base uses `dynamic_img_size` to accept this resolution.
- **Epochs:** 6 (vs. 50) with early-stopping patience unused (run completed all 6).
- **Pretrained weights:** ImageNet-pretrained backbones (timm/Hugging Face Hub), same as
  the full design -- an initial run trained from scratch because of a local SSL
  certificate-trust issue; this was root-caused (a system-level intercepting root CA trusted
  by Windows but not by Python's bundled certifi) and fixed properly via the `truststore`
  package, which routes verification through the OS trust store rather than disabling
  verification. All reported numbers use pretrained backbones.
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
| **Hybrid CNN-Transformer (proposed)** | **0.761** | **0.457** | **0.466** | **0.913** | **0.494** | 0.119 |
| EfficientNet-B0 | 0.751 | 0.450 | 0.464 | 0.878 | 0.462 | 0.104 |
| ResNet50 | 0.711 | 0.180 | 0.168 | 0.842 | 0.187 | 0.120 |
| ViT-Base | 0.677 | 0.157 | 0.141 | 0.813 | 0.075 | 0.141 |

XAI/uncertainty/Trust Score (hybrid model only, faithfulness measured against ground-truth
lesion segmentation masks):

| Metric | Value |
|---|---|
| Mean faithfulness IoU (Grad-CAM++ vs. GT mask) | 0.163 |
| Pointing Game accuracy | 0.428 |
| Mean Trust Score | 0.250 |

## Honest reading of these numbers

- The proposed hybrid model outperforms all three baselines on every classification metric
  except ECE (where EfficientNet-B0 is marginally better-calibrated). This supports the
  architectural design choice but should **not** be over-claimed as a large-scale result --
  n=201 test images is small, and a single run (no repeated seeds/cross-validation) means no
  confidence intervals are available.
- Balanced accuracy and macro-F1 are much lower than raw accuracy for every model, reflecting
  HAM10000's known severe class imbalance (nv dominates); ResNet50 and ViT-Base in
  particular appear to lean heavily on the majority class (kappa 0.19 and 0.08).
- Faithfulness IoU (0.16) is modest in absolute terms -- expected given only 6 epochs of
  fine-tuning and a small training set; it is reported honestly rather than cherry-picked or
  inflated.
- These are described in the paper as a **feasibility-scale pilot study**, with full-scale
  training (GPU, 50 epochs, full 10,015-image dataset, all three XAI methods) left as the
  natural next step -- the Colab notebook
  ([notebooks/run_experiments.ipynb](../notebooks/run_experiments.ipynb)) is provided for
  exactly that.

## Reproducing this run

```bash
python -m src.preprocessing.dataset_split --metadata data/raw/HAM10000_metadata.csv --out data/splits
python scripts/subsample_splits.py --train-n 800 --val-n 200 --test-n 200
for model in hybrid_cnn_transformer resnet50 efficientnet_b0 vit_base; do
    python -m src.train --config configs/config_cpu_reduced.yaml --model "$model"
    python -m src.evaluate --config configs/config_cpu_reduced.yaml --model "$model" \
        --checkpoint "runs/$model/best.pt" --out "runs/eval_report_$model.json"
done
```
