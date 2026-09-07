# Methodology

## 1. Research Gap

- Most XAI-integrated dermoscopic classifiers (2024-2026 literature) present explanation
  heatmaps only *qualitatively*; explanation **faithfulness is not verified against
  ground-truth lesion anatomy**.
- Reviews confirm no single XAI method (Grad-CAM, SHAP, LIME) gives a holistic explanation,
  yet **cross-method agreement is rarely measured or reported as a metric**.
- Predictive **uncertainty and explanation quality are treated separately** in existing work
  — no combined signal tells a clinician when to trust a prediction.
- Recent hybrid CNN-Transformer + dual-attention approaches (e.g., DermaScanAI) improve
  accuracy but still evaluate interpretability qualitatively.

## 2. Problem Statement

Deep learning dermoscopic classifiers achieve high accuracy, but their explanations are not
quantitatively validated against anatomical ground truth, and predictive confidence is not
fused with explanation reliability -- limiting clinical trustworthiness and safe deployment
support.

## 3. Objectives

1. Build a dermoscopic image-processing pipeline (artifact removal, normalization,
   leakage-safe splitting) on HAM10000.
2. Design a lightweight hybrid CNN-Transformer for 7-class lesion classification.
3. Integrate a multi-method XAI ensemble (Grad-CAM++, SHAP, Attention Rollout) with a
   quantitative agreement score.
4. Quantitatively validate explanation faithfulness against ISIC ground-truth segmentation
   masks (IoU, Pointing Game, Deletion/Insertion AUC).
5. Add uncertainty quantification (MC-Dropout) and fuse it with faithfulness/agreement into
   a single per-prediction Trust Score.
6. Benchmark against standard baselines and run ablations isolating each component's
   contribution.

## 4. Novelty

- A joint uncertainty x explanation-faithfulness x cross-method-agreement **Trust Score**,
  not just a heatmap.
- **Ground-truth-anchored** (segmentation-mask) quantitative faithfulness evaluation,
  replacing purely qualitative inspection common in cited 2024-2026 work.
- Automatic detection of unreliable explanations via cross-XAI-method disagreement.
- Lightweight architecture kept reproducible on modest compute while still supporting
  rigorous XAI evaluation.

## 5. Image-Processing Pipeline

1. **Deduplication & leakage-safe split** -- group by `lesion_id`, stratified patient-wise
   train/val/test split.
2. **Hair/artifact removal** -- morphological black-hat filtering + inpainting
   (DullRazor-style).
3. **Color/illumination normalization** -- gray-world correction + CLAHE on the luminance
   channel.
4. **Resize/normalize** to model input resolution, aspect-preserving pad.
5. **Lesion-focused cropping** using ISIC segmentation masks (ablated as optional).
6. **Augmentation + class-imbalance handling** -- flips/rotations/color-jitter + focal
   loss/class weighting for rare classes (DF, VASC).

## 6. Proposed AI Model

Hybrid CNN-Transformer with MC-Dropout uncertainty head:

- CNN stem (EfficientNet-B0 or compact ResNet) extracts local texture feature maps.
- Feature maps tokenized (patch flatten + linear projection + positional embedding) and
  passed through a lightweight Transformer encoder for global lesion-context modeling.
- Classification head: softmax over 7 classes.
- Dropout retained active at inference (MC-Dropout) for predictive uncertainty via repeated
  stochastic passes.
- Explanations extracted via Grad-CAM++ (CNN stem), Attention Rollout (Transformer), and
  SHAP (model-agnostic) -- combined into an agreement score.

## 7. Mathematical Formulation

- CNN features: `F = CNN_theta(X) in R^(HxWxC)`
- Tokenization: `z_i = W_p . flatten(p_i) + E_pos`
- Self-attention: `Attention(Q,K,V) = softmax(QK^T / sqrt(d)) . V`
- Prediction: `y_hat = softmax(W_cls . z_cls + b)`
- Class-imbalance loss (focal loss): `FL(p_t) = -alpha_t (1-p_t)^gamma log(p_t)`
- Uncertainty (MC-Dropout, T stochastic passes):
  `mu = (1/T) sum_t y_hat_t`, `sigma^2 = (1/T) sum_t (y_hat_t - mu)^2`
- Faithfulness: `IoU(SaliencyMap_bin, GT_mask)`; Deletion/Insertion AUC
- Cross-method agreement: Spearman rho or IoU between saliency maps of Grad-CAM++,
  Attention Rollout, SHAP
- Trust Score: `T = w1*(1 - sigma^2_norm) + w2*Faithfulness + w3*Agreement`, `sum(w_i) = 1`

## 8. Evaluation Metrics

- **Classification:** Accuracy, Balanced Accuracy, macro-Precision/Recall/F1, per-class
  sensitivity/specificity, macro-AUROC, Cohen's kappa, confusion matrix.
- **XAI faithfulness:** IoU/Dice vs. segmentation mask, Pointing Game accuracy,
  Deletion/Insertion AUC, cross-method agreement.
- **Uncertainty calibration:** Expected Calibration Error (ECE), reliability diagrams,
  predictive entropy.
- **Efficiency:** parameter count, FLOPs, inference latency.

## 9. Baseline Models

- ResNet50 (standard CNN baseline in the cited literature)
- EfficientNet-B0 (lightweight CNN baseline)
- Vanilla ViT-Base (transformer-only baseline)
- Proposed hybrid without uncertainty/multi-XAI (internal ablated variant)

## 10. Ablation Study

1. CNN-only vs. Hybrid CNN-Transformer (architecture contribution)
2. With/without preprocessing pipeline (hair removal, CLAHE)
3. Single-method (Grad-CAM only) vs. multi-method XAI ensemble
4. With/without uncertainty quantification (Trust Score utility)
5. Focal loss/class-weighting vs. plain cross-entropy
6. With/without segmentation-guided cropping

## References (verification checked at design time)

- Tschandl, P. et al. "The HAM10000 dataset..." *Scientific Data* (2018).
  https://www.nature.com/articles/sdata2018161
- "Explainable Artificial Intelligence for Skin Lesion Classification: A Comprehensive
  Review of Methods and Challenges." *MDPI* (2024/2025). https://www.mdpi.com/2227-7080/14/7/391
- "TIxAI: A Trustworthiness Index for eXplainable AI in skin lesions classification."
  *ScienceDirect*. https://www.sciencedirect.com/science/article/pii/S092523122500373X
- "DermaScanAI: an explainable hybrid deep learning framework..." *Scientific Reports*.
  https://www.nature.com/articles/s41598-026-46011-0
- "SkinSage XAI: An explainable deep learning solution for skin lesion diagnosis." *PMC*.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11671215/

Full IEEE-formatted reference list to be finalized in Stage 5 against the final
implementation and results.
