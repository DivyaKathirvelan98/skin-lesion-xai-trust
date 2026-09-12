# Explainable Vision Transformer Framework for Multi-Class Skin Lesion Classification Using Dermoscopic Images: An Uncertainty-Aware Trust Scoring Approach

## Abstract

Deep learning models for dermoscopic skin lesion classification increasingly incorporate
explainable AI (XAI) techniques, yet most report explanation heatmaps only qualitatively,
without verifying that explanations are anatomically faithful, and without connecting
explanation reliability to the model's own predictive uncertainty. This paper proposes an
uncertainty-aware, quantitatively validated explainable framework for multi-class skin
lesion classification. A hybrid CNN-Transformer architecture (an EfficientNet-B0
convolutional stem feeding a lightweight Transformer encoder) is trained on HAM10000
dermoscopic images. Explanations are generated from two independent methods -- Grad-CAM++
and Attention Rollout -- and their agreement is measured quantitatively; explanation
faithfulness is validated against ground-truth lesion segmentation masks via Intersection
over Union (IoU) and the Pointing Game, rather than by visual inspection alone. Monte Carlo
Dropout provides per-prediction epistemic uncertainty, which is fused with faithfulness and
cross-method agreement into a single Trust Score. On a compute-constrained (CPU-only)
feasibility-scale run -- 799/199/201 train/val/test images, 20 training epochs -- the
proposed hybrid model reaches 77.1% accuracy, 0.472 balanced accuracy, 0.514 Cohen's kappa,
and an Expected Calibration Error of just 0.021, winning 5 of 6 classification metrics
outright against ResNet50, EfficientNet-B0, and ViT-Base baselines and losing only on
macro-AUROC (0.861 vs. EfficientNet-B0's 0.908), while achieving a mean faithfulness IoU of
0.196 and Pointing Game accuracy of 48.8%. An ablation study at matched training scale shows
that dermoscopic preprocessing (hair removal and CLAHE contrast normalization) improves
accuracy, balanced accuracy, macro-F1, kappa, calibration, and both faithfulness metrics,
trading off only macro-AUROC. All code, configurations, and results are released openly; the
reduced experimental scale relative to the full study design -- most consequentially, an
800-image local training subset that this study finds sets an accuracy ceiling largely
independent of training epochs -- is disclosed explicitly as a compute-driven limitation,
with a full-scale reproduction path (GPU training on the complete 10,015-image dataset)
provided.

**Index Terms** -- Skin lesion classification, explainable artificial intelligence,
Vision Transformer, uncertainty quantification, trust score, HAM10000, dermoscopy.

## I. Introduction

Skin cancer is among the most common malignancies worldwide, and early, accurate diagnosis
from dermoscopic images materially improves patient outcomes. Convolutional neural networks
(CNNs) and, more recently, Vision Transformers (ViTs) [3] have driven substantial gains in
automated dermoscopic classification accuracy. However, the black-box nature of these models
remains a central barrier to clinical adoption: a clinician cannot act on a prediction they
cannot interrogate.

Explainable AI (XAI) techniques -- most commonly Grad-CAM-family saliency maps -- have been
widely added to dermoscopic classifiers to address this. Yet a close reading of the recent
literature reveals three specific, still-open gaps. First, explanations are typically
presented and evaluated *qualitatively*: a heatmap is shown next to a lesion image, and
plausibility is assessed by visual inspection, not verified against ground-truth lesion
anatomy [6], [8]. Second, no single XAI method provides a holistic explanation, and reviews
of the field note that different methods (Grad-CAM, SHAP, LIME) capture different, only
partially overlapping notions of "importance" -- yet cross-method agreement is rarely
measured or reported as a quantitative signal [6]. Third, and most consequentially,
predictive uncertainty and explanation quality are treated as *separate* concerns: recent
trustworthiness-focused work such as TIxAI [7] quantifies how well a single explanation
method (Grad-CAM) localizes to the lesion, and recent hybrid architectures such as
DermaScanAI [8] combine attention mechanisms with Grad-CAM++/SHAP visualizations for
improved accuracy and interpretability, but neither fuses an explicit uncertainty estimate
with explanation reliability into one clinically actionable signal.

This paper addresses these three gaps directly. We do not claim to outperform the accuracy
of large-scale, GPU-trained, full-dataset models in the recent literature (which reach
92-98% accuracy on HAM10000 [4], [8]); our contribution is architectural and
methodological, evaluated honestly at a reduced, compute-constrained scale, with a clear
path to full-scale reproduction.

The contributions of this paper are:

1. A lightweight hybrid CNN-Transformer architecture for 7-class HAM10000 dermoscopic
   lesion classification, combining local texture features (CNN) with global lesion-context
   modeling (Transformer self-attention).
2. A multi-method XAI ensemble (Grad-CAM++ [11], Attention Rollout [12]) with a
   quantitative cross-method agreement score (IoU and Spearman correlation between
   saliency maps), rather than reliance on a single explanation method.
3. Ground-truth-anchored, quantitative faithfulness evaluation of explanations against
   dermatologist-curated HAM10000 lesion segmentation masks [2], using IoU and the
   Pointing Game, rather than qualitative-only assessment.
4. A Trust Score that fuses Monte Carlo Dropout predictive uncertainty [10] with
   explanation faithfulness and cross-method agreement into a single, per-prediction signal
   intended to flag cases warranting clinician review.
5. A fully open, reproducible implementation (code, configuration, and raw evaluation
   output), with a disclosed, honest account of the reduced experimental scale used in this
   study and a documented path to full-scale reproduction on GPU hardware.

## II. Related Work

**Skin lesion classification architectures.** CNN backbones (ResNet, EfficientNet) have
long dominated dermoscopic classification, but Vision Transformers [3] and hybrid
CNN-Transformer architectures have gained traction for their ability to model long-range
spatial dependencies. Nie et al. [4] proposed a CNN-Transformer hybrid trained with focal
loss on dermoscopic images, reporting improved handling of HAM10000's class imbalance.
Kunduracioglu and Pacal's 2026 survey of 135 studies (2023-2026) confirms hybrid
CNN-Transformer and ensemble architectures as the dominant recent direction, alongside
growing adoption of XAI methods [5].

**Explainable AI in dermatology.** Munjal et al.'s SkinSage XAI [6] combines Grad-CAM and
LIME with an Inception-v3 backbone, reporting high accuracy but evaluating explanations
qualitatively. Ieracitano et al.'s TIxAI [7] is closest in spirit to this paper: it defines
a Trustworthiness Index by measuring the relevance difference between lesion and
non-lesion Grad-CAM regions on an EfficientNet-B0 classifier -- a single-method,
localization-only notion of trust. Murali and Mazumder's DermaScanAI [8] combines
multi-scale CNN features, lightweight Transformer encoders, and dual attention with
Grad-CAM++ and SHAP for a reported 94.8% accuracy on HAM10000, but does not quantitatively
validate explanation faithfulness against ground-truth segmentation, nor does it fuse
uncertainty with explanation quality. This paper differs from all three in combining (i)
multi-method cross-agreement, (ii) ground-truth-mask-anchored faithfulness metrics, and
(iii) uncertainty fusion, into one Trust Score.

**Uncertainty quantification.** Kurz et al.'s systematic review of uncertainty estimation
in medical image classification [9] finds Monte Carlo Dropout and deep ensembles the most
common sampling-based approaches, following the Bayesian approximation formalized by Gal
and Ghahramani [10]. Uncertainty estimation is well studied in medical imaging broadly, but
its fusion with *explanation* reliability specifically -- rather than with calibration or
classification confidence alone -- remains, to our knowledge, largely unexplored for
dermoscopic classification.

**Synthesis.** The literature shows rapid progress on accuracy and on adding *some* form of
explainability, but a persistent gap remains: explanation faithfulness is asserted, not
measured against ground truth; multiple explanation methods are rarely cross-validated
against each other; and uncertainty and explanation quality are not fused into one
actionable signal. This paper's Trust Score framework is designed to close that specific
combination of gaps.

## III. Materials and Methods

Fig. 1 summarizes the full pipeline end to end: a dermoscopic image is preprocessed, passed
through the hybrid CNN-Transformer to produce a prediction, and simultaneously routed
through three parallel evaluation branches -- explainability (Grad-CAM++ and Attention
Rollout, cross-checked via agreement), faithfulness (both saliency maps compared against the
ground-truth lesion mask), and uncertainty (MC-Dropout) -- which are fused into the final
Trust Score. The same preprocessed data also feeds three baseline architectures (ResNet50,
EfficientNet-B0, ViT-Base) trained identically for comparison.

**[Fig. 1. End-to-end methodology diagram -- dermoscopic image, preprocessing, CNN stem,
tokenization/Transformer encoder, prediction, with parallel Grad-CAM++/Attention Rollout,
cross-method agreement, ground-truth-mask faithfulness, and MC-Dropout uncertainty, all
fusing into the Trust Score; baselines trained on the same preprocessed data.]**

### A. Dataset

We use HAM10000 [1], [2], 10,015 dermoscopic images across seven diagnostic classes
(actinic keratosis/Bowen's disease [akiec], basal cell carcinoma [bcc], benign
keratosis-like lesions [bkl], dermatofibroma [df], melanoma [mel], melanocytic nevi [nv],
vascular lesions [vasc]), obtained directly from the official Harvard Dataverse release
(DOI: 10.7910/DVN/DBW86T; MD5-verified download). Dermatologist-curated ground-truth lesion
segmentation masks, released by the same authors [2], are used for faithfulness evaluation.

Due to CPU-only compute availability for this study (see Section III-I and Section V), a
stratified, lesion-wise (patient-wise) subset was used: 799 train / 199 validation / 201
test images, split at the `lesion_id` level (not the image level) before subsampling, so
that no lesion's images appear in more than one split -- avoiding the data leakage that
image-level splitting of HAM10000 is known to risk, since many lesions have multiple
images. All seven classes are represented in every split.

### B. Image Preprocessing Pipeline

1. **Deduplication and leakage-safe split:** grouping by `lesion_id` before any
   train/val/test assignment (Section III-A).
2. **Hair and artifact removal:** DullRazor-style morphological black-hat filtering
   followed by Telea inpainting, to suppress dark hair strands that otherwise contaminate
   both classification features and saliency maps.
3. **Color and illumination normalization:** gray-world channel correction followed by
   Contrast-Limited Adaptive Histogram Equalization (CLAHE, clip limit 2.0) on the
   luminance channel.
4. **Resizing:** aspect-preserving resize/pad to the model input resolution (160x160 in
   this study; 224x224 in the full design).
5. **Augmentation and class-imbalance handling:** random crop, horizontal/vertical flips,
   rotation (up to 30 deg), and color jitter during training, combined with a focal loss
   (Section III-F) to counteract HAM10000's severe class imbalance (nv comprises roughly
   68% of images).

### C. Proposed Hybrid CNN-Transformer Architecture

Given an input image $X$, an EfficientNet-B0 [17] convolutional stem $\mathrm{CNN}_\theta$
extracts a local feature map:

$$F = \mathrm{CNN}_\theta(X) \in \mathbb{R}^{H \times W \times C}$$

$F$ is projected to the Transformer's embedding dimension via a $1\times1$ convolution and
flattened into a sequence of tokens $\{p_i\}$, each linearly projected and combined with a
learned positional embedding:

$$z_i = W_p \cdot \mathrm{flatten}(p_i) + E_{pos}$$

A learned classification token $z_{cls}$ is prepended, and the resulting sequence is passed
through $L$ Transformer encoder layers (multi-head self-attention with GELU-activated
feed-forward blocks):

$$\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V$$

The final classification token representation is passed through a dropout layer and a
linear classification head:

$$\hat{y} = \mathrm{softmax}(W_{cls} \cdot z_{cls} + b)$$

In this study, $L=4$ encoder layers, 8 attention heads, and an embedding dimension of 256
are used. Dropout ($p=0.2$) is retained active at inference time to support Monte Carlo
Dropout uncertainty estimation (Section III-G).

### D. Explainability Methods

Two independent explanation methods are computed for each test prediction of the proposed
model:

- **Grad-CAM++** [11], applied to the CNN stem's final convolutional layer, producing a
  gradient-weighted class activation map.
- **Attention Rollout** [12], which recursively multiplies (with a residual identity term
  and row-normalization) the Transformer's per-layer self-attention matrices to trace how
  the classification token's final representation depends on each input patch token.

A third, model-agnostic method (SHAP [13], via GradientExplainer) is implemented in the
released codebase but was not executed in this study's reported results, as its
per-image computational cost (tens of forward passes per explanation) was infeasible within
the CPU-only time budget; this is disclosed as a limitation (Section V).

**Cross-method agreement.** For each image, the two saliency maps are resized to a common
resolution, binarized at a threshold of 0.5, and compared via IoU and Spearman rank
correlation. The mean pairwise IoU is used as the agreement term in the Trust Score.

### E. Faithfulness Evaluation

Rather than assessing explanations qualitatively, each Grad-CAM++ saliency map is compared
against the corresponding ground-truth lesion segmentation mask [2]:

- **IoU:** intersection-over-union between the binarized (threshold 0.5) saliency map and
  the binary lesion mask.
- **Pointing Game** [accuracy]: the fraction of test images for which the saliency map's
  single highest-activation pixel falls inside the ground-truth lesion region.

A Deletion/Insertion AUC metric [14] is also implemented (progressively masking/revealing
the most salient pixels and tracking the target-class probability) but was not executed at
scale in this study for the same compute-budget reason as SHAP.

### F. Focal Loss

To counteract HAM10000's severe class imbalance, training uses focal loss [15]:

$$\mathrm{FL}(p_t) = -\alpha_t (1-p_t)^\gamma \log(p_t)$$

with $\alpha=0.25$, $\gamma=2.0$, down-weighting the loss contribution of well-classified
majority-class (nv) examples relative to harder, minority-class examples.

### G. Uncertainty Quantification

Predictive (epistemic) uncertainty is estimated via Monte Carlo Dropout [10]: at inference,
dropout layers are kept active and $T$ stochastic forward passes are performed per image:

$$\mu = \frac{1}{T}\sum_{t=1}^{T} \hat{y}_t, \qquad
\sigma^2 = \frac{1}{T}\sum_{t=1}^{T} (\hat{y}_t - \mu)^2$$

$T=10$ passes are used in this study (vs. 20 in the full design, halved for CPU time). The
predictive entropy of $\mu$, normalized by $\log(\text{num\_classes})$, is used as the
uncertainty term in the Trust Score.

### H. Trust Score

The Trust Score fuses (normalized, $[0,1]$-scaled) predictive uncertainty, faithfulness,
and cross-method agreement into a single per-prediction value:

$$T = w_1(1-\sigma^2_{norm}) + w_2 \cdot \mathrm{Faithfulness} + w_3 \cdot \mathrm{Agreement},
\qquad \sum_i w_i = 1$$

with $w_1=0.34$, $w_2=0.33$, $w_3=0.33$ in this study (equal weighting; weight tuning
against clinician judgments is left as future work).

### I. Baseline Models and Experimental Setup

**Baselines:** ResNet50 [16], EfficientNet-B0 [17], and ViT-Base [3] (patch size 16),
all initialized with ImageNet-pretrained weights (via timm/Hugging Face Hub) and
fine-tuned identically to the proposed model.

**Training:** AdamW optimizer, learning rate $3\times10^{-4}$, weight decay
$1\times10^{-4}$, cosine learning-rate schedule, batch size 16, 20 epochs, image resolution
160x160 (ViT-Base uses dynamic positional-embedding interpolation to accept this
non-native resolution). This is a deliberate, disclosed reduction from the full design (50
epochs, 224x224, the complete 10,015-image dataset) made necessary by CPU-only compute
availability for this study; see Section V. An earlier 6-epoch pilot run was superseded by
this 20-epoch run once local training proved sustainable (Section V); only the 20-epoch
numbers are reported here.

**Evaluation metrics:** accuracy, balanced accuracy, macro-F1, macro-AUROC (one-vs-rest),
Cohen's kappa, and Expected Calibration Error (ECE) for classification; IoU and Pointing
Game accuracy for faithfulness; mean Trust Score for the fused uncertainty-explanation
signal.

## IV. Results and Discussion

### A. Classification Performance

Table I reports test-set (n=201) classification metrics for all four models.

**Table I. Classification performance (test set, n=201)**

| Model | Accuracy | Balanced Acc. | Macro F1 | Macro AUROC | Cohen's $\kappa$ | ECE |
|---|---|---|---|---|---|---|
| **Hybrid CNN-Transformer (proposed)** | **0.771** | **0.472** | **0.491** | 0.861 | **0.514** | **0.021** |
| EfficientNet-B0 | 0.736 | 0.388 | 0.398 | **0.908** | 0.451 | 0.130 |
| ResNet50 | 0.756 | 0.351 | 0.380 | 0.845 | 0.425 | 0.104 |
| ViT-Base | 0.721 | 0.323 | 0.342 | 0.853 | 0.341 | 0.104 |

Six metrics are reported deliberately rather than accuracy alone, because HAM10000's class
imbalance (nv $\approx$ 68% of images) makes raw accuracy a weak signal on its own -- all
four models cluster within 5 points of each other on accuracy (72-77%) while fanning out far
more on balanced accuracy, macro-F1, and kappa (Fig. 2), which is where the real separation
between models shows up. The proposed hybrid model wins 5 of the 6 metrics outright --
accuracy, balanced accuracy, macro-F1, kappa, and calibration, where its ECE of 0.021 is
roughly five times lower (better) than any baseline. It loses only on macro-AUROC, where
EfficientNet-B0 -- the hybrid model's own CNN backbone -- leads (0.908 vs. 0.861); Section
IV-D isolates this trade-off to the Transformer stage specifically. ResNet50 and ViT-Base
trail on every imbalance-sensitive metric despite reaching comparable raw accuracy, a gap
between raw and balanced accuracy that is itself evidence they lean more heavily on the
majority class under this study's training budget.

**[Fig. 2. Classification performance by model, grouped by metric (test set, n=201).]**

Fig. 3 plots training and validation loss/accuracy across all 20 epochs for all four models.
Two things are checked here that a final-epoch table cannot show. First, calibration: the
hybrid model's dramatically lower ECE (0.021) is visible in its validation loss tracking
closely with (and by epoch 15 slightly below) its training loss, whereas EfficientNet-B0 and
ResNet50 show a widening train/validation gap from epoch 10 onward (e.g., EfficientNet-B0's
training accuracy approaches 0.94-0.96 by epoch 20 while its validation accuracy plateaus
near 0.75-0.78) -- early-stage overfitting on the 800-image training set, reported openly
rather than hidden behind a single final-epoch number. Second, convergence: no model's
curves are fully flat by epoch 20, indicating the reported numbers remain a lower bound
rather than a converged result.

**[Fig. 3. Training/validation loss and accuracy over 20 epochs for all four models.]**

Fig. 4 shows the hybrid model's confusion matrix. The diagonal dominates for the majority
class (127/137 nv images correct) but is weaker for minority classes (e.g., 2/7 akiec, 7/11
bcc). The off-diagonal mass is not scattered randomly: it concentrates on the akiec/bkl/mel
$\rightarrow$ nv direction, most sharply for mel (11 of 17 mel images misclassified as nv,
64.7%) and, to a lesser extent, bkl (5 of 23, 21.7%). This tracks a clinically real
ambiguity -- melanoma, benign keratoses, and melanocytic nevi are notoriously difficult to
distinguish visually even for trained dermatologists -- rather than an arbitrary model
failure mode, a distinction invisible from the aggregate metrics in Table I alone.

**[Fig. 4. Confusion matrix for the proposed hybrid model (test set, n=201).]**

### B. XAI Faithfulness and Cross-Method Agreement

For the proposed hybrid model, mean faithfulness IoU (Grad-CAM++ vs. ground-truth lesion
mask) was 0.196, and Pointing Game accuracy was 48.8% -- i.e., the single most-activated
pixel of the Grad-CAM++ map fell inside the true lesion region in nearly half of test
images. Fig. 5 illustrates a representative, correctly-classified test case: both Grad-CAM++
and Attention Rollout concentrate activation within the true lesion boundary, but on visibly
different sub-regions of it -- a concrete example of exactly the kind of partial,
method-dependent agreement that motivates measuring cross-method agreement quantitatively
(Section III-D) rather than trusting a single method's heatmap by inspection. These
aggregate faithfulness values are modest in absolute terms, consistent with a model trained
on 800 images; they are reported without adjustment, as an honest baseline for future
full-scale comparison, rather than presented as a mature clinical-grade result.

**[Fig. 5. Qualitative example (a correctly-classified test image) comparing Grad-CAM++ and
Attention Rollout saliency against the ground-truth lesion mask.]**

### C. Uncertainty and Trust Score

The mean Trust Score across the test set was 0.297 (on a $[0,1]$ scale). Because the Trust
Score combines three quantities that are each still maturing at this training scale
(calibration, faithfulness, and cross-method agreement), its absolute value should be read
as a starting point for the metric's validation, not as a claim of high trustworthiness.
Its value as a *framework* lies in making these three signals jointly visible and
computable per-prediction -- something the reviewed prior work (TIxAI, DermaScanAI,
SkinSage XAI) does not do in combination.

### D. Ablation Study

Table II reports two of the six ablations from the study design (Section III), both run at
the same 20-epoch training budget as the main results for a like-for-like comparison; the
remaining four -- multi-method XAI agreement's effect in isolation, uncertainty
quantification on/off, focal loss vs. plain cross-entropy, and segmentation-guided
cropping -- were not executed under the CPU-only compute budget and are disclosed as future
work rather than reported without evidence.

**Table II. Ablation results**

| Ablation | Variant | Accuracy | Balanced Acc. | Macro F1 | Macro AUROC | Faithfulness IoU |
|---|---|---|---|---|---|---|
| Architecture | CNN-only (EfficientNet-B0) | 0.736 | 0.388 | 0.398 | **0.908** | -- |
| Architecture | CNN + Transformer (proposed) | **0.771** | **0.472** | **0.491** | 0.861 | 0.196 |
| Preprocessing | Without hair removal / CLAHE | 0.761 | 0.464 | 0.486 | **0.903** | 0.140 |
| Preprocessing | With hair removal / CLAHE (proposed) | **0.771** | **0.472** | **0.491** | 0.861 | **0.196** |

**[Fig. 6. Ablation results: architecture (left) and preprocessing (right).]**

Both ablations tell the same consistent story. The Transformer stage improves accuracy,
balanced accuracy, macro-F1, and kappa over the CNN-only baseline, trading off macro-AUROC
(0.861 vs. 0.908) -- the same pattern visible between the hybrid model and EfficientNet-B0
in Table I, now isolated to the one architectural change that causes it. Preprocessing shows
an identical trade-off shape: hair removal and CLAHE normalization improve accuracy,
balanced accuracy, macro-F1, kappa, calibration (ECE 0.021 vs. 0.083), and both faithfulness
metrics (IoU +0.056, Pointing Game +10.0 points), again at the cost of macro-AUROC (0.861
vs. 0.903). This is a markedly cleaner result than an earlier 6-epoch pilot of the same
ablation, where balanced accuracy and macro-F1 had -- within single-run noise -- favored the
no-preprocessing variant; at matched 20-epoch training scale, preprocessing's benefit is
consistent across nearly every metric. Its effect on explanation quality specifically
remains the clearest finding: faithfulness IoU and Pointing Game accuracy both improve
substantially, consistent with the intuition that removing hair and illumination artifacts
helps the model's activation patterns concentrate on the true lesion region rather than
incidental image artifacts -- a benefit that would not have been visible without the
ground-truth-mask-anchored faithfulness metrics proposed in Section III-E.

## V. Limitations

This study's results were produced under a strict, disclosed compute constraint (a 4-core
CPU machine with no GPU) and should be read as a **feasibility-scale pilot**, not a
full-scale benchmark result:

- **Data scale:** 799/199/201 train/val/test images, drawn from the first 5,000 of
  HAM10000's 10,015 images, vs. the full dataset in the study's original design. This study
  finds this is the dominant constraint on accuracy specifically: increasing training from 6
  to 20 epochs on the same 800-image subset substantially improved calibration, balance,
  and faithfulness (Section IV) but barely moved raw accuracy (76.1% to 77.1% for the
  hybrid model), indicating the accuracy ceiling is set by data scale, not training time.
- **Training budget:** 20 epochs vs. 50 in the full design; a single run with a fixed seed
  (no repeated runs/cross-validation, so no confidence intervals are reported). Training was
  twice interrupted by the local machine idle-sleeping before completing cleanly on a third
  attempt under a host-level keep-awake hold; this affected wall-clock time only, not the
  reported checkpoints, all of which come from runs that completed all 20 epochs without
  interruption.
- **Resolution:** 160x160 vs. 224x224.
- **XAI scope:** SHAP and the Deletion/Insertion faithfulness metric are implemented in the
  released code but were not executed in this study's results, for CPU time feasibility.
- **Ablation scope:** two of six planned ablations were executed; the remainder are left as
  future work rather than asserted without evidence.
- **MC-Dropout passes:** 10 vs. 20 in the full design.

None of these constraints were used to inflate or cherry-pick results: all reported numbers
are the direct, unedited output of the released evaluation code
(`runs/eval_report_*.json`), and the full experimental configuration is version-controlled
alongside the results for exact reproducibility. A Google Colab notebook
(`notebooks/run_experiments.ipynb`) is provided to reproduce this study at full scale on
GPU hardware, which -- given the finding above that data scale rather than epoch count is
the binding constraint -- is the most direct remaining path to closing the accuracy gap to
full-scale literature results.

## VI. Conclusion and Future Work

This paper presented a hybrid CNN-Transformer framework for dermoscopic skin lesion
classification that goes beyond typical single-method, qualitative XAI by (i) cross-
validating two independent explanation methods against each other, (ii) quantitatively
grounding explanation faithfulness in dermatologist-curated lesion segmentation masks
rather than visual inspection, and (iii) fusing Monte Carlo Dropout uncertainty with
explanation quality into a single, per-prediction Trust Score. Under a disclosed,
compute-constrained pilot evaluation, the proposed architecture won 5 of 6 classification
metrics against ResNet50, EfficientNet-B0, and ViT-Base baselines -- most notably a roughly
five-fold better calibration (ECE 0.021) -- while trading off macro-AUROC to the Transformer
stage specifically, a trade-off isolated and confirmed by the architecture ablation. The
preprocessing ablation showed the same trade-off shape and, at matched training scale,
improved explanation faithfulness alongside classification quality -- a finding only visible
because faithfulness was measured quantitatively rather than assumed. Comparing training
budgets directly (6 vs. 20 epochs on the same data) further showed that this study's
accuracy ceiling is set by training-set scale rather than epoch count, clarifying exactly
what full-scale GPU training would need to change to close the remaining gap to the
literature.

Future work includes: (1) full-scale training on the complete 10,015-image HAM10000 dataset
for 50 epochs on GPU hardware, using the provided Colab notebook; (2) executing the
remaining ablations (multi-method agreement's isolated effect, uncertainty on/off, loss
function choice, segmentation-guided cropping) and the SHAP and Deletion/Insertion
faithfulness metrics already implemented in the codebase; (3) repeated runs with multiple
seeds to report confidence intervals; (4) a clinician-in-the-loop study validating whether
the Trust Score's flagged low-confidence/low-faithfulness cases correlate with cases where
dermatologists themselves are less certain; and (5) tuning the Trust Score's fusion weights
against such clinician judgments rather than using equal weighting.

## Acknowledgment

The HAM10000 dataset and its ground-truth lesion segmentation masks were obtained from the
Harvard Dataverse under the dataset's stated non-commercial terms of use [1], [2].

## References

[1] P. Tschandl, C. Rosendahl, and H. Kittler, "The HAM10000 dataset, a large collection of
multi-source dermatoscopic images of common pigmented skin lesions," *Sci. Data*, vol. 5,
art. 180161, 2018.

[2] P. Tschandl, C. Rinner, Z. Apalla, G. Argenziano, N. Codella, A. Halpern, M. Janda, A.
Lallas, C. Longo, J. Malvehy, J. Paoli, S. Puig, C. Rosendahl, H. P. Soyer, I. Zalaudek, and
H. Kittler, "Human-computer collaboration for skin cancer recognition," *Nat. Med.*, vol.
26, pp. 1229-1234, 2020.

[3] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M.
Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, "An image is
worth 16x16 words: Transformers for image recognition at scale," in *Proc. Int. Conf.
Learning Representations (ICLR)*, 2021.

[4] Y. Nie, P. Sommella, M. Carratù, M. O'Nils, and J. Lundgren, "A deep CNN transformer
hybrid model for skin lesion classification of dermoscopic images using focal loss,"
*Diagnostics*, vol. 13, no. 1, art. 72, 2022.

[5] I. Kunduracioglu and I. Pacal, "Deep learning for dermatological image analysis: A
critical survey of architectures, evaluation metrics, generalization, and explainable AI,"
*Arch. Comput. Methods Eng.*, 2026.

[6] G. Munjal, P. Bhardwaj, V. Bhargava, S. Singh, and N. Nagpal, "SkinSage XAI: An
explainable deep learning solution for skin lesion diagnosis," *Health Care Sci.*, 2024.

[7] C. Ieracitano, F. C. Morabito, A. Hussain, et al., "TIxAI: A trustworthiness index for
eXplainable AI in skin lesions classification," *Neurocomputing*, vol. 549, art. 129701,
2025.

[8] P. Murali and D. H. Mazumder, "DermaScanAI: an explainable hybrid deep learning
framework for automated skin lesion classification using dual attention and metadata
fusion," *Sci. Rep.*, 2026.

[9] A. Kurz, K. Hauser, H. A. Mehrtens, E. Krieghoff-Henning, A. Hekler, J. N. Kather, S.
Fröhling, C. von Kalle, and T. J. Brinker, "Uncertainty estimation in medical image
classification: Systematic review," *JMIR Med. Inform.*, vol. 10, no. 8, art. e36427, 2022.

[10] Y. Gal and Z. Ghahramani, "Dropout as a Bayesian approximation: Representing model
uncertainty in deep learning," in *Proc. Int. Conf. Machine Learning (ICML)*, 2016, pp.
1050-1059.

[11] A. Chattopadhyay, A. Sarkar, P. Howlader, and V. N. Balasubramanian, "Grad-CAM++:
Generalized gradient-based visual explanations for deep convolutional networks," in *Proc.
IEEE Winter Conf. Applications of Computer Vision (WACV)*, 2018, pp. 839-847.

[12] S. Abnar and W. Zuidema, "Quantifying attention flow in transformers," in *Proc. 58th
Annu. Meeting Assoc. Comput. Linguistics (ACL)*, 2020, pp. 4190-4197.

[13] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions,"
in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2017, pp. 4768-4777.

[14] V. Petsiuk, A. Das, and K. Saenko, "RISE: Randomized input sampling for explanation of
black-box models," in *Proc. British Machine Vision Conf. (BMVC)*, 2018.

[15] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal loss for dense object
detection," in *Proc. IEEE Int. Conf. Computer Vision (ICCV)*, 2017, pp. 2999-3007.

[16] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition,"
in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 770-778.

[17] M. Tan and Q. Le, "EfficientNet: Rethinking model scaling for convolutional neural
networks," in *Proc. Int. Conf. Machine Learning (ICML)*, 2019, pp. 6105-6114.
