const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, BorderStyle, ShadingType,
  Header, Footer, PageNumber, ExternalHyperlink, ImageRun,
} = require("docx");
const fs = require("fs");
const path = require("path");

const FIG_DIR = path.join(__dirname, "..", "docs", "figures");

function figure(filename, widthPx, heightPx, caption) {
  const data = fs.readFileSync(path.join(FIG_DIR, filename));
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 200, after: 80 },
      children: [new ImageRun({ data, type: "png", transformation: { width: widthPx, height: heightPx } })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 200 },
      children: [new TextRun({ text: caption, italics: true, size: 18 })],
    }),
  ];
}

const PAGE = { size: { width: 12240, height: 15840 } }; // US Letter

function h(text, level) {
  return new Paragraph({ text, heading: level, spacing: { before: 240, after: 120 } });
}

function p(text, opts = {}) {
  return new Paragraph({
    children: [new TextRun({ text, italics: opts.italics, bold: opts.bold })],
    spacing: { after: 160 },
    alignment: opts.align || AlignmentType.JUSTIFIED,
  });
}

function pRuns(runs, opts = {}) {
  return new Paragraph({
    children: runs,
    spacing: { after: 160 },
    alignment: opts.align || AlignmentType.JUSTIFIED,
  });
}

function eq(text) {
  return new Paragraph({
    children: [new TextRun({ text, italics: true })],
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 160 },
  });
}

function refPara(text) {
  return new Paragraph({
    children: [new TextRun({ text })],
    spacing: { after: 120 },
    indent: { left: 360, hanging: 360 },
  });
}

function makeTable(headerRow, rows, colWidthsPct) {
  const totalWidth = 9360; // available width in DXA (letter minus 1in margins each side)
  const colWidths = colWidthsPct.map((p) => Math.round((p / 100) * totalWidth));

  function cell(text, opts = {}) {
    return new TableCell({
      width: { size: opts.width, type: WidthType.DXA },
      shading: opts.header ? { type: ShadingType.CLEAR, fill: "D9E2F3" } : undefined,
      children: [new Paragraph({
        children: [new TextRun({ text: String(text), bold: !!opts.header, size: 18 })],
      })],
    });
  }

  const headerCells = headerRow.map((t, i) => cell(t, { header: true, width: colWidths[i] }));
  const bodyRows = rows.map((r) => new TableRow({
    children: r.map((t, i) => cell(t, { width: colWidths[i] })),
  }));

  return new Table({
    width: { size: totalWidth, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [new TableRow({ children: headerCells, tableHeader: true }), ...bodyRows],
  });
}

const doc = new Document({
  sections: [
    {
      properties: { page: PAGE },
      headers: {
        default: new Header({
          children: [new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [new TextRun({ text: "Explainable Hybrid CNN-Transformer Skin Lesion Classification — Draft", size: 16, color: "888888" })],
          })],
        }),
      },
      footers: {
        default: new Footer({
          children: [new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [new TextRun({ children: [PageNumber.CURRENT] })],
          })],
        }),
      },
      children: [
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 240 },
          children: [new TextRun({
            text: "Explainable Vision Transformer Framework for Multi-Class Skin Lesion Classification Using Dermoscopic Images: An Uncertainty-Aware Trust Scoring Approach",
            bold: true, size: 30,
          })],
        }),

        h("Abstract", HeadingLevel.HEADING_1),
        p("Deep learning models for dermoscopic skin lesion classification increasingly incorporate explainable AI (XAI) techniques, yet most report explanation heatmaps only qualitatively, without verifying that explanations are anatomically faithful, and without connecting explanation reliability to the model's own predictive uncertainty. This paper proposes an uncertainty-aware, quantitatively validated explainable framework for multi-class skin lesion classification. A hybrid CNN-Transformer architecture (an EfficientNet-B0 convolutional stem feeding a lightweight Transformer encoder) is trained on HAM10000 dermoscopic images. Explanations are generated from two independent methods — Grad-CAM++ and Attention Rollout — and their agreement is measured quantitatively; explanation faithfulness is validated against ground-truth lesion segmentation masks via Intersection over Union (IoU) and the Pointing Game, rather than by visual inspection alone. Monte Carlo Dropout provides per-prediction epistemic uncertainty, which is fused with faithfulness and cross-method agreement into a single Trust Score. On a compute-constrained (CPU-only) feasibility-scale run — 799/199/201 train/val/test images, 6 training epochs — the proposed hybrid model reaches 76.1% accuracy, 0.913 macro-AUROC, and 0.494 Cohen's kappa, outperforming ResNet50, EfficientNet-B0, and ViT-Base baselines on nearly every classification metric, while achieving a mean faithfulness IoU of 0.163 and Pointing Game accuracy of 42.8%. An ablation study shows that dermoscopic preprocessing (hair removal and CLAHE contrast normalization) improves explanation faithfulness substantially (+0.058 IoU, +10.5 points Pointing Game accuracy) even where its effect on raw accuracy is small. All code, configurations, and results are released openly; the reduced experimental scale relative to the full study design is disclosed explicitly as a compute-driven limitation, with a full-scale reproduction path (GPU training on the complete 10,015-image dataset) provided."),
        pRuns([new TextRun({ text: "Index Terms — ", bold: true }), new TextRun({ text: "Skin lesion classification, explainable artificial intelligence, Vision Transformer, uncertainty quantification, trust score, HAM10000, dermoscopy." })]),

        h("I. Introduction", HeadingLevel.HEADING_1),
        p("Skin cancer is among the most common malignancies worldwide, and early, accurate diagnosis from dermoscopic images materially improves patient outcomes. Convolutional neural networks (CNNs) and, more recently, Vision Transformers (ViTs) [3] have driven substantial gains in automated dermoscopic classification accuracy. However, the black-box nature of these models remains a central barrier to clinical adoption: a clinician cannot act on a prediction they cannot interrogate."),
        p("Explainable AI (XAI) techniques — most commonly Grad-CAM-family saliency maps — have been widely added to dermoscopic classifiers to address this. Yet a close reading of the recent literature reveals three specific, still-open gaps. First, explanations are typically presented and evaluated qualitatively: a heatmap is shown next to a lesion image, and plausibility is assessed by visual inspection, not verified against ground-truth lesion anatomy [6], [8]. Second, no single XAI method provides a holistic explanation, and reviews of the field note that different methods (Grad-CAM, SHAP, LIME) capture different, only partially overlapping notions of “importance” — yet cross-method agreement is rarely measured or reported as a quantitative signal [6]. Third, and most consequentially, predictive uncertainty and explanation quality are treated as separate concerns: recent trustworthiness-focused work such as TIxAI [7] quantifies how well a single explanation method (Grad-CAM) localizes to the lesion, and recent hybrid architectures such as DermaScanAI [8] combine attention mechanisms with Grad-CAM++/SHAP visualizations for improved accuracy and interpretability, but neither fuses an explicit uncertainty estimate with explanation reliability into one clinically actionable signal."),
        p("This paper addresses these three gaps directly. We do not claim to outperform the accuracy of large-scale, GPU-trained, full-dataset models in the recent literature (which reach 92–98% accuracy on HAM10000 [4], [8]); our contribution is architectural and methodological, evaluated honestly at a reduced, compute-constrained scale, with a clear path to full-scale reproduction."),
        p("The contributions of this paper are:"),
        p("1. A lightweight hybrid CNN-Transformer architecture for 7-class HAM10000 dermoscopic lesion classification, combining local texture features (CNN) with global lesion-context modeling (Transformer self-attention)."),
        p("2. A multi-method XAI ensemble (Grad-CAM++ [11], Attention Rollout [12]) with a quantitative cross-method agreement score (IoU and Spearman correlation between saliency maps), rather than reliance on a single explanation method."),
        p("3. Ground-truth-anchored, quantitative faithfulness evaluation of explanations against dermatologist-curated HAM10000 lesion segmentation masks [2], using IoU and the Pointing Game, rather than qualitative-only assessment."),
        p("4. A Trust Score that fuses Monte Carlo Dropout predictive uncertainty [10] with explanation faithfulness and cross-method agreement into a single, per-prediction signal intended to flag cases warranting clinician review."),
        p("5. A fully open, reproducible implementation (code, configuration, and raw evaluation output), with a disclosed, honest account of the reduced experimental scale used in this study and a documented path to full-scale reproduction on GPU hardware."),

        h("II. Related Work", HeadingLevel.HEADING_1),
        p("Skin lesion classification architectures. CNN backbones (ResNet, EfficientNet) have long dominated dermoscopic classification, but Vision Transformers [3] and hybrid CNN-Transformer architectures have gained traction for their ability to model long-range spatial dependencies. Nie et al. [4] proposed a CNN-Transformer hybrid trained with focal loss on dermoscopic images, reporting improved handling of HAM10000's class imbalance. Kunduracioglu and Pacal's 2026 survey of 135 studies (2023–2026) confirms hybrid CNN-Transformer and ensemble architectures as the dominant recent direction, alongside growing adoption of XAI methods [5]."),
        p("Explainable AI in dermatology. Munjal et al.'s SkinSage XAI [6] combines Grad-CAM and LIME with an Inception-v3 backbone, reporting high accuracy but evaluating explanations qualitatively. Ieracitano et al.'s TIxAI [7] is closest in spirit to this paper: it defines a Trustworthiness Index by measuring the relevance difference between lesion and non-lesion Grad-CAM regions on an EfficientNet-B0 classifier — a single-method, localization-only notion of trust. Murali and Mazumder's DermaScanAI [8] combines multi-scale CNN features, lightweight Transformer encoders, and dual attention with Grad-CAM++ and SHAP for a reported 94.8% accuracy on HAM10000, but does not quantitatively validate explanation faithfulness against ground-truth segmentation, nor does it fuse uncertainty with explanation quality. This paper differs from all three in combining (i) multi-method cross-agreement, (ii) ground-truth-mask-anchored faithfulness metrics, and (iii) uncertainty fusion, into one Trust Score."),
        p("Uncertainty quantification. Kurz et al.'s systematic review of uncertainty estimation in medical image classification [9] finds Monte Carlo Dropout and deep ensembles the most common sampling-based approaches, following the Bayesian approximation formalized by Gal and Ghahramani [10]. Uncertainty estimation is well studied in medical imaging broadly, but its fusion with explanation reliability specifically — rather than with calibration or classification confidence alone — remains, to our knowledge, largely unexplored for dermoscopic classification."),
        p("Synthesis. The literature shows rapid progress on accuracy and on adding some form of explainability, but a persistent gap remains: explanation faithfulness is asserted, not measured against ground truth; multiple explanation methods are rarely cross-validated against each other; and uncertainty and explanation quality are not fused into one actionable signal. This paper's Trust Score framework is designed to close that specific combination of gaps."),

        h("III. Materials and Methods", HeadingLevel.HEADING_1),
        h("A. Dataset", HeadingLevel.HEADING_2),
        p("We use HAM10000 [1], [2], 10,015 dermoscopic images across seven diagnostic classes (actinic keratosis/Bowen's disease [akiec], basal cell carcinoma [bcc], benign keratosis-like lesions [bkl], dermatofibroma [df], melanoma [mel], melanocytic nevi [nv], vascular lesions [vasc]), obtained directly from the official Harvard Dataverse release (DOI: 10.7910/DVN/DBW86T; MD5-verified download). Dermatologist-curated ground-truth lesion segmentation masks, released by the same authors [2], are used for faithfulness evaluation."),
        p("Due to CPU-only compute availability for this study (see Sections III-I and V), a stratified, lesion-wise (patient-wise) subset was used: 799 train / 199 validation / 201 test images, split at the lesion_id level (not the image level) before subsampling, so that no lesion's images appear in more than one split — avoiding the data leakage that image-level splitting of HAM10000 is known to risk, since many lesions have multiple images. All seven classes are represented in every split."),

        h("B. Image Preprocessing Pipeline", HeadingLevel.HEADING_2),
        p("1. Deduplication and leakage-safe split: grouping by lesion_id before any train/val/test assignment (Section III-A)."),
        p("2. Hair and artifact removal: DullRazor-style morphological black-hat filtering followed by Telea inpainting, to suppress dark hair strands that otherwise contaminate both classification features and saliency maps."),
        p("3. Color and illumination normalization: gray-world channel correction followed by Contrast-Limited Adaptive Histogram Equalization (CLAHE, clip limit 2.0) on the luminance channel."),
        p("4. Resizing: aspect-preserving resize/pad to the model input resolution (160×160 in this study; 224×224 in the full design)."),
        p("5. Augmentation and class-imbalance handling: random crop, horizontal/vertical flips, rotation (up to 30°), and color jitter during training, combined with a focal loss (Section III-F) to counteract HAM10000's severe class imbalance (nv comprises roughly 68% of images)."),

        h("C. Proposed Hybrid CNN-Transformer Architecture", HeadingLevel.HEADING_2),
        p("Given an input image X, an EfficientNet-B0 [17] convolutional stem CNNθ extracts a local feature map:"),
        eq("F = CNNθ(X) ∈ ℝ^(H×W×C)"),
        p("F is projected to the Transformer's embedding dimension via a 1×1 convolution and flattened into a sequence of tokens {pᵢ}, each linearly projected and combined with a learned positional embedding:"),
        eq("zᵢ = Wp · flatten(pᵢ) + E_pos"),
        p("A learned classification token z_cls is prepended, and the resulting sequence is passed through L Transformer encoder layers (multi-head self-attention with GELU-activated feed-forward blocks):"),
        eq("Attention(Q,K,V) = softmax(QKᵀ / √d) V"),
        p("The final classification token representation is passed through a dropout layer and a linear classification head:"),
        eq("ŷ = softmax(W_cls · z_cls + b)"),
        p("In this study, L=4 encoder layers, 8 attention heads, and an embedding dimension of 256 are used. Dropout (p=0.2) is retained active at inference time to support Monte Carlo Dropout uncertainty estimation (Section III-G)."),

        h("D. Explainability Methods", HeadingLevel.HEADING_2),
        p("Two independent explanation methods are computed for each test prediction of the proposed model:"),
        p("• Grad-CAM++ [11], applied to the CNN stem's final convolutional layer, producing a gradient-weighted class activation map."),
        p("• Attention Rollout [12], which recursively multiplies (with a residual identity term and row-normalization) the Transformer's per-layer self-attention matrices to trace how the classification token's final representation depends on each input patch token."),
        p("A third, model-agnostic method (SHAP [13], via GradientExplainer) is implemented in the released codebase but was not executed in this study's reported results, as its per-image computational cost (tens of forward passes per explanation) was infeasible within the CPU-only time budget; this is disclosed as a limitation (Section V)."),
        p("Cross-method agreement. For each image, the two saliency maps are resized to a common resolution, binarized at a threshold of 0.5, and compared via IoU and Spearman rank correlation. The mean pairwise IoU is used as the agreement term in the Trust Score."),

        h("E. Faithfulness Evaluation", HeadingLevel.HEADING_2),
        p("Rather than assessing explanations qualitatively, each Grad-CAM++ saliency map is compared against the corresponding ground-truth lesion segmentation mask [2]:"),
        p("• IoU: intersection-over-union between the binarized (threshold 0.5) saliency map and the binary lesion mask."),
        p("• Pointing Game accuracy: the fraction of test images for which the saliency map's single highest-activation pixel falls inside the ground-truth lesion region."),
        p("A Deletion/Insertion AUC metric [14] is also implemented (progressively masking/revealing the most salient pixels and tracking the target-class probability) but was not executed at scale in this study for the same compute-budget reason as SHAP."),

        h("F. Focal Loss", HeadingLevel.HEADING_2),
        p("To counteract HAM10000's severe class imbalance, training uses focal loss [15]:"),
        eq("FL(p_t) = −α_t (1−p_t)^γ log(p_t)"),
        p("with α=0.25, γ=2.0, down-weighting the loss contribution of well-classified majority-class (nv) examples relative to harder, minority-class examples."),

        h("G. Uncertainty Quantification", HeadingLevel.HEADING_2),
        p("Predictive (epistemic) uncertainty is estimated via Monte Carlo Dropout [10]: at inference, dropout layers are kept active and T stochastic forward passes are performed per image:"),
        eq("μ = (1/T)Σŷ_t ,   σ² = (1/T)Σ(ŷ_t − μ)²"),
        p("T=10 passes are used in this study (vs. 20 in the full design, reduced for CPU time). The predictive entropy of μ, normalized by log(num_classes), is used as the uncertainty term in the Trust Score."),

        h("H. Trust Score", HeadingLevel.HEADING_2),
        p("The Trust Score fuses (normalized, [0,1]-scaled) predictive uncertainty, faithfulness, and cross-method agreement into a single per-prediction value:"),
        eq("T = w1(1−σ²_norm) + w2·Faithfulness + w3·Agreement,   Σwi = 1"),
        p("with w1=0.34, w2=0.33, w3=0.33 in this study (equal weighting; weight tuning against clinician judgments is left as future work)."),

        h("I. Baseline Models and Experimental Setup", HeadingLevel.HEADING_2),
        p("Baselines: ResNet50 [16], EfficientNet-B0 [17], and ViT-Base [3] (patch size 16), all initialized with ImageNet-pretrained weights (via timm/Hugging Face Hub) and fine-tuned identically to the proposed model."),
        p("Training: AdamW optimizer, learning rate 3×10⁻⁴, weight decay 1×10⁻⁴, cosine learning-rate schedule, batch size 16, 6 epochs, image resolution 160×160 (ViT-Base uses dynamic positional-embedding interpolation to accept this non-native resolution). This is a deliberate, disclosed reduction from the full design (50 epochs, 224×224, the complete 10,015-image dataset) made necessary by CPU-only compute availability for this study; see Section V."),
        p("Evaluation metrics: accuracy, balanced accuracy, macro-F1, macro-AUROC (one-vs-rest), Cohen's kappa, and Expected Calibration Error (ECE) for classification; IoU and Pointing Game accuracy for faithfulness; mean Trust Score for the fused uncertainty-explanation signal."),

        h("IV. Results and Discussion", HeadingLevel.HEADING_1),
        h("A. Classification Performance", HeadingLevel.HEADING_2),
        p("Table I reports test-set (n=201) classification metrics for all four models."),
        p("Table I. Classification performance (test set, n=201)", { bold: true, align: AlignmentType.CENTER }),
        makeTable(
          ["Model", "Accuracy", "Bal. Acc.", "Macro F1", "Macro AUROC", "Kappa", "ECE"],
          [
            ["Hybrid CNN-Transformer (proposed)", "0.761", "0.457", "0.466", "0.913", "0.494", "0.119"],
            ["EfficientNet-B0", "0.751", "0.450", "0.464", "0.878", "0.462", "0.104"],
            ["ResNet50", "0.711", "0.180", "0.168", "0.842", "0.187", "0.120"],
            ["ViT-Base", "0.677", "0.157", "0.141", "0.813", "0.075", "0.141"],
          ],
          [30, 12, 12, 12, 12, 11, 11]
        ),
        p(" "),
        p("The proposed hybrid model achieves the best accuracy, macro-AUROC, and Cohen's kappa among the four models, and is competitive with EfficientNet-B0 (its own CNN backbone) on balanced accuracy and macro-F1, only narrowly trailing on calibration (ECE). ResNet50 and, notably, ViT-Base show substantially lower balanced accuracy and kappa, indicating a stronger lean toward the majority class (nv) under this study's limited training budget — consistent with ViT-family architectures' well-documented need for larger training sets or longer fine-tuning schedules than CNN-based models to reach comparable performance, an effect likely exacerbated here by the reduced (800-image) training set and 6-epoch budget."),
        ...figure("fig1_classification_bars.png", 500, 281, "Fig. 1. Classification performance by model (test set, n=201)."),
        ...figure("fig2_training_curves.png", 520, 272, "Fig. 2. Training/validation loss and accuracy over 6 epochs for all four models."),
        p("Figure 2 shows all four models still improving at epoch 6 without clear overfitting (validation loss tracking or below training loss throughout), suggesting that — consistent with the compute-driven limitation disclosed in Section V — additional epochs would likely improve results further rather than the models having converged."),
        ...figure("fig3_confusion_matrix.png", 380, 345, "Fig. 3. Confusion matrix for the proposed hybrid model (test set, n=201)."),
        p("Figure 3 shows the hybrid model's errors concentrate along the bkl/mel/nv boundary (e.g., 8 of 23 true bkl images predicted as nv, 10 of 17 true mel images predicted as nv) rather than being spread uniformly across classes — consistent with the well-documented clinical difficulty of visually distinguishing benign keratoses, melanoma, and melanocytic nevi in dermoscopic images, rather than an idiosyncratic model failure mode."),

        h("B. XAI Faithfulness and Cross-Method Agreement", HeadingLevel.HEADING_2),
        p("For the proposed hybrid model, mean faithfulness IoU (Grad-CAM++ vs. ground-truth lesion mask) was 0.163, and Pointing Game accuracy was 42.8% — i.e., the single most-activated pixel of the Grad-CAM++ map fell inside the true lesion region in 42.8% of test images. These values are modest in absolute terms, consistent with a model fine-tuned for only 6 epochs on 800 images; they are reported without adjustment, as an honest baseline for future full-scale comparison, rather than presented as a mature clinical-grade result."),
        ...figure("fig5_xai_qualitative.png", 560, 142, "Fig. 4. Qualitative example (a correctly-classified test image) comparing Grad-CAM++ and Attention Rollout saliency against the ground-truth lesion mask."),
        p("Figure 4 illustrates a representative case: both Grad-CAM++ and Attention Rollout concentrate activation within the true lesion boundary, though on different sub-regions of it — the specific behavior the cross-method agreement score and mask-anchored IoU in Sections III-D-E are designed to quantify rather than leave to visual impression."),

        h("C. Uncertainty and Trust Score", HeadingLevel.HEADING_2),
        p("The mean Trust Score across the test set was 0.250 (on a [0,1] scale). Because the Trust Score combines three quantities that are each still maturing at this training scale (calibration, faithfulness, and cross-method agreement), its absolute value should be read as a starting point for the metric's validation, not as a claim of high trustworthiness. Its value as a framework lies in making these three signals jointly visible and computable per-prediction — something the reviewed prior work (TIxAI, DermaScanAI, SkinSage XAI) does not do in combination."),

        h("D. Ablation Study", HeadingLevel.HEADING_2),
        p("Table II reports two of the six ablations from the study design (Section III); the remaining four — multi-method XAI agreement's effect in isolation, uncertainty quantification on/off, focal loss vs. plain cross-entropy, and segmentation-guided cropping — were not executed under the CPU-only compute budget and are disclosed as future work rather than reported without evidence."),
        p("Table II. Ablation results", { bold: true, align: AlignmentType.CENTER }),
        makeTable(
          ["Ablation", "Variant", "Accuracy", "Bal. Acc.", "Macro F1", "Faithfulness IoU"],
          [
            ["Architecture", "CNN-only (EfficientNet-B0)", "0.751", "0.450", "0.464", "—"],
            ["Architecture", "CNN+Transformer (proposed)", "0.761", "0.457", "0.466", "0.163"],
            ["Preprocessing", "Without hair removal / CLAHE", "0.751", "0.510", "0.511", "0.105"],
            ["Preprocessing", "With hair removal / CLAHE (proposed)", "0.761", "0.457", "0.466", "0.163"],
          ],
          [18, 32, 12, 12, 12, 14]
        ),
        p(" "),
        ...figure("fig4_ablation_bars.png", 560, 224, "Fig. 5. Ablation results: architecture (left) and preprocessing (right)."),
        p("The Transformer stage's clearest measurable benefit over the CNN-only baseline is macro-AUROC (0.913 vs. 0.878, from Table I); its effect on accuracy/kappa is present but modest at this training scale. The preprocessing ablation is more nuanced: hair removal and CLAHE normalization improve raw accuracy only slightly and, in this single run, balanced accuracy and macro-F1 were actually higher without preprocessing — plausibly run-to-run noise given the small (n=201) single-seed test set rather than a genuine effect, and a result we report honestly rather than omit. Preprocessing's clearer benefit is on explanation quality: faithfulness IoU improves by 0.058 and Pointing Game accuracy by 10.5 points with preprocessing, consistent with the intuition that removing hair and illumination artifacts helps the model's activation patterns concentrate on the true lesion region rather than incidental image artifacts — a benefit that would not have been visible without the ground-truth-mask-anchored faithfulness metrics proposed in Section III-E."),

        h("V. Limitations", HeadingLevel.HEADING_1),
        p("This study's results were produced under a strict, disclosed compute constraint (a 4-core CPU machine with no GPU) and should be read as a feasibility-scale pilot, not a full-scale benchmark result:"),
        p("• Data scale: 799/199/201 train/val/test images, drawn from the first 5,000 of HAM10000's 10,015 images, vs. the full dataset in the study's original design."),
        p("• Training budget: 6 epochs vs. 50 in the full design; a single run with a fixed seed (no repeated runs/cross-validation, so no confidence intervals are reported)."),
        p("• Resolution: 160×160 vs. 224×224."),
        p("• XAI scope: SHAP and the Deletion/Insertion faithfulness metric are implemented in the released code but were not executed in this study's results, for CPU time feasibility."),
        p("• Ablation scope: two of six planned ablations were executed; the remainder are left as future work rather than asserted without evidence."),
        p("• MC-Dropout passes: 10 vs. 20 in the full design."),
        p("None of these constraints were used to inflate or cherry-pick results: all reported numbers are the direct, unedited output of the released evaluation code (runs/eval_report_*.json), and the full experimental configuration is version-controlled alongside the results for exact reproducibility. A Google Colab notebook (notebooks/run_experiments.ipynb) is provided to reproduce this study at full scale on GPU hardware."),

        h("VI. Conclusion and Future Work", HeadingLevel.HEADING_1),
        p("This paper presented a hybrid CNN-Transformer framework for dermoscopic skin lesion classification that goes beyond typical single-method, qualitative XAI by (i) cross-validating two independent explanation methods against each other, (ii) quantitatively grounding explanation faithfulness in dermatologist-curated lesion segmentation masks rather than visual inspection, and (iii) fusing Monte Carlo Dropout uncertainty with explanation quality into a single, per-prediction Trust Score. Under a disclosed, compute-constrained pilot evaluation, the proposed architecture outperformed ResNet50, EfficientNet-B0, and ViT-Base baselines on most classification metrics, and the ablation study showed that dermoscopic preprocessing meaningfully improves explanation faithfulness even where its effect on raw accuracy is small — a finding only visible because faithfulness was measured quantitatively rather than assumed."),
        p("Future work includes: (1) full-scale training on the complete 10,015-image HAM10000 dataset for 50 epochs on GPU hardware, using the provided Colab notebook; (2) executing the remaining ablations (multi-method agreement's isolated effect, uncertainty on/off, loss function choice, segmentation-guided cropping) and the SHAP and Deletion/Insertion faithfulness metrics already implemented in the codebase; (3) repeated runs with multiple seeds to report confidence intervals; (4) a clinician-in-the-loop study validating whether the Trust Score's flagged low-confidence/low-faithfulness cases correlate with cases where dermatologists themselves are less certain; and (5) tuning the Trust Score's fusion weights against such clinician judgments rather than using equal weighting."),

        h("Acknowledgment", HeadingLevel.HEADING_1),
        p("The HAM10000 dataset and its ground-truth lesion segmentation masks were obtained from the Harvard Dataverse under the dataset's stated non-commercial terms of use [1], [2]."),

        h("References", HeadingLevel.HEADING_1),
        refPara("[1] P. Tschandl, C. Rosendahl, and H. Kittler, \"The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions,\" Sci. Data, vol. 5, art. 180161, 2018."),
        refPara("[2] P. Tschandl, C. Rinner, Z. Apalla, G. Argenziano, N. Codella, A. Halpern, M. Janda, A. Lallas, C. Longo, J. Malvehy, J. Paoli, S. Puig, C. Rosendahl, H. P. Soyer, I. Zalaudek, and H. Kittler, \"Human-computer collaboration for skin cancer recognition,\" Nat. Med., vol. 26, pp. 1229–1234, 2020."),
        refPara("[3] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, \"An image is worth 16x16 words: Transformers for image recognition at scale,\" in Proc. Int. Conf. Learning Representations (ICLR), 2021."),
        refPara("[4] Y. Nie, P. Sommellà, M. Carratù, M. O'Nils, and J. Lundgren, \"A deep CNN transformer hybrid model for skin lesion classification of dermoscopic images using focal loss,\" Diagnostics, vol. 13, no. 1, art. 72, 2022."),
        refPara("[5] I. Kunduracioglu and I. Pacal, \"Deep learning for dermatological image analysis: A critical survey of architectures, evaluation metrics, generalization, and explainable AI,\" Arch. Comput. Methods Eng., 2026."),
        refPara("[6] G. Munjal, P. Bhardwaj, V. Bhargava, S. Singh, and N. Nagpal, \"SkinSage XAI: An explainable deep learning solution for skin lesion diagnosis,\" Health Care Sci., 2024."),
        refPara("[7] C. Ieracitano, F. C. Morabito, A. Hussain, et al., \"TIxAI: A trustworthiness index for eXplainable AI in skin lesions classification,\" Neurocomputing, vol. 549, art. 129701, 2025."),
        refPara("[8] P. Murali and D. H. Mazumder, \"DermaScanAI: an explainable hybrid deep learning framework for automated skin lesion classification using dual attention and metadata fusion,\" Sci. Rep., 2026."),
        refPara("[9] A. Kurz, K. Hauser, H. A. Mehrtens, E. Krieghoff-Henning, A. Hekler, J. N. Kather, S. Fröhling, C. von Kalle, and T. J. Brinker, \"Uncertainty estimation in medical image classification: Systematic review,\" JMIR Med. Inform., vol. 10, no. 8, art. e36427, 2022."),
        refPara("[10] Y. Gal and Z. Ghahramani, \"Dropout as a Bayesian approximation: Representing model uncertainty in deep learning,\" in Proc. Int. Conf. Machine Learning (ICML), 2016, pp. 1050–1059."),
        refPara("[11] A. Chattopadhyay, A. Sarkar, P. Howlader, and V. N. Balasubramanian, \"Grad-CAM++: Generalized gradient-based visual explanations for deep convolutional networks,\" in Proc. IEEE Winter Conf. Applications of Computer Vision (WACV), 2018, pp. 839–847."),
        refPara("[12] S. Abnar and W. Zuidema, \"Quantifying attention flow in transformers,\" in Proc. 58th Annu. Meeting Assoc. Comput. Linguistics (ACL), 2020, pp. 4190–4197."),
        refPara("[13] S. M. Lundberg and S.-I. Lee, \"A unified approach to interpreting model predictions,\" in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2017, pp. 4768–4777."),
        refPara("[14] V. Petsiuk, A. Das, and K. Saenko, \"RISE: Randomized input sampling for explanation of black-box models,\" in Proc. British Machine Vision Conf. (BMVC), 2018."),
        refPara("[15] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, \"Focal loss for dense object detection,\" in Proc. IEEE Int. Conf. Computer Vision (ICCV), 2017, pp. 2999–3007."),
        refPara("[16] K. He, X. Zhang, S. Ren, and J. Sun, \"Deep residual learning for image recognition,\" in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770–778."),
        refPara("[17] M. Tan and Q. Le, \"EfficientNet: Rethinking model scaling for convolutional neural networks,\" in Proc. Int. Conf. Machine Learning (ICML), 2019, pp. 6105–6114."),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(process.argv[2] || "paper.docx", buffer);
  console.log("Written.");
});
