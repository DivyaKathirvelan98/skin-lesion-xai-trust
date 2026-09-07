"""MC-Dropout predictive uncertainty: repeated stochastic forward passes at inference."""
import torch
import torch.nn.functional as F

from src.models.hybrid_cnn_transformer import enable_mc_dropout


@torch.no_grad()
def mc_dropout_predict(model: torch.nn.Module, image: torch.Tensor, num_passes: int = 20):
    """Run `num_passes` stochastic forward passes with dropout active.

    Returns:
        mean_probs: (num_classes,) averaged softmax probabilities.
        variance: (num_classes,) per-class predictive variance across passes.
        predictive_entropy: scalar entropy of mean_probs (higher = more uncertain).
    """
    model.eval()
    enable_mc_dropout(model)

    probs_per_pass = []
    for _ in range(num_passes):
        logits = model(image)
        probs_per_pass.append(F.softmax(logits, dim=-1))

    probs_stack = torch.stack(probs_per_pass, dim=0)  # (T, B, num_classes)
    mean_probs = probs_stack.mean(dim=0)
    variance = probs_stack.var(dim=0)
    predictive_entropy = -(mean_probs * torch.log(mean_probs.clamp_min(1e-12))).sum(dim=-1)

    return mean_probs, variance, predictive_entropy
