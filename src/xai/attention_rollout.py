"""Attention Rollout for the Transformer encoder (Abnar & Zuidema, 2020).

Multiplies attention matrices across layers (with residual/identity added, then
row-normalized) to trace how the CLS token attends to input patch tokens end-to-end.
"""
import torch
import torch.nn as nn


class AttentionRolloutHook:
    """Captures self-attention weights from every nn.MultiheadAttention layer in a model.

    nn.TransformerEncoderLayer internally calls self_attn with need_weights=False (to use
    the fused/flash-attention fast path), so no forward hook alone can observe attention
    weights. We wrap each self_attn.forward to force need_weights=True,
    average_attn_weights=False regardless of how it's called, then hook the (now
    non-None) output.
    """

    def __init__(self, transformer_encoder: nn.TransformerEncoder):
        self.attentions = []
        self.handles = []
        self._original_forwards = []
        for layer in transformer_encoder.layers:
            self._original_forwards.append((layer.self_attn, layer.self_attn.forward))
            layer.self_attn.forward = self._make_forced_forward(layer.self_attn.forward)
            handle = layer.self_attn.register_forward_hook(self._hook)
            self.handles.append(handle)

    @staticmethod
    def _make_forced_forward(original_forward):
        def forced_forward(query, key, value, *args, **kwargs):
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = False
            return original_forward(query, key, value, *args, **kwargs)
        return forced_forward

    def _hook(self, module, inputs, output):
        _, attn_weights = output if isinstance(output, tuple) else (output, None)
        if attn_weights is not None:
            self.attentions.append(attn_weights.detach())

    def clear(self):
        self.attentions = []

    def remove(self):
        for handle in self.handles:
            handle.remove()
        for module, original_forward in self._original_forwards:
            module.forward = original_forward


def compute_rollout(attentions: list, discard_ratio: float = 0.0, head_fusion: str = "mean") -> torch.Tensor:
    """Combine per-layer attention matrices into a single rollout matrix.

    Args:
        attentions: list of tensors, each (B, heads, N, N) or (B, N, N) if already head-fused.
        discard_ratio: fraction of lowest attention values to zero out per layer (noise reduction).
        head_fusion: "mean" or "max" across attention heads.

    Returns:
        Rollout attention matrix of shape (B, N, N).
    """
    result = None
    for attn in attentions:
        if attn.dim() == 4:
            attn = attn.mean(dim=1) if head_fusion == "mean" else attn.max(dim=1).values

        if discard_ratio > 0:
            flat = attn.flatten(1)
            k = int(flat.shape[1] * discard_ratio)
            if k > 0:
                threshold = flat.kthvalue(k, dim=1).values.unsqueeze(1)
                attn = torch.where(attn < threshold.unsqueeze(-1), torch.zeros_like(attn), attn)

        identity = torch.eye(attn.shape[-1], device=attn.device).unsqueeze(0)
        attn = (attn + identity) / 2
        attn = attn / attn.sum(dim=-1, keepdim=True)

        result = attn if result is None else torch.bmm(attn, result)
    return result


def rollout_saliency_map(rollout: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Extract the CLS-to-patch attention row and reshape it to a (grid_size, grid_size) map."""
    cls_attention = rollout[:, 0, 1:]  # drop attention to itself
    return cls_attention.reshape(-1, grid_size, grid_size)
