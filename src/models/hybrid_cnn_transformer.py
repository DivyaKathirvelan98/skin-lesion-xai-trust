"""Hybrid CNN-Transformer classifier with an MC-Dropout uncertainty head.

Architecture: a convolutional stem (timm backbone, features_only) extracts local texture
feature maps; these are tokenized (flatten + linear projection + learned positional
embedding) and passed through a lightweight Transformer encoder for global lesion-context
modeling; a CLS token is used for the final classification head. Dropout stays active at
inference time (MC-Dropout) to produce predictive uncertainty over repeated stochastic
forward passes.
"""
import timm
import torch
import torch.nn as nn


class HybridCNNTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int = 7,
        cnn_backbone: str = "efficientnet_b0",
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        transformer_dim: int = 256,
        mlp_ratio: float = 4.0,
        dropout: float = 0.2,
        pretrained: bool = True,
    ):
        super().__init__()
        self.cnn = timm.create_model(
            cnn_backbone, pretrained=pretrained, features_only=True, out_indices=(-1,)
        )
        cnn_out_channels = self.cnn.feature_info.channels()[-1]

        self.proj = nn.Conv2d(cnn_out_channels, transformer_dim, kernel_size=1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, transformer_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 49, transformer_dim))
        self.pos_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=transformer_heads,
            dim_feedforward=int(transformer_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        self.norm = nn.LayerNorm(transformer_dim)
        self.head_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(transformer_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.cnn(x)[-1]              # (B, C, H, W)
        feat_map = self.proj(feat_map)           # (B, D, H, W)
        b, d, h, w = feat_map.shape
        tokens = feat_map.flatten(2).transpose(1, 2)  # (B, H*W, D)

        cls_tokens = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)

        pos_embed = self._interpolate_pos_embed(tokens.shape[1])
        tokens = tokens + pos_embed
        tokens = self.pos_drop(tokens)

        encoded = self.transformer(tokens)
        return self.norm(encoded)

    def _interpolate_pos_embed(self, seq_len: int) -> torch.Tensor:
        if seq_len == self.pos_embed.shape[1]:
            return self.pos_embed
        cls_pos, patch_pos = self.pos_embed[:, :1], self.pos_embed[:, 1:]
        n_patches = seq_len - 1
        patch_pos = patch_pos.transpose(1, 2)
        patch_pos = nn.functional.interpolate(patch_pos, size=n_patches, mode="linear", align_corners=False)
        patch_pos = patch_pos.transpose(1, 2)
        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.forward_features(x)
        cls_repr = self.head_dropout(encoded[:, 0])
        return self.classifier(cls_repr)


def enable_mc_dropout(model: nn.Module) -> None:
    """Set all Dropout layers to train mode while keeping the rest of the model in eval mode."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
