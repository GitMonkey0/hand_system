# src/slt/models/visual_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class VisualEncoder(nn.Module):
    """
    Input:  video_frames [bs, T, C, H, W]
            video_attention_mask [bs, T] (1=valid, 0=pad) 可选
    Output: visual_embeds [bs, Lh, H]
            visual_attention_mask [bs, Lh]
    """
    def __init__(self, hidden_size: int, out_tokens: int = 64, in_channels: int = 3, feat_dim: int = 512):
        super().__init__()
        self.out_tokens = out_tokens

        # 一个很简陋的 per-frame CNN backbone（占位）
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, feat_dim, 1),
            nn.ReLU(inplace=True),
        )
        self.pool2d = nn.AdaptiveAvgPool2d((1, 1))  # per-frame -> token
        self.proj = nn.Linear(feat_dim, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)

        # 时间维从 T -> Lh（固定 token 数）；也可以改成 Transformer/TCN
        self.temporal_pool = nn.AdaptiveAvgPool1d(out_tokens)

    def forward(self, video_frames: torch.Tensor, video_attention_mask: torch.Tensor | None = None):
        bs, T, C, H, W = video_frames.shape
        x = video_frames.view(bs * T, C, H, W)          # [bs*T,C,H,W]
        f = self.backbone(x)                            # [bs*T,feat_dim,h,w]
        f = self.pool2d(f).flatten(1)                   # [bs*T,feat_dim]
        f = f.view(bs, T, -1)                           # [bs,T,feat_dim]

        # mask 可选：如果 padding 帧很多，建议在这里把无效帧置0再池化
        if video_attention_mask is not None:
            f = f * video_attention_mask.unsqueeze(-1).to(f.dtype)

        # [bs,T,feat] -> [bs,feat,T] -> pool -> [bs,feat,Lh] -> [bs,Lh,feat]
        f = f.transpose(1, 2)
        f = self.temporal_pool(f).transpose(1, 2)       # [bs,Lh,feat_dim]

        embeds = self.ln(self.proj(f))                  # [bs,Lh,H]
        attn = torch.ones(bs, embeds.size(1), device=embeds.device, dtype=torch.long)
        return embeds, attn