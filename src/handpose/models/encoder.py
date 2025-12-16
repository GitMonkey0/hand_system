import torch
import torch.nn as nn
from torchvision.models import resnet50


class SiMHandEncoder(nn.Module):
    """ResNet50-based encoder, loading HL-pretrained weights and outputting 2048-dim features."""

    def __init__(self, ckpt_path: str | None = None):
        super().__init__()
        self.backbone = resnet50(weights=None)
        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path, map_location="cpu")
            self.backbone.load_state_dict(state_dict, strict=True)
        self.backbone.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,H,W]
        return self.backbone(x)  # [B,2048]
