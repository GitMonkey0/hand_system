import torch.nn as nn

from handpose.models.encoder import SiMHandEncoder
from handpose.losses.ce_loss import ce_loss

class HLClassifier(nn.Module):
    """
    image -> encoder(2048) -> logits [B,40,26]
    """
    def __init__(self, encoder_ckpt=None):
        super().__init__()
        self.encoder = SiMHandEncoder(ckpt_path=encoder_ckpt)
        self.head = nn.Linear(2048, 40 * 26)

    def forward(self, pixel_values, labels=None):
        feat = self.encoder(pixel_values)  # [B,2048]
        logits = self.head(feat).view(-1, 40, 26)
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = ce_loss(logits, labels)
        return out