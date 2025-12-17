import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from transformers import Qwen3ForCausalLM


class HandFeatureEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [bs, Lh, D]
        return self.ln(self.proj(x))  # [bs, Lh, H]


class Qwen3ForSignTranslation(nn.Module):
    def __init__(self, llm: Qwen3ForCausalLM, hand_feat_dim: int):
        super().__init__()
        self.llm = llm
        self.hand_encoder = HandFeatureEncoder(hand_feat_dim, llm.config.hidden_size)

        self._cached_hand_embeds: Optional[torch.Tensor] = None
        self._cached_hand_len: int = 0

    def reset_hand_cache(self):
        self._cached_hand_embeds = None
        self._cached_hand_len = 0

    def _build_position_ids(self, attention_mask: torch.Tensor) -> torch.LongTensor:
        # attention_mask: [bs, L]
        pos = attention_mask.long().cumsum(-1) - 1
        pos.masked_fill_(attention_mask == 0, 0)
        return pos

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,           # [bs, Lt]
        attention_mask: Optional[torch.Tensor] = None,          # [bs, Lt]
        labels: Optional[torch.LongTensor] = None,              # [bs, Lt]
        hand_features: Optional[torch.Tensor] = None,           # [bs, Lh, D]
        hand_attention_mask: Optional[torch.Tensor] = None,     # [bs, Lh]
        past_key_values=None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("input_ids required")

        device = input_ids.device
        bs, Lt = input_ids.shape

        if attention_mask is None:
            attention_mask = torch.ones(bs, Lt, dtype=torch.long, device=device)

        if past_key_values is None and hand_features is not None:
            hand_embeds = self.hand_encoder(hand_features.to(device=device, dtype=self.llm.model.embed_tokens.weight.dtype))
            Lh = hand_embeds.shape[1]

            if hand_attention_mask is None:
                hand_attention_mask = torch.ones(bs, Lh, dtype=attention_mask.dtype, device=device)

            text_embeds = self.llm.model.embed_tokens(input_ids)  # [bs, Lt, H]
            inputs_embeds = torch.cat([hand_embeds, text_embeds], dim=1)  # [bs, Lh+Lt, H]
            attention_mask = torch.cat([hand_attention_mask, attention_mask], dim=1)  # [bs, Lh+Lt]
            position_ids = self._build_position_ids(attention_mask)

            if labels is not None:
                pad = torch.full((bs, Lh), -100, dtype=labels.dtype, device=device)
                labels = torch.cat([pad, labels], dim=1)

            return self.llm(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=labels,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        return self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        hand_features=None,
        hand_attention_mask=None,
        **kwargs,
    ) -> Dict[str, Any]:
        if past_key_values is None and hand_features is not None:
            device = input_ids.device
            bs, Lt = input_ids.shape

            if attention_mask is None:
                attention_mask = torch.ones(bs, Lt, dtype=torch.long, device=device)

            hand_embeds = self.hand_encoder(
                hand_features.to(device=device, dtype=self.llm.model.embed_tokens.weight.dtype)
            )
            Lh = hand_embeds.shape[1]
            if hand_attention_mask is None:
                hand_attention_mask = torch.ones(bs, Lh, dtype=attention_mask.dtype, device=device)

            text_embeds = self.llm.model.embed_tokens(input_ids)
            inputs_embeds = torch.cat([hand_embeds, text_embeds], dim=1)
            attention_mask = torch.cat([hand_attention_mask, attention_mask], dim=1)
            position_ids = self._build_position_ids(attention_mask)

            model_inputs = {
                "input_ids": None,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_values": None,
                "use_cache": use_cache,
                "cache_position": cache_position,
            }
            model_inputs.update(kwargs)
            return model_inputs

        model_inputs = self.llm.prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            **kwargs,
        )
        model_inputs.pop("hand_features", None)
        model_inputs.pop("hand_attention_mask", None)
        return model_inputs