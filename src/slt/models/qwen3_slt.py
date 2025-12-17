# src/slt/models/qwen3_slt.py
from transformers import Qwen3ForCausalLM
from typing import Optional, Dict, Any
import torch
import torch.nn as nn

class Qwen3ForSignTranslation(nn.Module):
    def __init__(self, llm: Qwen3ForCausalLM, out_visual_tokens: int = 64):
        super().__init__()
        self.llm = llm
        self.visual_encoder = VisualEncoder(hidden_size=llm.config.hidden_size, out_tokens=out_visual_tokens)

    def _build_position_ids(self, attention_mask: torch.Tensor) -> torch.LongTensor:
        pos = attention_mask.long().cumsum(-1) - 1
        pos.masked_fill_(attention_mask == 0, 0)
        return pos

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        video_frames: Optional[torch.Tensor] = None,            # [bs,T,C,H,W]
        video_attention_mask: Optional[torch.Tensor] = None,    # [bs,T]
        past_key_values=None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        device = input_ids.device
        bs, Lt = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones(bs, Lt, dtype=torch.long, device=device)

        # 训练/首token时，把视觉token拼到文本前面
        if past_key_values is None and video_frames is not None:
            visual_embeds, visual_mask = self.visual_encoder(video_frames.to(device), video_attention_mask)
            text_embeds = self.llm.model.embed_tokens(input_ids)

            inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
            attention_mask = torch.cat([visual_mask.to(attention_mask.device), attention_mask], dim=1)
            position_ids = self._build_position_ids(attention_mask)

            if labels is not None:
                pad = torch.full((bs, visual_embeds.size(1)), -100, dtype=labels.dtype, device=device)
                labels = torch.cat([pad, labels], dim=1)

            return self.llm(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=labels,
                past_key_values=None,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        # 生成时：past_key_values!=None 走原始路径（视觉前缀已在第一次喂入）
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
        video_frames=None,
        video_attention_mask=None,
        **kwargs,
    ) -> Dict[str, Any]:
        # 第一次（past=None）把视觉前缀拼进去；后续 step 让 HF 自己处理
        if past_key_values is None and video_frames is not None:
            device = input_ids.device
            bs, Lt = input_ids.shape
            if attention_mask is None:
                attention_mask = torch.ones(bs, Lt, dtype=torch.long, device=device)

            visual_embeds, visual_mask = self.visual_encoder(video_frames.to(device), video_attention_mask)
            text_embeds = self.llm.model.embed_tokens(input_ids)

            inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
            attention_mask = torch.cat([visual_mask.to(attention_mask.device), attention_mask], dim=1)
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
        model_inputs.pop("video_frames", None)
        model_inputs.pop("video_attention_mask", None)
        return model_inputs