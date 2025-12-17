import os
import math
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from PIL import Image

from transformers import (
    AutoTokenizer,
    Qwen3ForCausalLM,
    TrainingArguments,
    Trainer,
)

from slt.data.dataset import SLTDataset
from slt.models.qwen3_slt import Qwen3ForSignTranslation


# -------------------------
# Utils
# -------------------------
def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_chat(tokenizer, user_text: str, assistant_text: Optional[str] = None):
    """
    Returns:
      prompt_ids: only prompt (system+user(+assistant header))
      full_ids: prompt + assistant content
    """
    messages = [
        {"role": "system", "content": "你是一个专业的手语翻译助手。"},
        {"role": "user", "content": user_text},
    ]

    # 仅构造 prompt（不含 assistant 内容），用于 mask prompt loss
    prompt_str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # adds assistant header
    )
    prompt_ids = tokenizer(prompt_str, add_special_tokens=False).input_ids

    if assistant_text is None:
        return prompt_ids, None

    full_str = prompt_str + assistant_text
    full_ids = tokenizer(full_str, add_special_tokens=False).input_ids
    return prompt_ids, full_ids


# -------------------------
# Collator (text + video)
# -------------------------
@dataclass
class SLTDataCollator:
    tokenizer: Any
    image_size: int = 224
    max_length: int = 1024

    def _load_image_tensor(self, path: str) -> torch.Tensor:
        # Simple PIL -> tensor in [0,1], CHW
        img = Image.open(path).convert("RGB").resize((self.image_size, self.image_size))
        x = torch.from_numpy(__import__("numpy").array(img)).float() / 255.0  # HWC
        x = x.permute(2, 0, 1).contiguous()  # CHW
        return x

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # -------- text: build input_ids + labels (completion-only loss) --------
        input_ids_list = []
        labels_list = []
        attn_list = []

        # -------- video: pad frames to max_T --------
        max_T = max(b["num_frames"] for b in batch)

        video_frames = []
        video_attn = []

        for b in batch:
            # user prompt: "translate sign video to ..."; target: translation
            user_text = "请根据给定的手语视频内容，翻译成德语（translation）。只输出译文。"
            prompt_ids, full_ids = build_chat(self.tokenizer, user_text, b["translation"])
            if full_ids is None:
                raise RuntimeError("full_ids is None")

            # truncate
            full_ids = full_ids[: self.max_length]
            prompt_len = min(len(prompt_ids), len(full_ids))

            labels = [-100] * prompt_len + full_ids[prompt_len:]
            labels = labels[: len(full_ids)]

            input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))
            attn_list.append(torch.ones(len(full_ids), dtype=torch.long))

            # video
            # b["frames"] is frame_paths (dataset uses load_images=False by default)
            frame_paths: List[str] = b["frames"]
            T = len(frame_paths)

            # load frames to tensor [T,C,H,W]
            frames_t = torch.stack([self._load_image_tensor(p) for p in frame_paths], dim=0)

            # pad to [max_T,C,H,W]
            if T < max_T:
                pad = torch.zeros((max_T - T, *frames_t.shape[1:]), dtype=frames_t.dtype)
                frames_t = torch.cat([frames_t, pad], dim=0)
                vmask = torch.cat([torch.ones(T, dtype=torch.long), torch.zeros(max_T - T, dtype=torch.long)], dim=0)
            else:
                frames_t = frames_t[:max_T]
                vmask = torch.ones(max_T, dtype=torch.long)

            video_frames.append(frames_t)
            video_attn.append(vmask)

        # pad text to max_L
        max_L = max(x.size(0) for x in input_ids_list)
        input_ids = torch.full((len(batch), max_L), self.tokenizer.pad_token_id, dtype=torch.long)
        labels = torch.full((len(batch), max_L), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_L), dtype=torch.long)

        for i, (ids, lab, att) in enumerate(zip(input_ids_list, labels_list, attn_list)):
            L = ids.size(0)
            input_ids[i, :L] = ids
            labels[i, :L] = lab
            attention_mask[i, :L] = att

        video_frames = torch.stack(video_frames, dim=0)          # [bs,T,C,H,W]
        video_attention_mask = torch.stack(video_attn, dim=0)    # [bs,T]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "video_frames": video_frames,
            "video_attention_mask": video_attention_mask,
        }


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="PHOENIX-2014-T root")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs/qwen3_slt")

    parser.add_argument("--max_frames", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--out_visual_tokens", type=int, default=64)

    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)

    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)

    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_to", type=str, default="none")  # "wandb" etc.
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        # Qwen 系列通常 pad_token_id= eos
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    train_ds = SLTDataset(
        root=args.root,
        split="train",
        load_images=False,
        max_frames=args.max_frames,
    )
    dev_ds = SLTDataset(
        root=args.root,
        split="dev",
        load_images=False,
        max_frames=args.max_frames,
    )

    # LLM + multimodal wrapper
    llm = Qwen3ForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
        device_map=None,
    )
    if args.gradient_checkpointing:
        llm.gradient_checkpointing_enable()

    model = Qwen3ForSignTranslation(llm=llm, out_visual_tokens=args.out_visual_tokens)

    # Collator
    collator = SLTDataCollator(tokenizer=tokenizer, image_size=args.image_size, max_length=args.max_length)

    # Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,

        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",

        bf16=args.bf16,
        fp16=args.fp16,

        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,

        report_to=args.report_to,
        remove_unused_columns=False,  # important for video_frames/video_attention_mask
        dataloader_num_workers=4,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    # Train
    trainer.train()

    # Save
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()