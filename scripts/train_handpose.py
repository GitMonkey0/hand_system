import os
import argparse
import torch
from torch.utils.data import random_split

from transformers import Trainer, TrainingArguments

from handpose.data.dataset import HandPoseDataset
from handpose.data.transforms.hl import Compose, HLImageTransform, Keypoints3DToHLTargets
from handpose.models.hl_classifier import HLClassifier
from handpose.training.metrics import hl_top1_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="outputs/hl")
    p.add_argument("--encoder-ckpt", type=str, default=None)

    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--val-ratio", type=float, default=0.05)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    transform = Compose([
        HLImageTransform(image_size=args.image_size),
        Keypoints3DToHLTargets(assume_right_first=True),
    ])

    ds = HandPoseDataset(
        root=args.root,
        require_image=True,
        require_keypoints=True,
        transform=transform,
    )

    n_val = max(1, int(len(ds) * args.val_ratio))
    n_train = len(ds) - n_val
    g = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=g)

    model = HLClassifier(encoder_ckpt=args.encoder_ckpt)

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        dataloader_num_workers=args.num_workers,

        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.save_steps,
        evaluation_strategy="steps",
        save_strategy="steps",

        remove_unused_columns=False,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to="none",
        seed=args.seed,
        load_best_model_at_end=True,
        metric_for_best_model="labanacc@1",
        greater_is_better=True,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=hl_top1_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()