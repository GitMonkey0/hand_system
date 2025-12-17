# src/slt/data/dataset.py
import os
import glob
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image


@dataclass
class Sample:
    name: str
    speaker: str
    orth: str
    translation: str
    frame_paths: List[str]


class SLTDataset(Dataset):
    """
    Phoenix2014T dataset with auto-discovery from a single root.

    Directory example:
      root/
        annotations/manual/PHOENIX-2014-T.{train,dev,test}.corpus.csv
        features/fullFrame-210x260px/{train,dev,test}/...

    Usage:
      ds = MyDataset(root="/path/to/PHOENIX-2014-T", split="dev")
    """

    def __init__(
        self,
        root: str,                 # only required path
        split: str,                # "train" / "dev" / "test"
        load_images: bool = False,
        transform=None,
        max_frames: Optional[int] = None,
        # optional overrides (normally you don't pass these)
        features_root: Optional[str] = None,
        anno_csv: Optional[str] = None,
        features_dirname: Optional[str] = None,  # e.g. "fullFrame-210x260px"
    ):
        self.root = os.path.abspath(root)
        self.split = split
        self.load_images = load_images
        self.transform = transform
        self.max_frames = max_frames

        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Root dir not found: {self.root}")

        self.features_root = features_root or self._find_features_root(features_dirname)
        self.anno_csv = anno_csv or self._find_anno_csv()

        self.split_root = os.path.join(self.features_root, self.split)
        if not os.path.isdir(self.split_root):
            raise FileNotFoundError(f"Split dir not found: {self.split_root}")

        self.samples: List[Sample] = self._load_annotations()

    def _find_features_root(self, features_dirname: Optional[str]) -> str:
        # Prefer root/features/<features_dirname>
        if features_dirname:
            cand = os.path.join(self.root, "features", features_dirname)
            if os.path.isdir(cand):
                return cand

        # Common default
        default = os.path.join(self.root, "features", "fullFrame-210x260px")
        if os.path.isdir(default):
            return default

        # Otherwise search for a directory that contains train/dev/test subdirs
        # under root/features/**
        candidates = []
        for d in glob.glob(os.path.join(self.root, "features", "**"), recursive=True):
            if not os.path.isdir(d):
                continue
            if all(os.path.isdir(os.path.join(d, s)) for s in ["train", "dev", "test"]):
                candidates.append(d)

        if not candidates:
            raise FileNotFoundError(
                f"Could not find features root under {self.root}/features "
                f"(expect a dir containing train/dev/test)."
            )

        # choose the shortest path (closest)
        candidates.sort(key=lambda x: len(x))
        return candidates[0]

    def _find_anno_csv(self) -> str:
        # Prefer root/annotations/manual/*{split}*.corpus.csv
        manual_dir = os.path.join(self.root, "annotations", "manual")
        patterns = [
            os.path.join(manual_dir, f"*{self.split}*.corpus.csv"),
            os.path.join(self.root, "annotations", "**", f"*{self.split}*.corpus.csv"),
            os.path.join(self.root, "**", f"*{self.split}*.corpus.csv"),
        ]
        for pat in patterns:
            hits = sorted(glob.glob(pat, recursive=True))
            if hits:
                # Prefer exact "PHOENIX-2014-T.{split}.corpus.csv" if exists
                exact = [h for h in hits if os.path.basename(h) == f"PHOENIX-2014-T.{self.split}.corpus.csv"]
                return exact[0] if exact else hits[0]

        raise FileNotFoundError(
            f"Could not find annotation corpus csv for split='{self.split}' under {self.root} "
            f"(expect something like annotations/manual/PHOENIX-2014-T.{self.split}.corpus.csv)."
        )

    def _load_annotations(self) -> List[Sample]:
        samples: List[Sample] = []

        with open(self.anno_csv, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        header = lines[0].split("|")
        col = {k: i for i, k in enumerate(header)}

        required = ["name", "video", "speaker", "orth", "translation"]
        for k in required:
            if k not in col:
                raise ValueError(f"Missing column '{k}' in {self.anno_csv}, got header={header}")

        for line in lines[1:]:
            parts = line.split("|")
            name = parts[col["name"]]
            video = parts[col["video"]]
            speaker = parts[col["speaker"]]
            orth = parts[col["orth"]]
            translation = parts[col["translation"]]

            pattern = os.path.join(self.split_root, video)  # e.g. .../dev/<clip>/*.png
            frame_paths = sorted(glob.glob(pattern))
            if not frame_paths:
                raise FileNotFoundError(f"No frames found for pattern: {pattern}")

            if self.max_frames is not None:
                frame_paths = frame_paths[: self.max_frames]

            samples.append(Sample(
                name=name,
                speaker=speaker,
                orth=orth,
                translation=translation,
                frame_paths=frame_paths
            ))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_frame(self, path: str):
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]

        if self.load_images:
            frames = [self._load_frame(p) for p in s.frame_paths]
            if frames and torch.is_tensor(frames[0]):
                frames = torch.stack(frames, dim=0)  # [T,C,H,W]
        else:
            frames = s.frame_paths

        return {
            "name": s.name,
            "speaker": s.speaker,
            "orth": s.orth,
            "translation": s.translation,
            "frames": frames,
            "num_frames": len(s.frame_paths),
        }