import os
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


def _load_frame_list(frame_list_path: str):
    segments = defaultdict(list)
    with open(frame_list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seg, frame_str = line.split()
            segments[seg].append(int(frame_str))
    for seg in segments:
        segments[seg] = sorted(segments[seg])
    return dict(segments)


def _scan_cameras(mugsy_img_root: str):
    mugsy_img_root = Path(mugsy_img_root)
    if not mugsy_img_root.is_dir():
        raise FileNotFoundError(f"{mugsy_img_root} not found.")
    cam_ids = [p.name for p in mugsy_img_root.iterdir() if p.is_dir()]
    cam_ids.sort()
    return cam_ids


def _find_image_path(img_dir: Path, frame_idx: int):
    cand1 = img_dir / f"{frame_idx:06d}.png"
    if cand1.is_file():
        return cand1
    cand2 = img_dir / f"{frame_idx}.png"
    if cand2.is_file():
        return cand2
    return None


class HandPoseDataset(Dataset):
    """
    Online dataset loader for ReInterHand-like structure:

    ROOT/
      frame_list.txt
      keypoints_orig/keypoints_orig/{frame_idx}.json
      Mugsy_cameras/envmap_per_segment/images/{cam_id}/{frame}.png
    """

    def __init__(
        self,
        root: str,
        cam_ids=None,                 # None -> use all cams found
        require_image=True,           # True -> filter out missing images
        require_keypoints=True,       # True -> filter out missing kps
        image_mode="RGB",             # "RGB" or keep as-is
        transform=None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.image_mode = image_mode

        self.frame_list_path = self.root / "frame_list.txt"
        self.keypoints_dir = self.root / "keypoints_orig" / "keypoints_orig"
        self.mugsy_img_root = self.root / "Mugsy_cameras" / "envmap_per_segment" / "images"

        segments = _load_frame_list(str(self.frame_list_path))

        all_cam_ids = _scan_cameras(str(self.mugsy_img_root))
        if cam_ids is None:
            self.cam_ids = all_cam_ids
        else:
            missing = [c for c in cam_ids if c not in all_cam_ids]
            if missing:
                raise ValueError(f"Requested cams not found: {missing}. Available: {all_cam_ids}")
            self.cam_ids = list(cam_ids)

        # Build sample list: each sample = (seg_id, frame_idx, cam_id, img_path, kp_path)
        samples = []
        for seg_id, frames in segments.items():
            for frame_idx in frames:
                kp_path = self.keypoints_dir / f"{frame_idx}.json"
                kp_ok = kp_path.is_file()
                if require_keypoints and not kp_ok:
                    continue

                for cam_id in self.cam_ids:
                    img_dir = self.mugsy_img_root / cam_id
                    img_path = _find_image_path(img_dir, frame_idx)
                    img_ok = img_path is not None

                    if require_image and not img_ok:
                        continue

                    samples.append((seg_id, frame_idx, cam_id, img_path, kp_path))

        if len(samples) == 0:
            raise RuntimeError("No samples found. Check directory structure and flags.")
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def _load_keypoints(self, kp_path: Path):
        with open(kp_path, "r", encoding="utf-8") as f:
            kp = json.load(f)
        kp = np.asarray(kp, dtype=np.float32)
        if kp.shape != (42, 3):
            raise ValueError(f"Keypoints {kp_path} shape {kp.shape} != (42,3)")
        return kp  # (42,3) float32

    def _load_image(self, img_path: Path):
        if img_path is None:
            return None
        img = Image.open(img_path)
        if self.image_mode is not None:
            img = img.convert(self.image_mode)
        return img

    def __getitem__(self, idx):
        seg_id, frame_idx, cam_id, img_path, kp_path = self.samples[idx]

        image = self._load_image(img_path)
        keypoints_3d = self._load_keypoints(kp_path)

        sample = {
            "seg_id": seg_id,
            "frame_idx": frame_idx,
            "cam_id": cam_id,
            "image": image,                 # PIL.Image (or None)
            "keypoints_3d": keypoints_3d,   # np.ndarray (42,3)
            "image_path": str(img_path) if img_path is not None else None,
            "keypoints_path": str(kp_path),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample