from pathlib import Path
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset


def deterministic_ttv_split(shape_id: str, train=0.8, val=0.1):
    h = int(hashlib.md5(shape_id.encode()).hexdigest(), 16) % 10000
    r = h / 10000.0
    if r < train:
        return "train"
    if r < train + val:
        return "val"
    return "test"


class ShapeNetDataset(Dataset):
    def __init__(self, root, split, npoints=2048, category="02691156"):
        self.root = Path(root)
        self.npoints = npoints
        self.split = split

        # works for root="./ShapeNetPart"
        # and also for root="./ShapeNetPart/02691156"
        if (self.root / category).exists():
            self.cat_dir = self.root / category
        else:
            self.cat_dir = self.root

        self.pts_dir = self.cat_dir / "points"
        self.seg_root = self.cat_dir / "points_label"

        if not self.pts_dir.exists():
            raise FileNotFoundError(f"Missing points folder: {self.pts_dir}")
        if not self.seg_root.exists():
            raise FileNotFoundError(f"Missing points_label folder: {self.seg_root}")

        # part folders: body, engine, tail, wing, ...
        self.part_names = sorted([p.name for p in self.seg_root.iterdir() if p.is_dir()])
        if not self.part_names:
            raise RuntimeError(f"No part subfolders found in {self.seg_root}")

        self.part_to_label = {part: i for i, part in enumerate(self.part_names)}

        self.samples = []
        for pts_path in self.pts_dir.glob("*.pts"):
            shape_id = pts_path.stem

            part_seg_paths = {}
            for part in self.part_names:
                seg_path = self.seg_root / part / f"{shape_id}.seg"
                if seg_path.exists():
                    part_seg_paths[part] = seg_path

            if not part_seg_paths:
                continue

            if deterministic_ttv_split(shape_id) == split:
                self.samples.append((shape_id, pts_path, part_seg_paths))

        if len(self.samples) == 0:
            raise RuntimeError("No samples found. Check root path and folder layout.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        shape_id, pts_path, part_seg_paths = self.samples[idx]

        x = np.loadtxt(pts_path, dtype=np.float32)   # [N, d]
        N = x.shape[0]

        # initialize with ignore label, or background
        y = np.full(N, -1, dtype=np.int64)
        for part, seg_path in part_seg_paths.items():
            mask = np.loadtxt(seg_path, dtype=np.int64)
            if mask.ndim == 0:
                mask = np.array([mask], dtype=np.int64)

            if mask.shape[0] != N:
                raise ValueError(
                    f"Mismatch for shape {shape_id}, part {part}: "
                    f"{mask.shape[0]} labels vs {N} points"
                )

            # assume binary mask: 1 means point belongs to this part
            active = mask > 0
            y[active] = self.part_to_label[part]



        choice = np.random.choice(N, self.npoints, replace=(N < self.npoints))
        x = x[choice]
        y = y[choice]

        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(y).long(),
            "02691156",
            shape_id
        )