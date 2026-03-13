from pathlib import Path
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset


def deterministic_train_val_split(stem: str, train_ratio=0.8):
    h = int(hashlib.md5(stem.encode()).hexdigest(), 16) % 10000
    return "train" if (h / 10000.0) < train_ratio else "val"


class S3DISProcessedDataset(Dataset):
    def __init__(self, root, split="train", npoints=512, mmap=True, train_ratio=0.8):
        self.root = Path(root)
        self.split = split
        self.npoints = npoints
        self.mmap = mmap

        if split == "test":
            base_dir = self.root / "test"
            x_files = sorted(base_dir.glob("*_x.npy"))
            self.samples = []
            for x_path in x_files:
                stem = x_path.stem[:-2]  # remove "_x"
                y_path = base_dir / f"{stem}_y.npy"
                if y_path.exists():
                    self.samples.append((stem, x_path, y_path))

        elif split in ("train", "val"):
            base_dir = self.root / "train"
            x_files = sorted(base_dir.glob("*_x.npy"))
            self.samples = []
            for x_path in x_files:
                stem = x_path.stem[:-2]
                y_path = base_dir / f"{stem}_y.npy"
                if not y_path.exists():
                    continue

                tv = deterministic_train_val_split(stem, train_ratio=train_ratio)
                if tv == split:
                    self.samples.append((stem, x_path, y_path))
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found for split={split} in {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        room_id, x_path, y_path = self.samples[idx]
        mmap_mode = "r" if self.mmap else None

        x = np.load(x_path, mmap_mode=mmap_mode)
        y = np.load(y_path, mmap_mode=mmap_mode)

        M = x.shape[0]
        choice = np.random.choice(M, self.npoints, replace=(M < self.npoints))

        x = np.asarray(x[choice], dtype=np.float32)
        y = np.asarray(y[choice], dtype=np.int64)

        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(y).long(),
            "s3dis",
            room_id,
        )