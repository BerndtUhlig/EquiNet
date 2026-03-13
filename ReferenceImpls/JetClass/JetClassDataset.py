from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# official helper from jet-universe/particle_transformer
from dataloader import read_file


class JetClassAnomalyDataset(Dataset):
    """
    Binary anomaly version of JetClass.

    normal_classes: set of class ids treated as label 0
    anomaly_classes: set of class ids treated as label 1

    By default this uses only particle-level features.
    """
    def __init__(
        self,
        files,
        normal_classes,
        anomaly_classes,
        max_jets=None,
        max_particles=128,
        use_jet_features=False,
        particle_feature_slice=None,
    ):
        self.X = []
        self.Y = []

        normal_classes = set(normal_classes)
        anomaly_classes = set(anomaly_classes)

        loaded = 0
        for fp in files:
            x_particles, x_jets, y_onehot = read_file(str(fp))

            # convert one-hot to integer class id
            cls = y_onehot.argmax(axis=1)

            # keep only selected classes
            keep = np.isin(cls, list(normal_classes | anomaly_classes))
            if not np.any(keep):
                continue

            x_particles = x_particles[keep]
            x_jets = x_jets[keep]
            cls = cls[keep]

            # particle arrays come as (num_jets, num_particle_features, max_num_particles)
            # convert to (num_jets, max_num_particles, num_particle_features)
            x_particles = np.transpose(x_particles, (0, 2, 1))

            # optional feature subset
            if particle_feature_slice is not None:
                x_particles = x_particles[:, :, particle_feature_slice]

            # optional truncate particle count
            x_particles = x_particles[:, :max_particles, :]

            # binary labels
            y_bin = np.isin(cls, list(anomaly_classes)).astype(np.int64)

            # optional concatenate jet-level features to every particle
            if use_jet_features:
                # x_jets shape: (num_jets, num_jet_features)
                jet_feats = np.repeat(x_jets[:, None, :], x_particles.shape[1], axis=1)
                x = np.concatenate([x_particles, jet_feats], axis=-1)
            else:
                x = x_particles

            self.X.append(torch.tensor(x, dtype=torch.float32))
            self.Y.append(torch.tensor(y_bin, dtype=torch.long))

            loaded += x.shape[0]
            if max_jets is not None and loaded >= max_jets:
                break

        if not self.X:
            raise RuntimeError("No samples loaded. Check your files and class selection.")

        self.X = torch.cat(self.X, dim=0)
        self.Y = torch.cat(self.Y, dim=0)

        if max_jets is not None and len(self.Y) > max_jets:
            self.X = self.X[:max_jets]
            self.Y = self.Y[:max_jets]

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]