import random
from typing import Tuple

import torch
from torch.utils.data import Dataset


def _rng(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def sample_normal_set(
    N: int,
    d: int,
    gen: torch.Generator,
    mu: torch.Tensor = None,
    sigma: float = 1.0
) -> torch.Tensor:
    if mu is None:
        mu = torch.zeros(d)
    z = torch.randn((N, d), generator=gen)
    return z * sigma + mu


def _unit_dir(d: int, gen: torch.Generator) -> torch.Tensor:
    v = torch.randn(d, generator=gen)
    return v / (v.norm() + 1e-8)


def single_outlier(
    X: torch.Tensor,
    gen: torch.Generator,
    py_rng: random.Random,
    strength: Tuple[float, float]
):
    N, d = X.shape
    idx = py_rng.randint(0, N - 1)
    out = X.clone()
    rstrength = strength[0] + (strength[1] - strength[0]) * torch.rand((), generator=gen).item()
    out[idx] = X.mean(0) + rstrength * _unit_dir(d, gen)
    y_pt = torch.zeros(N, dtype=torch.long)
    y_pt[idx] = 1
    return out, y_pt


def local_cluster(
    X: torch.Tensor,
    gen: torch.Generator,
    frac: Tuple[float, float],
    offset: Tuple[float, float],
    radius: Tuple[float, float]
):
    N, d = X.shape
    rfrac = frac[0] + (frac[1] - frac[0]) * torch.rand((), generator=gen).item()
    rradius = radius[0] + (radius[1] - radius[0]) * torch.rand((), generator=gen).item()
    roffset = offset[0] + (offset[1] - offset[0]) * torch.rand((), generator=gen).item()
    k = max(1, int(rfrac * N))
    indices = torch.randperm(N, generator=gen)[:k]
    center = X.mean(0) + roffset * _unit_dir(d, gen)
    out = X.clone()
    out[indices] = center + rradius * torch.randn((k, d), generator=gen)
    y_pt = torch.zeros(N, dtype=torch.long)
    y_pt[indices] = 1
    return out, y_pt


def mean_shift_subset(
    X: torch.Tensor,
    gen: torch.Generator,
    frac: Tuple[float, float],
    delta: Tuple[float, float]
):
    N, d = X.shape
    rfrac = frac[0] + (frac[1] - frac[0]) * torch.rand((), generator=gen).item()
    rdelta = delta[0] + (delta[1] - delta[0]) * torch.rand((), generator=gen).item()
    k = max(1, int(rfrac * N))
    indices = torch.randperm(N, generator=gen)[:k]
    out = X.clone()
    out[indices] = out[indices] + rdelta * _unit_dir(d, gen)
    y_pt = torch.zeros(N, dtype=torch.long)
    y_pt[indices] = 1
    return out, y_pt


def cov_shift_subset(
    X: torch.Tensor,
    gen: torch.Generator,
    frac: Tuple[float, float],
    scale: Tuple[float, float]
):
    N, d = X.shape
    rfrac = frac[0] + (frac[1] - frac[0]) * torch.rand((), generator=gen).item()
    rscale = scale[0] + (scale[1] - scale[0]) * torch.rand((), generator=gen).item()
    k = max(1, int(rfrac * N))
    indices = torch.randperm(N, generator=gen)[:k]
    out = X.clone()
    out[indices] = X.mean(0) + rscale * torch.randn((k, d), generator=gen)
    y_pt = torch.zeros(N, dtype=torch.long)
    y_pt[indices] = 1
    return out, y_pt


_ANOM_FUNS = {
    "single_outlier": single_outlier,
    "local_cluster": local_cluster,
    "mean_shift": mean_shift_subset,
    "cov_shift": cov_shift_subset,
}


class FixedAnomalySetDataset(Dataset):
    """
    Precompute every sample once so both models see exactly the same data.
    """
    def __init__(
        self,
        n_samples: int = 10000,
        set_size: int = 100,
        d: int = 8,
        p_anom: float = 0.5,
        base_sigma: float = 1.0,
        modes=("single_outlier", "local_cluster", "mean_shift", "cov_shift"),
        seed: int = 42,
        single_outlier_strength: Tuple[float, float] = (3.0, 6.0),
        local_cluster_k: Tuple[float, float] = (0.3, 0.5),
        local_cluster_offset: Tuple[float, float] = (1.5, 3.0),
        local_cluster_radius: Tuple[float, float] = (0.15, 0.3),
        mean_shift_delta: Tuple[float, float] = (2.5, 4.0),
        mean_shift_frac: Tuple[float, float] = (0.15, 0.3),
        cov_shift_scale: Tuple[float, float] = (3.0, 5.0),
        cov_shift_frac: Tuple[float, float] = (0.15, 0.3),
    ):
        self.n_samples = n_samples
        self.set_size = set_size
        self.d = d
        self.p_anom = p_anom
        self.base_sigma = base_sigma
        self.modes = list(modes)

        self.gen = _rng(seed)
        self.py_rng = random.Random(seed)

        self.kw = {
            "single_outlier": {"strength": single_outlier_strength},
            "local_cluster": {
                "frac": local_cluster_k,
                "offset": local_cluster_offset,
                "radius": local_cluster_radius,
            },
            "mean_shift": {"frac": mean_shift_frac, "delta": mean_shift_delta},
            "cov_shift": {"frac": cov_shift_frac, "scale": cov_shift_scale},
        }

        self.samples = []
        self._build()

    def _build(self):
        for _ in range(self.n_samples):
            X = sample_normal_set(self.set_size, self.d, self.gen, sigma=self.base_sigma)

            if torch.rand((), generator=self.gen).item() >= self.p_anom:
                y = torch.zeros(self.set_size, dtype=torch.long)
                self.samples.append((X, y))
                continue

            mode = self.py_rng.choice(self.modes)
            fun = _ANOM_FUNS[mode]
            args = self.kw[mode]

            if mode == "single_outlier":
                X_aug, y = fun(X, gen=self.gen, py_rng=self.py_rng, **args)
            else:
                X_aug, y = fun(X, gen=self.gen, **args)

            self.samples.append((X_aug, y))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        X, y = self.samples[index]
        return X.clone(), y.clone()