import math, random
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co


def _rng(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def _randint(a: int, b: int) -> int:
    return random.randint(a,b)

def sample_normal_set(N: int, d: int, gen: torch.Generator, mu: torch.Tensor = None, sigma: float = 1.0) -> torch.Tensor:
    if mu is None:
        mu = torch.zeros(d)
    z = torch.randn((N, d), generator=gen)
    return z * sigma + mu

def _unit_dir(d:int , gen: torch.Generator) -> torch.Tensor:
    v = torch.randn(d, generator=gen)
    return v/(v.norm()+ 1e-8)

def single_outlier(X: torch.Tensor, gen: torch.Generator, strength: Tuple[float,float]) -> Tuple[torch.Tensor, torch.Tensor]:
    N, d = X.shape
    idx = _randint(0,N-1)
    out = X.clone()
    rstrength = strength[0] + (strength[1] - strength[0]) * torch.rand((), generator=gen).item()
    out[idx] = X.mean(0) + rstrength * _unit_dir(d,gen)
    y_pt = torch.zeros(N,dtype=torch.long)
    y_pt[idx] = 1
    return out, y_pt

def local_cluster(X: torch.Tensor,gen: torch.Generator, frac: Tuple[float,float],  offset: Tuple[float,float], radius: Tuple[float,float]) -> Tuple[torch.Tensor, torch.Tensor]:
    N, d = X.shape
    rfrac = frac[0] + (frac[1] - frac[0]) * torch.rand((), generator=gen).item()
    rradius= radius[0] + (radius[1] - radius[0]) * torch.rand((), generator=gen).item()
    roffset = offset[0] + (offset[1] - offset[0]) * torch.rand((), generator=gen).item()
    k = max(1, int(rfrac*N))
    indices = torch.randperm(N, generator=gen)[:k]
    center = X.mean(0) + roffset * _unit_dir(d,gen)
    out = X.clone()
    out[indices] = center + rradius * torch.randn((k,d), generator = gen)
    y_pt = torch.zeros(N, dtype=torch.long)
    y_pt[indices] = 1
    return out, y_pt

def mean_shift_subset(X: torch.Tensor, gen: torch.Generator,  frac: Tuple[float,float], delta: Tuple[float,float] ) -> Tuple[torch.Tensor, torch.Tensor]:
    N, d = X.shape
    rfrac = frac[0] + (frac[1] - frac[0]) * torch.rand((), generator=gen).item()
    rdelta = delta[0] + (delta[1] - delta[0]) * torch.rand((), generator=gen).item()
    k = max(1, int(rfrac * N))
    indices = torch.randperm(N, generator=gen)[:k]
    out = X.clone()
    out[indices] = out[indices] + rdelta* _unit_dir(d,gen)
    y_pt = torch.zeros(N, dtype=torch.long)
    y_pt[indices] = 1
    return out, y_pt

def cov_shift_subset(X: torch.Tensor, gen: torch.Generator,  frac: Tuple[float,float], scale: Tuple[float,float]) -> Tuple[torch.Tensor, torch.Tensor]:
    N,d = X.shape
    rfrac = frac[0] + (frac[1]-frac[0])*torch.rand((),generator=gen).item()
    rscale = scale[0] + (scale[1]-scale[0])*torch.rand((),generator=gen).item()
    k = max(1, int(rfrac * N))
    indices = torch.randperm(N, generator=gen)[:k]
    out = X.clone()
    out[indices] = X.mean(0) + rscale* torch.randn((k,d), generator=gen)
    y_pt = torch.zeros(N, dtype=torch.long)
    y_pt[indices] = 1
    return out, y_pt



_ANOM_FUNS = {
    "single_outlier": single_outlier,
    "local_cluster": local_cluster,
    "mean_shift": mean_shift_subset,
    "cov_shift": cov_shift_subset,
}
class AnomalySetDataset(Dataset):

    def __init__(self,
                 n_samples: int = 10000,
                 set_size: int = 100,
                 d: int = 8,
                 p_anom: float = 0.5,
                 base_sigma: float = 1.0,
                 modes = ("single_outlier", "local_cluster", "mean_shift", "cov_shift"),
                 seed: int = 42,
                 single_outlier_strength: Tuple[float,float] = (3.0,6.0),
                 local_cluster_k: Tuple[float,float] = (0.3,0.5),
                 local_cluster_offset: Tuple[float,float] = (1.5,3.0),
                 local_cluster_radius: Tuple[float,float] = (0.15,0.3),
                 mean_shift_delta: Tuple[float,float] = (2.5,4),
                 mean_shift_frac: Tuple[float,float] = (0.15,0.3),
                 cov_shift_scale: Tuple[float,float] = (3.0,5.0),
                 cov_shift_frac: Tuple[float,float] = (0.15,0.3),
                 ):
        self.n_samples = n_samples
        self.set_Size = set_size
        self.d = d
        self.p_anom = p_anom
        self.base_sigma = base_sigma
        self.modes = list(modes)
        self.gen = _rng(seed)
        self.kw = {
            "single_outlier": {"strength": single_outlier_strength},
            "local_cluster": {"frac": local_cluster_k, "offset": local_cluster_offset, "radius": local_cluster_radius},
            "mean_shift": {"frac": mean_shift_frac, "delta": mean_shift_delta},
            "cov_shift": { "frac": cov_shift_frac, "scale": cov_shift_scale}
        }

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index) -> T_co:
        X = sample_normal_set(self.set_Size,self.d,self.gen)
        if torch.rand((), generator=self.gen).item() >= self.p_anom:
            y_p = torch.zeros(self.set_Size)
            return X, y_p

        mode = random.choice(self.modes)
        args = self.kw.get(mode)
        fun = _ANOM_FUNS[mode]
        X_aug, Y_aug = fun(X,  gen=self.gen, **args)
        return X_aug, Y_aug




