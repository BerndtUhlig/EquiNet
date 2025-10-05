#!/usr/bin/env python
"""
Generate ModelNet40_cloud.h5 for DeepSets Point‑Cloud experiments.

Usage
-----
python make_modelnet40_cloud.py \
       --modelnet_dir /path/to/ModelNet40 \
       --outfile      ModelNet40_cloud.h5 \
       --points       10000
"""
import argparse, pathlib, h5py, numpy as np, trimesh
from tqdm import tqdm

def sample_mesh(mesh: trimesh.Trimesh, n_pts: int = 10_000) -> np.ndarray:
    """Uniformly sample `n_pts` points on the surface of `mesh`."""
    return mesh.sample(n_pts).astype(np.float32)          # (n_pts,3)

def collect_split(split_dir: pathlib.Path, label: int, n_pts: int):
    clouds, labels = [], []
    for off_file in sorted(split_dir.glob("*.off")):
        mesh = trimesh.load(off_file, force='mesh')
        clouds.append(sample_mesh(mesh, n_pts))
        labels.append(label)
    return clouds, labels

def main(modelnet_dir, outfile, n_pts):
    modelnet_dir = pathlib.Path(modelnet_dir).expanduser()
    class_names = sorted([d.name for d in modelnet_dir.iterdir() if d.is_dir()])

    tr_cloud, tr_label, te_cloud, te_label = [], [], [], []

    print("Sampling point clouds …")
    for lbl, cname in enumerate(class_names):
        cls_dir = modelnet_dir / cname
        print(f"[{lbl:02d}] {cname}")

        clouds, labels = collect_split(cls_dir/"train", lbl, n_pts)
        tr_cloud.extend(clouds); tr_label.extend(labels)

        clouds, labels = collect_split(cls_dir/"test",  lbl, n_pts)
        te_cloud.extend(clouds); te_label.extend(labels)

    # Stack into (N, M, 3) + (N,)
    tr_cloud  = np.stack(tr_cloud,  axis=0)   # (9843,10000,3)
    tr_label  = np.array(tr_label, dtype=np.int64)
    te_cloud  = np.stack(te_cloud,  axis=0)   # (2468,10000,3)
    te_label  = np.array(te_label, dtype=np.int64)

    print("\nWriting HDF5 …")
    with h5py.File(outfile, "w") as f:
        f.create_dataset("tr_cloud",  data=tr_cloud,  compression="gzip")
        f.create_dataset("tr_labels",  data=tr_label)
        f.create_dataset("test_cloud",data=te_cloud,  compression="gzip")
        f.create_dataset("test_labels",data=te_label)
        f.create_dataset("class_names",
                         data=np.array(class_names, dtype="S"))

    print(f"Done.  Saved to {outfile}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelnet_dir", required=True)
    ap.add_argument("--outfile",      default="ModelNet40_cloud.h5")
    ap.add_argument("--points",       type=int, default=10_000)
    args = ap.parse_args()
    main(args.modelnet_dir, args.outfile, args.points)
