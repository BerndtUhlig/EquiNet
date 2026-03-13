from pathlib import Path
import numpy as np

CLASS_MAP = {
    "ceiling": 0,
    "floor": 1,
    "wall": 2,
    "beam": 3,
    "column": 4,
    "window": 5,
    "door": 6,
    "table": 7,
    "chair": 8,
    "sofa": 9,
    "bookcase": 10,
    "board": 11,
    "clutter": 12,
}

def process_room(anno_dir: Path, use_rgb=True, normalize_xyz=True):
    xs = []
    ys = []

    for txt_file in anno_dir.glob("*.txt"):
        class_name = txt_file.stem.split("_")[0].lower()
        if class_name not in CLASS_MAP:
            class_name = "clutter"

        label = CLASS_MAP[class_name]
        arr = np.loadtxt(txt_file, dtype=np.float32)

        if arr.ndim == 1:
            arr = arr[None, :]

        xyz = arr[:, :3]
        rgb = arr[:, 3:6]

        if use_rgb:
            feat = np.concatenate([xyz, rgb / 255.0], axis=1)
        else:
            feat = xyz

        xs.append(feat.astype(np.float32))
        ys.append(np.full(feat.shape[0], label, dtype=np.int64))

    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)

    if normalize_xyz:
        xyz = x[:, :3]
        center = xyz.mean(axis=0, keepdims=True)
        xyz = xyz - center
        scale = np.linalg.norm(xyz, axis=1).max()
        scale = max(scale, 1e-6)
        x[:, :3] = xyz / scale

    return x.astype(np.float32), y.astype(np.int64)


def preprocess_s3dis(raw_root, out_root, test_area=5, use_rgb=True, normalize_xyz=True):
    raw_root = Path(raw_root)
    out_root = Path(out_root)

    for split in ["train", "test"]:
        (out_root / split).mkdir(parents=True, exist_ok=True)

    for area_dir in sorted(raw_root.glob("Area_*")):
        area_num = int(area_dir.name.split("_")[1])
        split = "test" if area_num == test_area else "train"

        for room_dir in area_dir.iterdir():
            if not room_dir.is_dir():
                continue

            anno_dir = room_dir / "Annotations"
            if not anno_dir.exists():
                continue

            print("Processing", area_dir.name, room_dir.name)
            x, y = process_room(anno_dir, use_rgb=use_rgb, normalize_xyz=normalize_xyz)

            stem = f"{area_dir.name}_{room_dir.name}"
            np.save(out_root / split / f"{stem}_x.npy", x)
            np.save(out_root / split / f"{stem}_y.npy", y)


if __name__ == "__main__":
    preprocess_s3dis(
        raw_root="./Stanford3dDataset_v1.2/Stanford3dDataset_v1.2",
        out_root="./processed",
        test_area=5,
        use_rgb=True,
        normalize_xyz=True,
    )