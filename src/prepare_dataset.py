"""
Builds train/val/test split manifests from the corrected Group_3 dataset
(Group_3/<Site>/<VideoSequence>/images|annotations).

Group_3 uses annotations/obj_train_data/<frame>.txt (the instructor's
required submission layout), but Ultralytics YOLO auto-discovers labels by
swapping the LAST "/images/" segment of an image path for "/labels/" and
expects a sibling .txt there - it cannot see annotations/obj_train_data/ at
all. To bridge this without touching Group_3 (which must stay exactly in the
submitted format), a lightweight mirror is built under dataset/yolo_view/
mirroring each video's images/ (via hardlinks - zero extra disk space, same
volume) alongside a labels/ folder (small text-file copies) that Ultralytics
can auto-discover. Only dataset/yolo_view is referenced by data.yaml.

Each video sequence is split using contiguous temporal blocks (first N% of
frames -> train, next -> val, last -> test) rather than random shuffling,
since adjacent frames are near-duplicates (extracted from drone video).
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GROUP_ROOT = ROOT / "Group_3"
MIRROR_ROOT = ROOT / "dataset" / "yolo_view"

SITES = ["DK_backyard", "DK_parking", "THI_Bikepark", "THI_Grass"]

# Canonical Project 1 class scheme confirmed by the instructor.
CLASS_NAMES = {0: "Human", 1: "Vehicle", 2: "Bicycle"}

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

OUT_DIR = ROOT / "dataset" / "splits"


def contiguous_split(items, train_ratio, val_ratio, test_ratio):
    n = len(items)
    if n == 1:
        return items, [], []
    if n == 2:
        return [items[0]], [items[1]], []

    n_test = max(1, round(n * test_ratio))
    n_val = max(1, round(n * val_ratio))
    n_val = min(n_val, n - n_test - 1) if n - n_test - 1 > 0 else 0
    n_test = min(n_test, n - 1)
    n_train = n - n_val - n_test

    return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]


def link_or_copy(src: Path, dst: Path):
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def main():
    if MIRROR_ROOT.exists():
        shutil.rmtree(MIRROR_ROOT)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    splits = {"train": [], "val": [], "test": []}
    report_lines = []

    for site_name in SITES:
        site_dir = GROUP_ROOT / site_name
        video_dirs = sorted(p for p in site_dir.iterdir() if p.is_dir())

        site_counts = {"train": 0, "val": 0, "test": 0}
        site_class_counts = {"train": {}, "val": {}, "test": {}}

        for video_dir in video_dirs:
            images = sorted((video_dir / "images").glob("*.jpg"))
            obj_train_data = video_dir / "annotations" / "obj_train_data"

            mirror_video_dir = MIRROR_ROOT / site_name / video_dir.name
            mirror_images_dir = mirror_video_dir / "images"
            mirror_labels_dir = mirror_video_dir / "labels"
            mirror_images_dir.mkdir(parents=True, exist_ok=True)
            mirror_labels_dir.mkdir(parents=True, exist_ok=True)

            mirrored_images = []
            for img in images:
                label_src = obj_train_data / (img.stem + ".txt")
                if not label_src.exists():
                    raise RuntimeError(f"Missing annotation for image: {img}")

                mirror_img = mirror_images_dir / img.name
                link_or_copy(img, mirror_img)
                shutil.copy2(label_src, mirror_labels_dir / (img.stem + ".txt"))
                mirrored_images.append(mirror_img)

            train, val, test = contiguous_split(mirrored_images, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

            for split_name, split_images in [("train", train), ("val", val), ("test", test)]:
                for img in split_images:
                    splits[split_name].append(img)
                    site_counts[split_name] += 1
                    label_path = mirror_labels_dir / (img.stem + ".txt")
                    for line in label_path.read_text().splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        cls_id = int(line.split()[0])
                        site_class_counts[split_name][cls_id] = (
                            site_class_counts[split_name].get(cls_id, 0) + 1
                        )

        report_lines.append(f"{site_name}:")
        for split_name in ["train", "val", "test"]:
            cls_summary = ", ".join(
                f"{CLASS_NAMES.get(c, c)}={n}"
                for c, n in sorted(site_class_counts[split_name].items())
            )
            report_lines.append(
                f"  {split_name}: {site_counts[split_name]} images"
                + (f" ({cls_summary})" if cls_summary else " (0 instances)")
            )

    for split_name, image_list in splits.items():
        out_file = OUT_DIR / f"{split_name}.txt"
        with out_file.open("w") as f:
            for img in image_list:
                f.write(str(img.resolve()) + "\n")
        report_lines.append(f"\nTotal {split_name}: {len(image_list)} images -> {out_file}")

    print("\n".join(report_lines))

    data_yaml = ROOT / "dataset" / "data.yaml"
    data_yaml.write_text(
        "train: {}\nval: {}\ntest: {}\nnc: {}\nnames:\n{}\n".format(
            (OUT_DIR / "train.txt").resolve().as_posix(),
            (OUT_DIR / "val.txt").resolve().as_posix(),
            (OUT_DIR / "test.txt").resolve().as_posix(),
            len(CLASS_NAMES),
            "\n".join(f"  {k}: {v}" for k, v in sorted(CLASS_NAMES.items())),
        )
    )
    print(f"\nWrote {data_yaml}")


if __name__ == "__main__":
    main()
