"""
Builds the Group_3 submission tree required by the instructor, matching the
reference format observed in Group 1's correct submission:

    Group_3/
      <Site>/
        <VideoSequenceName>/      (no _mp4/_mov suffix)
          images/
            frame_00001.jpg, frame_00002.jpg, ...   (renumbered from 1)
          annotations/
            obj.data
            obj.names
            train.txt
            obj_train_data/
              frame_00001.txt, frame_00002.txt, ...

Canonical class order for Project 1 (confirmed by instructor): 0 Human,
1 Vehicle, 2 Bicycle. Our raw Roboflow labels use a different internal order
(0 Bicycle, 1 Human - verified visually, see RAW_CLASS_NAME below) so every
label line's class id is remapped to the canonical id before being written
out.

This only COPIES/transforms files into the new Group_3/ tree - the original
<Site>/<Site>/images|labels folders used by the training pipeline
(prepare_dataset.py, train.py, evaluate.py) are left untouched.
"""

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

GROUP_NAME = "Group_3"

SITES = [
    ("DK_backyard", ROOT / "DK_backyard" / "DK_backyard"),
    ("DK_parking", ROOT / "DK_parking" / "DK_parking"),
    ("THI_Bikepark", ROOT / "THI_Bikepark" / "THI_Bikepark"),
    ("THI_Grass", ROOT / "THI_Grass" / "THI_Grass"),
]

OUT_ROOT = ROOT / GROUP_NAME

# What each raw Roboflow class id actually represents in our original
# (Vehicle-less) export. Index = raw id, value = semantic name.
RAW_CLASS_NAME = {0: "Bicycle", 1: "Human", 2: "Vehicle"}

# Canonical Project 1 class scheme confirmed by the instructor.
CANONICAL_NAME_TO_ID = {"Human": 0, "Vehicle": 1, "Bicycle": 2}
CANONICAL_NAMES_IN_ORDER = ["Human", "Vehicle", "Bicycle"]

# Corrected re-export for DK_parking: separate Roboflow project with its own
# data.yaml (nc=2, names=['Human','Vehicle']) that adds the previously-missing
# Vehicle (parked car) annotations. Its raw ids already equal the canonical
# ones (0=Human, 1=Vehicle), so lines from this source are used as-is, not
# remapped through RAW_CLASS_NAME.
VEHICLE_FIX_LABELS_DIR = (
    ROOT / "My First Project.yolo26" / "train" / "labels"
)
VEHICLE_FIX_RAW_TO_CANONICAL = {0: CANONICAL_NAME_TO_ID["Human"], 1: CANONICAL_NAME_TO_ID["Vehicle"]}

# 5 DK_parking frames absent from the corrected export, filled in by
# interpolate_missing_vehicle.py (already in canonical ids - used verbatim).
VEHICLE_INTERPOLATED_DIR = ROOT / "dataset" / "vehicle_interpolated"

SEQUENCE_RE = re.compile(r"^(.*?)-(\d+)_jpg")
VIDEO_EXT_SUFFIX_RE = re.compile(r"_(mp4|mov)$", re.IGNORECASE)


def sequence_key(image_path: Path):
    m = SEQUENCE_RE.match(image_path.name)
    if not m:
        raise ValueError(f"Unrecognized filename pattern: {image_path.name}")
    return m.group(1), int(m.group(2))


def clean_video_name(raw_prefix: str) -> str:
    return VIDEO_EXT_SUFFIX_RE.sub("", raw_prefix)


def remap_label_line(line: str) -> str:
    parts = line.split()
    raw_id = int(parts[0])
    name = RAW_CLASS_NAME[raw_id]
    canonical_id = CANONICAL_NAME_TO_ID[name]
    return " ".join([str(canonical_id)] + parts[1:])


def remap_vehicle_fix_line(line: str) -> str:
    parts = line.split()
    raw_id = int(parts[0])
    canonical_id = VEHICLE_FIX_RAW_TO_CANONICAL[raw_id]
    return " ".join([str(canonical_id)] + parts[1:])


def main():
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)

    summary = []

    for site_name, site_root in SITES:
        images_dir = site_root / "images"
        labels_dir = site_root / "labels"

        images = sorted(images_dir.glob("*.jpg"))
        groups = {}
        for img in images:
            raw_prefix, idx = sequence_key(img)
            groups.setdefault(raw_prefix, []).append((idx, img))

        for raw_prefix, frames in sorted(groups.items()):
            video_name = clean_video_name(raw_prefix)
            seq_dir = OUT_ROOT / site_name / video_name
            images_out = seq_dir / "images"
            annot_out = seq_dir / "annotations"
            obj_train_data = annot_out / "obj_train_data"
            images_out.mkdir(parents=True, exist_ok=True)
            obj_train_data.mkdir(parents=True, exist_ok=True)

            frames.sort(key=lambda t: t[0])

            train_txt_lines = []
            fixed_count = 0
            for new_idx, (_, img) in enumerate(frames, start=1):
                frame_name = f"frame_{new_idx:05d}"

                vehicle_fix_label = (
                    VEHICLE_FIX_LABELS_DIR / (img.stem + ".txt")
                    if site_name == "DK_parking"
                    else None
                )
                interpolated_label = (
                    VEHICLE_INTERPOLATED_DIR / (img.stem + ".txt")
                    if site_name == "DK_parking"
                    else None
                )
                if vehicle_fix_label is not None and vehicle_fix_label.exists():
                    remapped_lines = [
                        remap_vehicle_fix_line(line)
                        for line in vehicle_fix_label.read_text().splitlines()
                        if line.strip()
                    ]
                    fixed_count += 1
                elif interpolated_label is not None and interpolated_label.exists():
                    remapped_lines = [
                        line for line in interpolated_label.read_text().splitlines()
                        if line.strip()
                    ]
                    fixed_count += 1
                else:
                    label_src = labels_dir / (img.stem + ".txt")
                    if not label_src.exists():
                        raise RuntimeError(f"Missing label file for image: {img}")
                    remapped_lines = [
                        remap_label_line(line)
                        for line in label_src.read_text().splitlines()
                        if line.strip()
                    ]

                shutil.copy2(img, images_out / f"{frame_name}.jpg")
                (obj_train_data / f"{frame_name}.txt").write_text(
                    "\n".join(remapped_lines) + ("\n" if remapped_lines else "")
                )
                train_txt_lines.append(f"data/obj_train_data/{frame_name}.jpg")

            if site_name == "DK_parking":
                print(f"  -> {fixed_count}/{len(frames)} frames used corrected "
                      f"Human/Vehicle labels; {len(frames) - fixed_count} fell back "
                      "to the old (no-Vehicle) labels - these still need fixing.")

            (annot_out / "obj.names").write_text(
                "\n".join(CANONICAL_NAMES_IN_ORDER) + "\n"
            )
            (annot_out / "obj.data").write_text(
                f"classes = {len(CANONICAL_NAMES_IN_ORDER)}\n"
                "train = data/train.txt\n"
                "names = data/obj.names\n"
                "backup = backup/\n"
            )
            (annot_out / "train.txt").write_text("\n".join(train_txt_lines) + "\n")

            summary.append(f"{site_name}/{video_name}: {len(frames)} frames")

    print("\n".join(summary))
    print(f"\nSubmission tree built at: {OUT_ROOT}")

    zip_path = ROOT / f"{GROUP_NAME}.zip"
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=ROOT, base_dir=GROUP_NAME)
    print(f"Zip created at: {zip_path}")


if __name__ == "__main__":
    main()
