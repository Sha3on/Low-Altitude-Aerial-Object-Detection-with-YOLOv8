"""
Fills in the 5 DK_parking frames missing from the corrected Roboflow export
(-0001, -0088, -0089, -0093, -0094) by linearly interpolating the Vehicle box
between the nearest annotated frames before/after each gap. The car is
parked, so its position shifts smoothly as the drone moves - interpolation is
a safe approximation, not a guess.

Existing Human boxes for these 5 frames (from the original export) are kept
as-is; only the missing Vehicle line is added. Output is written to
dataset/vehicle_interpolated/<stem>.txt in canonical ids (0 Human, 1 Vehicle),
ready for build_submission.py to pick up as a second-priority override.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

OLD_LABELS_DIR = ROOT / "DK_parking" / "DK_parking" / "labels"
FIX_LABELS_DIR = ROOT / "My First Project.yolo26" / "train" / "labels"
OUT_DIR = ROOT / "dataset" / "vehicle_interpolated"

# stem (without .txt) -> (before_stem, after_stem, t)
GAPS = [
    ("DJI_20260319144429_0003_V_mp4-0001_jpg.rf.mFgIKz9wY01wySXERmzq",
     "DJI_20260319144429_0003_V_mp4-0000_jpg.rf.qdPAU9334oH42q2prc6O",
     "DJI_20260319144429_0003_V_mp4-0002_jpg.rf.rl8iOlhF1rFjjw8pCNhB", 1, 2),
    ("DJI_20260319144429_0003_V_mp4-0088_jpg.rf.jyDZN2IKjqhJThYzUe2C",
     "DJI_20260319144429_0003_V_mp4-0087_jpg.rf.Tb4W5P5TOUM1Cfv9tvIr",
     "DJI_20260319144429_0003_V_mp4-0090_jpg.rf.zArEdpvSQEMr1Uhq609q", 1, 3),
    ("DJI_20260319144429_0003_V_mp4-0089_jpg.rf.sbYmWadxErK9oQzH4ciL",
     "DJI_20260319144429_0003_V_mp4-0087_jpg.rf.Tb4W5P5TOUM1Cfv9tvIr",
     "DJI_20260319144429_0003_V_mp4-0090_jpg.rf.zArEdpvSQEMr1Uhq609q", 2, 3),
    ("DJI_20260319144429_0003_V_mp4-0093_jpg.rf.VYm8QzodL6HvUnAmIVPs",
     "DJI_20260319144429_0003_V_mp4-0092_jpg.rf.Vabm14zngl7FvaIlibYu",
     "DJI_20260319144429_0003_V_mp4-0095_jpg.rf.qy9WqQJBABkPAfdGmI8I", 1, 3),
    ("DJI_20260319144429_0003_V_mp4-0094_jpg.rf.u3AVSub4aaFvsO8KcLUs",
     "DJI_20260319144429_0003_V_mp4-0092_jpg.rf.Vabm14zngl7FvaIlibYu",
     "DJI_20260319144429_0003_V_mp4-0095_jpg.rf.qy9WqQJBABkPAfdGmI8I", 2, 3),
]

OLD_RAW_HUMAN_ID = 1  # in the original (Vehicle-less) DK_parking export
FIX_RAW_VEHICLE_ID = 1  # in the corrected export (0 Human, 1 Vehicle)
CANONICAL_HUMAN = 0
CANONICAL_VEHICLE = 1


def read_vehicle_box(stem: str):
    path = FIX_LABELS_DIR / f"{stem}.txt"
    for line in path.read_text().splitlines():
        parts = line.split()
        if int(parts[0]) == FIX_RAW_VEHICLE_ID:
            return [float(x) for x in parts[1:]]
    raise RuntimeError(f"No Vehicle box found in {path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for missing_stem, before_stem, after_stem, num, den in GAPS:
        t = num / den
        before_box = read_vehicle_box(before_stem)
        after_box = read_vehicle_box(after_stem)
        interp_box = [b + (a - b) * t for b, a in zip(before_box, after_box)]

        old_label_path = OLD_LABELS_DIR / f"{missing_stem}.txt"
        human_lines = []
        for line in old_label_path.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            raw_id = int(parts[0])
            if raw_id == OLD_RAW_HUMAN_ID:
                human_lines.append(f"{CANONICAL_HUMAN} " + " ".join(parts[1:]))

        vehicle_line = f"{CANONICAL_VEHICLE} " + " ".join(f"{v:.6f}" for v in interp_box)

        out_lines = human_lines + [vehicle_line]
        (OUT_DIR / f"{missing_stem}.txt").write_text("\n".join(out_lines) + "\n")
        print(f"{missing_stem[:40]}...: {len(human_lines)} Human box(es) + "
              f"1 interpolated Vehicle box (t={num}/{den})")
        print(f"  interpolated vehicle box: {interp_box}")

    print(f"\nWrote {len(GAPS)} files to {OUT_DIR}")


if __name__ == "__main__":
    main()
