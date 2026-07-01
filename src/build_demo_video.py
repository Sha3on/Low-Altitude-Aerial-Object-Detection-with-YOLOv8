"""
Reconstructs a playable .mp4 from one Group_3 video sequence's extracted
frames (the dataset only ships individual JPG frames, not the original
video files), so it can be used as a live-tracking demo source.
"""

import argparse
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
GROUP_ROOT = ROOT / "Group_3"
OUT_DIR = ROOT / "demo"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--site", default="DK_parking", help="e.g. DK_backyard, DK_parking, THI_Bikepark, THI_Grass")
    p.add_argument("--video", default=None, help="video sequence folder name; defaults to the first one found")
    p.add_argument("--fps", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    site_dir = GROUP_ROOT / args.site
    if not site_dir.exists():
        raise FileNotFoundError(f"Site not found: {site_dir}")

    if args.video:
        video_dir = site_dir / args.video
    else:
        video_dir = sorted(p for p in site_dir.iterdir() if p.is_dir())[0]

    images = sorted((video_dir / "images").glob("*.jpg"))
    if not images:
        raise RuntimeError(f"No frames found under {video_dir / 'images'}")

    first = cv2.imread(str(images[0]))
    height, width = first.shape[:2]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.site}_{video_dir.name}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (width, height))

    for img_path in images:
        frame = cv2.imread(str(img_path))
        writer.write(frame)
    writer.release()

    print(f"Reconstructed {len(images)} frames at {args.fps} fps -> {out_path}")


if __name__ == "__main__":
    main()
