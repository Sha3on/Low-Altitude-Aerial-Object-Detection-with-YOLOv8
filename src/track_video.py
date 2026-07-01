"""
Live demo: runs the trained YOLOv8 model with Ultralytics' built-in
multi-object tracker (ByteTrack) over a video, showing a live window with
bounding boxes + persistent track IDs for Human / Vehicle / Bicycle, and
saves an annotated copy of the video to disk.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="path to a trained .pt checkpoint")
    p.add_argument("--source", required=True, help="path to a video file (see build_demo_video.py)")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml or botsort.yaml")
    p.add_argument("--name", default="track_demo")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)

    model.track(
        source=args.source,
        conf=args.conf,
        tracker=args.tracker,
        show=True,
        save=True,
        project=str(ROOT / "runs" / "track"),
        name=args.name,
        persist=True,
    )

    print(f"\nAnnotated tracking video saved under: {ROOT / 'runs' / 'track' / args.name}")


if __name__ == "__main__":
    main()
