"""
Trains a YOLO detector (Human / Vehicle / Bicycle) on the corrected Group_3
dataset described by dataset/data.yaml (built by prepare_dataset.py).
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "dataset" / "data.yaml"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolov8n.pt", help="base/pretrained weights to start from")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--patience", type=int, default=10, help="early stopping patience")
    p.add_argument("--name", default="d_yolo_exp2")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"{DATA_YAML} not found - run prepare_dataset.py first")

    model = YOLO(args.model)
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        project=str(ROOT / "runs" / "train"),
        name=args.name,
        seed=args.seed,
        plots=True,
    )


if __name__ == "__main__":
    main()
