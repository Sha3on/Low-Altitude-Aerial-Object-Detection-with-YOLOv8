"""
Interactive single-image inference: lets the user supply a photo (either a
file path or a live webcam capture) and reports what the trained model finds
- Human, Vehicle, and/or Bicycle - with confidence scores and an annotated
image saved to disk.

Usage:
    python src/predict_photo.py --weights runs/train/d_yolo_exp2/weights/best.pt
    python src/predict_photo.py --weights ... --image path\to\photo.jpg
"""

import argparse
from pathlib import Path

import cv2

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "runs" / "predict_photo"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="path to a trained .pt checkpoint")
    p.add_argument("--image", default=None, help="path to an existing photo; if omitted, opens the webcam")
    p.add_argument("--camera", type=int, default=0, help="webcam device index")
    p.add_argument("--conf", type=float, default=0.25)
    return p.parse_args()


def capture_from_webcam(camera_index: int):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Could not open camera index {camera_index}.")
        return None

    print("Webcam preview opened. Press SPACE to capture a photo, ESC to cancel.")
    captured = None
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read from camera.")
            break
        cv2.imshow("Press SPACE to capture, ESC to cancel", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if key == 32:  # SPACE
            captured = frame
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured


def get_input_image(args):
    if args.image:
        path = Path(args.image)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Could not read image: {path}")
        return img

    frame = capture_from_webcam(args.camera)
    if frame is not None:
        return frame

    manual_path = input("No webcam capture available. Enter a path to a photo: ").strip()
    path = Path(manual_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    return img


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    image = get_input_image(args)
    model = YOLO(args.weights)

    results = model.predict(source=image, conf=args.conf, verbose=False)
    result = results[0]

    if len(result.boxes) == 0:
        print("\nNo Human, Vehicle, or Bicycle detected in this photo.")
    else:
        print(f"\nDetected {len(result.boxes)} object(s):")
        for box in result.boxes:
            cls_name = result.names[int(box.cls)]
            conf = float(box.conf)
            x1, y1, x2, y2 = (round(v) for v in box.xyxy[0].tolist())
            print(f"  - {cls_name}  confidence={conf:.2f}  box=({x1}, {y1}, {x2}, {y2})")

    annotated = result.plot()
    out_path = OUT_DIR / "last_capture_annotated.jpg"
    cv2.imwrite(str(out_path), annotated)
    print(f"\nAnnotated image saved to: {out_path}")

    cv2.imshow("Detection result (press any key to close)", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
