"""
Evaluates a trained YOLO checkpoint on the held-out test split and writes a
per-class metrics report (precision, recall, mAP50, mAP50-95) plus the
confusion matrix / PR curve plots that Ultralytics generates.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "dataset" / "data.yaml"
COLORS = {"Human": "#3D5A80", "Vehicle": "#EE6C4D", "Bicycle": "#98C1D9"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="path to best.pt from training")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.6)
    p.add_argument("--name", default="d_yolo_test_eval")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        project=str(ROOT / "runs" / "eval"),
        name=args.name,
        plots=True,
    )

    names = metrics.names
    print("\n=== Test-set results ===")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision (mean): {metrics.box.mp:.4f}")
    print(f"Recall (mean):    {metrics.box.mr:.4f}")

    print("\nPer-class:")
    print(f"{'class':<10}{'P':>8}{'R':>8}{'mAP50':>10}{'mAP50-95':>10}")
    class_indices = metrics.ap_class_index
    for i, cls_idx in enumerate(class_indices):
        cls_name = names[int(cls_idx)]
        p, r, ap50, ap = (
            metrics.box.p[i],
            metrics.box.r[i],
            metrics.box.ap50[i],
            metrics.box.ap[i],
        )
        print(f"{cls_name:<10}{p:>8.4f}{r:>8.4f}{ap50:>10.4f}{ap:>10.4f}")

    missing = sorted(set(names.values()) - {names[int(c)] for c in class_indices})
    if missing:
        print(f"\nNote: no ground-truth instances in the test split for: "
              f"{', '.join(missing)} (metrics undefined for these classes).")

    eval_dir = ROOT / "runs" / "eval" / args.name
    print(f"\nPlots and full results saved under: {eval_dir}")

    save_summary_dashboard(metrics, names, class_indices, eval_dir)


def save_summary_dashboard(metrics, names, class_indices, eval_dir):
    class_names = [names[int(c)] for c in class_indices]
    precisions = list(metrics.box.p)
    recalls = list(metrics.box.r)
    map50s = list(metrics.box.ap50)
    map5095s = list(metrics.box.ap)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#F7F7F2")

    ax = axes[0]
    x = range(len(class_names))
    width = 0.2
    metric_sets = [("Precision", precisions), ("Recall", recalls),
                   ("mAP50", map50s), ("mAP50-95", map5095s)]
    for i, (label, values) in enumerate(metric_sets):
        offsets = [xi + (i - 1.5) * width for xi in x]
        ax.bar(offsets, values, width=width, label=label)
    ax.set_xticks(list(x))
    ax.set_xticklabels(class_names)
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-class metrics", fontsize=13, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax.set_facecolor("#F7F7F2")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax2 = axes[1]
    overall = [metrics.box.mp, metrics.box.mr, metrics.box.map50, metrics.box.map]
    labels = ["Precision", "Recall", "mAP50", "mAP50-95"]
    bars = ax2.bar(labels, overall, color=["#3D5A80", "#EE6C4D", "#98C1D9", "#293241"])
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Overall test-set performance", fontsize=13, fontweight="bold", loc="left")
    for bar, val in zip(bars, overall):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}",
                  ha="center", fontsize=10, fontweight="bold")
    ax2.set_facecolor("#F7F7F2")
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)

    fig.suptitle("Evaluation Dashboard — Test Split", fontsize=16, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = eval_dir / "eval_dashboard.png"
    fig.savefig(out_path, dpi=150, facecolor="#F7F7F2")
    print(f"Evaluation dashboard saved to: {out_path}")


if __name__ == "__main__":
    main()
