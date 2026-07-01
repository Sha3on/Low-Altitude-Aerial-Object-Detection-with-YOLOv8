"""
Renders a one-page visual dashboard (PNG) of dataset statistics computed
from the final Group_3 submission tree - KPI cards, a per-video breakdown
table, and a per-scene class-distribution chart.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
GROUP_ROOT = ROOT / "Group_3"
SITES = ["DK_backyard", "DK_parking", "THI_Bikepark", "THI_Grass"]
CANONICAL_NAMES = {0: "Human", 1: "Vehicle", 2: "Bicycle"}
OUT_FILE = ROOT / "dataset" / "dataset_dashboard.png"

COLORS = {"Human": "#3D5A80", "Vehicle": "#EE6C4D", "Bicycle": "#98C1D9"}
BG = "#F7F7F2"


def collect_stats():
    per_video_rows = []
    per_scene_totals = {site: {c: 0 for c in CANONICAL_NAMES} for site in SITES}
    overall_class_counts = {c: 0 for c in CANONICAL_NAMES}
    total_images = 0
    total_objects = 0

    for site_name in SITES:
        site_dir = GROUP_ROOT / site_name
        video_dirs = sorted(p for p in site_dir.iterdir() if p.is_dir())
        for v_idx, video_dir in enumerate(video_dirs, start=1):
            images = sorted((video_dir / "images").glob("*.jpg"))
            obj_train_data = video_dir / "annotations" / "obj_train_data"

            class_counts = {c: 0 for c in CANONICAL_NAMES}
            empty = 0
            for img in images:
                label_path = obj_train_data / (img.stem + ".txt")
                non_empty = [l for l in label_path.read_text().splitlines() if l.strip()]
                if not non_empty:
                    empty += 1
                for line in non_empty:
                    cls_id = int(line.split()[0])
                    class_counts[cls_id] += 1
                    overall_class_counts[cls_id] += 1
                    per_scene_totals[site_name][cls_id] += 1

            obj_total = sum(class_counts.values())
            per_video_rows.append({
                "scene": site_name,
                "video": f"v{v_idx}",
                "images": len(images),
                "empty": empty,
                "total_obj": obj_total,
                **{CANONICAL_NAMES[c]: class_counts[c] for c in CANONICAL_NAMES},
            })
            total_images += len(images)
            total_objects += obj_total

    return per_video_rows, per_scene_totals, overall_class_counts, total_images, total_objects


def kpi_card(ax, value, label, color):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box = FancyBboxPatch((0.03, 0.08), 0.94, 0.84, boxstyle="round,pad=0.02,rounding_size=0.08",
                          linewidth=2, edgecolor=color, facecolor="white")
    ax.add_patch(box)
    ax.text(0.5, 0.58, f"{value:,}", ha="center", va="center", fontsize=26, fontweight="bold", color=color)
    ax.text(0.5, 0.24, label, ha="center", va="center", fontsize=11, color="#444444")


def main():
    rows, per_scene_totals, overall, total_images, total_objects = collect_stats()

    fig = plt.figure(figsize=(13, 9), facecolor=BG)
    fig.suptitle("Group_3 — Project 1 Dataset Statistics", fontsize=20, fontweight="bold",
                 x=0.05, ha="left", y=0.97)
    fig.text(0.05, 0.935, "Low-Altitude Aerial Object Detection · YOLO format",
              fontsize=11, color="#666666")

    kpis = [
        (total_images, "Total Images", "#3D5A80"),
        (total_objects, "Total Objects", "#293241"),
        (overall[0], "Human (class 0)", COLORS["Human"]),
        (overall[1], "Vehicle (class 1)", COLORS["Vehicle"]),
        (overall[2], "Bicycle (class 2)", COLORS["Bicycle"]),
    ]
    for i, (value, label, color) in enumerate(kpis):
        ax = fig.add_axes([0.04 + i * 0.192, 0.78, 0.17, 0.13])
        kpi_card(ax, value, label, color)

    ax_table = fig.add_axes([0.04, 0.06, 0.55, 0.66])
    ax_table.axis("off")
    col_labels = ["Scene", "Video", "Images", "Empty", "Total", "Human", "Vehicle", "Bicycle"]
    cell_text = [[r["scene"], r["video"], r["images"], r["empty"], r["total_obj"],
                  r["Human"], r["Vehicle"], r["Bicycle"]] for r in rows]
    table = ax_table.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        if r == 0:
            cell.set_facecolor("#293241")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("white" if r % 2 else "#F0F0EC")
    ax_table.set_title("Per-Video Breakdown", loc="left", fontsize=13, fontweight="bold", pad=10)

    ax_chart = fig.add_axes([0.66, 0.08, 0.30, 0.62])
    scenes = SITES
    y_pos = range(len(scenes))
    left = [0] * len(scenes)
    for cls_id, cls_name in CANONICAL_NAMES.items():
        vals = [per_scene_totals[s][cls_id] for s in scenes]
        ax_chart.barh(y_pos, vals, left=left, color=COLORS[cls_name], label=cls_name, height=0.6)
        left = [l + v for l, v in zip(left, vals)]
    ax_chart.set_yticks(list(y_pos))
    ax_chart.set_yticklabels(scenes)
    ax_chart.invert_yaxis()
    ax_chart.set_xlabel("Annotation count")
    ax_chart.set_title("Annotations per Scene", loc="left", fontsize=13, fontweight="bold")
    ax_chart.legend(loc="lower right", frameon=False, fontsize=9)
    for spine in ["top", "right"]:
        ax_chart.spines[spine].set_visible(False)
    ax_chart.set_facecolor(BG)

    n_videos = len(rows)
    n_scenes = len(SITES)
    fig.text(0.5, 0.015,
              f"Global total: {total_objects:,} objects across {total_images:,} frames · "
              f"{n_videos} video sequences · {n_scenes} scenes · Format: YOLO (.txt)",
              ha="center", fontsize=9, color="#666666")

    fig.savefig(OUT_FILE, dpi=160, facecolor=BG)
    print(f"Saved dashboard to {OUT_FILE}")


if __name__ == "__main__":
    main()
