"""
Computes dataset statistics directly from the final Group_3 submission tree
(Group_3/<Site>/<VideoSequence>/images|annotations) and writes a report to
dataset/dataset_statistics.txt.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GROUP_ROOT = ROOT / "Group_3"

SITES = ["DK_backyard", "DK_parking", "THI_Bikepark", "THI_Grass"]

CANONICAL_NAMES = {0: "Human", 1: "Vehicle", 2: "Bicycle"}

OUT_FILE = ROOT / "dataset" / "dataset_statistics.txt"


def main():
    lines = []
    lines.append("DATASET STATISTICS")
    lines.append("=" * 60)

    per_site_image_counts = {}
    per_site_video_counts = {}
    per_site_class_counts = {}
    per_site_empty_counts = {}
    overall_class_counts = {c: 0 for c in CANONICAL_NAMES}
    total_images = 0
    total_empty = 0

    for site_name in SITES:
        site_dir = GROUP_ROOT / site_name
        video_dirs = sorted(p for p in site_dir.iterdir() if p.is_dir())

        n_images = 0
        class_counts = {c: 0 for c in CANONICAL_NAMES}
        empty_count = 0

        for video_dir in video_dirs:
            images = sorted((video_dir / "images").glob("*.jpg"))
            obj_train_data = video_dir / "annotations" / "obj_train_data"

            for img in images:
                label_path = obj_train_data / (img.stem + ".txt")
                if not label_path.exists():
                    raise RuntimeError(f"Missing annotation for image: {img}")
                non_empty_lines = [
                    l for l in label_path.read_text().splitlines() if l.strip()
                ]
                if not non_empty_lines:
                    empty_count += 1
                for line in non_empty_lines:
                    cls_id = int(line.split()[0])
                    class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
                    overall_class_counts[cls_id] = overall_class_counts.get(cls_id, 0) + 1

            n_images += len(images)

        per_site_image_counts[site_name] = n_images
        per_site_video_counts[site_name] = len(video_dirs)
        per_site_class_counts[site_name] = class_counts
        per_site_empty_counts[site_name] = empty_count
        total_images += n_images
        total_empty += empty_count

    lines.append("")
    lines.append(f"TOTAL NUMBER OF IMAGES IN DATASET: {total_images}")
    lines.append("")
    lines.append("NUMBER OF IMAGES PER SUBFOLDER (SITE):")
    lines.append("-" * 60)
    lines.append(f"{'Site':<16}{'Videos':>9}{'Images':>10}{'% of total':>14}")
    for site_name in SITES:
        n_img = per_site_image_counts[site_name]
        n_vid = per_site_video_counts[site_name]
        pct = (n_img / total_images * 100) if total_images else 0
        lines.append(f"{site_name:<16}{n_vid:>9}{n_img:>10}{pct:>13.1f}%")
    lines.append("-" * 60)
    n_videos_total = sum(per_site_video_counts.values())
    lines.append(f"{'TOTAL':<16}{n_videos_total:>9}{total_images:>10}{100.0:>13.1f}%")

    lines.append("")
    lines.append("ANNOTATION INSTANCE COUNTS PER CLASS (PER SITE):")
    lines.append("-" * 60)
    header = f"{'Site':<16}" + "".join(f"{name:>12}" for name in CANONICAL_NAMES.values())
    lines.append(header)
    for site_name in SITES:
        counts = per_site_class_counts[site_name]
        row = f"{site_name:<16}" + "".join(
            f"{counts.get(c, 0):>12}" for c in CANONICAL_NAMES
        )
        lines.append(row)
    lines.append("-" * 60)
    total_row = f"{'TOTAL':<16}" + "".join(
        f"{overall_class_counts.get(c, 0):>12}" for c in CANONICAL_NAMES
    )
    lines.append(total_row)

    lines.append("")
    lines.append("BACKGROUND (ZERO-OBJECT) IMAGES PER SITE:")
    lines.append("-" * 60)
    for site_name in SITES:
        n_img = per_site_image_counts[site_name]
        n_empty = per_site_empty_counts[site_name]
        pct = (n_empty / n_img * 100) if n_img else 0
        lines.append(f"{site_name:<16}{n_empty:>6} / {n_img:<6} empty ({pct:.1f}%)")
    lines.append(f"{'TOTAL':<16}{total_empty:>6} / {total_images:<6} empty "
                  f"({(total_empty/total_images*100) if total_images else 0:.1f}%)")

    report = "\n".join(lines)
    print(report)
    OUT_FILE.write_text(report + "\n")
    print(f"\nSaved to: {OUT_FILE}")


if __name__ == "__main__":
    main()
