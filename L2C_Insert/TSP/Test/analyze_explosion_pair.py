import argparse
import csv
import os
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


def load_results(csv_path: str) -> Dict[Tuple[str, str], Dict[str, str]]:
    """
    Load tour_lengths.csv into a dict keyed by (instance_id, problem_size).
    """
    data: Dict[Tuple[str, str], Dict[str, str]] = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row.get("instance_id", "").strip(), row.get("problem_size", "").strip())
            if not key[0]:
                continue
            data[key] = row
    return data


def compute_comparison(
    baseline: Dict[Tuple[str, str], Dict[str, str]],
    advanced: Dict[Tuple[str, str], Dict[str, str]],
) -> List[Dict[str, str]]:
    """
    Join baseline and advanced by (instance_id, problem_size) and compute deltas.
    """
    rows: List[Dict[str, str]] = []
    common_keys = sorted(set(baseline.keys()) & set(advanced.keys()))

    for key in common_keys:
        b = baseline[key]
        a = advanced[key]
        try:
            b_len = float(b.get("tour_length", "nan"))
            a_len = float(a.get("tour_length", "nan"))
        except ValueError:
            continue
        if not np.isfinite(b_len) or not np.isfinite(a_len) or b_len == 0:
            continue

        abs_impr = b_len - a_len
        rel_impr = abs_impr / b_len

        rows.append(
            {
                "instance_id": key[0],
                "problem_size": key[1],
                "tour_length_baseline": f"{b_len:.6f}",
                "tour_length_advanced": f"{a_len:.6f}",
                "abs_improvement": f"{abs_impr:.6f}",
                "rel_improvement": f"{rel_impr:.6f}",
            }
        )

    return rows


def aggregate_by_size(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
    by_size: Dict[str, Dict[str, List[float]]] = {}
    for r in rows:
        size = r["problem_size"]
        abs_imp = float(r["abs_improvement"])
        rel_imp = float(r["rel_improvement"])
        if size not in by_size:
            by_size[size] = {
                "abs": [],
                "rel": [],
            }
        by_size[size]["abs"].append(abs_imp)
        by_size[size]["rel"].append(rel_imp)

    stats: Dict[str, Dict[str, float]] = {}
    for size, vals in by_size.items():
        abs_arr = np.asarray(vals["abs"], dtype=float)
        rel_arr = np.asarray(vals["rel"], dtype=float)
        stats[size] = {
            "count": float(len(abs_arr)),
            "abs_mean": float(abs_arr.mean()) if abs_arr.size else float("nan"),
            "abs_min": float(abs_arr.min()) if abs_arr.size else float("nan"),
            "abs_max": float(abs_arr.max()) if abs_arr.size else float("nan"),
            "rel_mean": float(rel_arr.mean()) if rel_arr.size else float("nan"),
            "rel_min": float(rel_arr.min()) if rel_arr.size else float("nan"),
            "rel_max": float(rel_arr.max()) if rel_arr.size else float("nan"),
        }
    return stats


def save_comparison_csv(rows: List[Dict[str, str]], out_path: str) -> None:
    if not rows:
        return
    fieldnames = [
        "instance_id",
        "problem_size",
        "tour_length_baseline",
        "tour_length_advanced",
        "abs_improvement",
        "rel_improvement",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def save_stats_txt(stats: Dict[str, Dict[str, float]], out_path: str) -> None:
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("COMPARISON STATISTICS (baseline vs advanced_sampling)")
    lines.append("=" * 80)
    if not stats:
        lines.append("No overlapping instances between runs.")
    else:
        lines.append(
            f"{'Size':<10} {'Count':<10} "
            f"{'Abs mean':<15} {'Abs min':<15} {'Abs max':<15} "
            f"{'Rel mean':<15} {'Rel min':<15} {'Rel max':<15}"
        )
        lines.append("-" * 80)
        for size in sorted(stats.keys(), key=lambda x: int(x)):
            s = stats[size]
            lines.append(
                f"{size:<10} {int(s['count']):<10d} "
                f"{s['abs_mean']:<15.6f} {s['abs_min']:<15.6f} {s['abs_max']:<15.6f} "
                f"{s['rel_mean']:<15.6f} {s['rel_min']:<15.6f} {s['rel_max']:<15.6f}"
            )
    lines.append("=" * 80)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def combine_plots(
    baseline_dir: str,
    advanced_dir: str,
    output_dir: str,
) -> None:
    """
    For each common *_plot.png in final_solutions, create side-by-side comparison.
    """
    base_fs = os.path.join(baseline_dir, "final_solutions")
    adv_fs = os.path.join(advanced_dir, "final_solutions")
    if not (os.path.isdir(base_fs) and os.path.isdir(adv_fs)):
        return

    os.makedirs(output_dir, exist_ok=True)

    baseline_files = {
        f
        for f in os.listdir(base_fs)
        if f.lower().endswith(".png")
    }
    advanced_files = {
        f
        for f in os.listdir(adv_fs)
        if f.lower().endswith(".png")
    }
    common = sorted(baseline_files & advanced_files)

    for fname in common:
        try:
            img_b = Image.open(os.path.join(base_fs, fname)).convert("RGBA")
            img_a = Image.open(os.path.join(adv_fs, fname)).convert("RGBA")
        except Exception:
            continue

        h = max(img_b.height, img_a.height)
        w = img_b.width + img_a.width
        combined = Image.new("RGBA", (w, h), (255, 255, 255, 255))
        combined.paste(img_b, (0, 0))
        combined.paste(img_a, (img_b.width, 0))

        out_name = fname.replace("_plot.png", "_compare.png")
        if out_name == fname:
            out_name = f"{os.path.splitext(fname)[0]}_compare.png"
        combined.save(os.path.join(output_dir, out_name))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two explosion runs (baseline vs advanced_sampling): "
            "tour_lengths.csv + финальные картинки."
        )
    )
    parser.add_argument(
        "--baseline_dir",
        type=str,
        required=True,
        help="Папка с результатами baseline (директория, где лежит tour_lengths.csv).",
    )
    parser.add_argument(
        "--advanced_dir",
        type=str,
        required=True,
        help="Папка с результатами advanced_sampling (директория, где лежит tour_lengths.csv).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Куда сохранять сравнение (по умолчанию: под advanced_dir/compare_with_baseline).",
    )

    args = parser.parse_args()

    baseline_csv = os.path.join(args.baseline_dir, "tour_lengths.csv")
    advanced_csv = os.path.join(args.advanced_dir, "tour_lengths.csv")

    if not os.path.isfile(baseline_csv):
        raise FileNotFoundError(f"Baseline CSV not found: {baseline_csv}")
    if not os.path.isfile(advanced_csv):
        raise FileNotFoundError(f"Advanced CSV not found: {advanced_csv}")

    out_dir = args.output_dir or os.path.join(args.advanced_dir, "compare_with_baseline")
    os.makedirs(out_dir, exist_ok=True)

    baseline = load_results(baseline_csv)
    advanced = load_results(advanced_csv)
    rows = compute_comparison(baseline, advanced)

    if not rows:
        print("No common instances between baseline and advanced runs.")
        return

    comparison_csv = os.path.join(out_dir, "comparison_baseline_vs_advanced.csv")
    stats_txt = os.path.join(out_dir, "comparison_stats.txt")
    plots_dir = os.path.join(out_dir, "combined_plots")

    save_comparison_csv(rows, comparison_csv)
    stats = aggregate_by_size(rows)
    save_stats_txt(stats, stats_txt)
    combine_plots(args.baseline_dir, args.advanced_dir, plots_dir)

    print(f"Сравнение сохранено в:\n  CSV: {comparison_csv}\n  TXT: {stats_txt}\n  Папка с картинками: {plots_dir}")


if __name__ == "__main__":
    main()

