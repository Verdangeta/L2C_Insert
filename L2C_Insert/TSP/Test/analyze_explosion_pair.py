import argparse
import csv
import json
import os
from typing import Dict, List, Optional, Tuple

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


def compute_gap_to_concorde_comparison(
    baseline: Dict[Tuple[str, str], Dict[str, str]],
    advanced: Dict[Tuple[str, str], Dict[str, str]],
) -> List[Dict[str, str]]:
    """
    Pair baseline/advanced by instance and compute gap to Concorde reference.

    gap_to_concorde = (tour_length - reference_cost) / reference_cost * 100
    Lower is better.
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

        # Prefer baseline reference; fallback to advanced if needed.
        ref_raw = b.get("reference_cost", "")
        if not str(ref_raw).strip():
            ref_raw = a.get("reference_cost", "")
        try:
            ref = float(ref_raw)
        except (TypeError, ValueError):
            continue

        if not np.isfinite(b_len) or not np.isfinite(a_len) or not np.isfinite(ref) or ref <= 0:
            continue

        b_gap = (b_len - ref) / ref * 100.0
        a_gap = (a_len - ref) / ref * 100.0
        delta_gap = b_gap - a_gap  # >0 means advanced is better (closer to Concorde)

        rows.append(
            {
                "instance_id": key[0],
                "problem_size": key[1],
                "reference_cost": f"{ref:.6f}",
                "tour_length_baseline": f"{b_len:.6f}",
                "tour_length_advanced": f"{a_len:.6f}",
                "gap_baseline_pct": f"{b_gap:.6f}",
                "gap_advanced_pct": f"{a_gap:.6f}",
                "delta_gap_pct_points": f"{delta_gap:.6f}",
            }
        )

    # Add per-size difficulty annotations for downstream research:
    # - baseline_gap_quartile: Q1..Q4 by baseline gap within the same problem size
    # - is_hard_instance: 1 if baseline gap >= median baseline gap for that size
    by_size: Dict[str, List[float]] = {}
    for row in rows:
        size = row["problem_size"]
        by_size.setdefault(size, []).append(float(row["gap_baseline_pct"]))

    thresholds: Dict[str, Dict[str, float]] = {}
    for size, values in by_size.items():
        arr = np.asarray(values, dtype=float)
        thresholds[size] = {
            "q25": float(np.quantile(arr, 0.25)),
            "q50": float(np.quantile(arr, 0.50)),
            "q75": float(np.quantile(arr, 0.75)),
        }

    for row in rows:
        size = row["problem_size"]
        b_gap = float(row["gap_baseline_pct"])
        t = thresholds[size]
        if b_gap <= t["q25"]:
            quartile = "Q1"
        elif b_gap <= t["q50"]:
            quartile = "Q2"
        elif b_gap <= t["q75"]:
            quartile = "Q3"
        else:
            quartile = "Q4"
        row["baseline_gap_quartile"] = quartile
        row["is_hard_instance"] = "1" if b_gap >= t["q50"] else "0"

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


def aggregate_gap_by_size(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
    by_size: Dict[str, Dict[str, List[float]]] = {}
    for r in rows:
        size = r["problem_size"]
        b_gap = float(r["gap_baseline_pct"])
        a_gap = float(r["gap_advanced_pct"])
        d_gap = float(r["delta_gap_pct_points"])

        if size not in by_size:
            by_size[size] = {
                "b_gap": [],
                "a_gap": [],
                "d_gap": [],
                "adv_better": [],
            }

        by_size[size]["b_gap"].append(b_gap)
        by_size[size]["a_gap"].append(a_gap)
        by_size[size]["d_gap"].append(d_gap)
        by_size[size]["adv_better"].append(1.0 if d_gap > 0 else 0.0)

    stats: Dict[str, Dict[str, float]] = {}
    for size, vals in by_size.items():
        b_arr = np.asarray(vals["b_gap"], dtype=float)
        a_arr = np.asarray(vals["a_gap"], dtype=float)
        d_arr = np.asarray(vals["d_gap"], dtype=float)
        w_arr = np.asarray(vals["adv_better"], dtype=float)

        stats[size] = {
            "count": float(len(b_arr)),
            "baseline_gap_mean": float(b_arr.mean()) if b_arr.size else float("nan"),
            "advanced_gap_mean": float(a_arr.mean()) if a_arr.size else float("nan"),
            "delta_gap_mean": float(d_arr.mean()) if d_arr.size else float("nan"),
            "delta_gap_min": float(d_arr.min()) if d_arr.size else float("nan"),
            "delta_gap_max": float(d_arr.max()) if d_arr.size else float("nan"),
            "adv_win_rate": float(w_arr.mean() * 100.0) if w_arr.size else float("nan"),
        }
    return stats


def load_rrc_step_logs(run_dir: str) -> List[Dict[str, float]]:
    """
    Load all RRC per-step logs from run_dir/rrc_logs.
    Each row contains parsed numeric fields for further aggregation.
    """
    logs_dir = os.path.join(run_dir, "rrc_logs")
    if not os.path.isdir(logs_dir):
        return []

    rows: List[Dict[str, float]] = []
    for fname in os.listdir(logs_dir):
        if not fname.startswith("rrc_steps_") or not fname.endswith(".csv"):
            continue
        fpath = os.path.join(logs_dir, fname)
        try:
            with open(fpath, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        before_length = float(row.get("before_length", "nan"))
                        after_length = float(row.get("after_length", "nan"))
                        abs_delta = float(row.get("abs_delta", "nan"))
                        rel_delta = float(row.get("rel_delta", "nan"))
                    except ValueError:
                        continue
                    if not (
                        np.isfinite(before_length)
                        and np.isfinite(after_length)
                        and np.isfinite(abs_delta)
                        and np.isfinite(rel_delta)
                    ):
                        continue
                    try:
                        improved = int(float(row.get("improved", "0")))
                    except ValueError:
                        improved = 0
                    try:
                        step = int(float(row.get("step", "0")))
                    except ValueError:
                        step = 0
                    try:
                        problem_size = int(float(row.get("problem_size", "0")))
                    except ValueError:
                        problem_size = 0

                    rows.append(
                        {
                            "before_length": before_length,
                            "after_length": after_length,
                            "abs_delta": abs_delta,
                            "rel_delta": rel_delta,
                            "improved": improved,
                            "step": step,
                            "problem_size": problem_size,
                        }
                    )
        except Exception:
            continue

    return rows


def aggregate_rrc_stats(rows: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate step-level RRC statistics over all steps for one method.
    """
    if not rows:
        return {}

    abs_arr = np.asarray([r["abs_delta"] for r in rows], dtype=float)
    rel_arr = np.asarray([r["rel_delta"] for r in rows], dtype=float)
    improved_mask = np.asarray([bool(r["improved"]) for r in rows], dtype=bool)
    worsened_mask = ~improved_mask

    stats: Dict[str, float] = {}
    stats["count_steps"] = float(len(abs_arr))
    stats["mean_abs_delta"] = float(abs_arr.mean()) if abs_arr.size else float("nan")
    stats["mean_rel_delta_pct"] = float(rel_arr.mean() * 100.0) if rel_arr.size else float("nan")

    if improved_mask.any():
        stats["mean_abs_delta_improved"] = float(abs_arr[improved_mask].mean())
        stats["mean_rel_delta_improved_pct"] = float(rel_arr[improved_mask].mean() * 100.0)
    else:
        stats["mean_abs_delta_improved"] = float("nan")
        stats["mean_rel_delta_improved_pct"] = float("nan")

    if worsened_mask.any():
        stats["mean_abs_delta_worsened"] = float(abs_arr[worsened_mask].mean())
        stats["mean_rel_delta_worsened_pct"] = float(rel_arr[worsened_mask].mean() * 100.0)
    else:
        stats["mean_abs_delta_worsened"] = float("nan")
        stats["mean_rel_delta_worsened_pct"] = float("nan")

    stats["success_rate_pct"] = float(improved_mask.mean() * 100.0) if abs_arr.size else float("nan")
    return stats


def save_rrc_stats_txt(
    baseline_rows: List[Dict[str, float]],
    advanced_rows: List[Dict[str, float]],
    out_path: str,
) -> None:
    """
    Save human-readable RRC step-level statistics for baseline vs advanced methods.
    """
    has_baseline = bool(baseline_rows)
    has_advanced = bool(advanced_rows)

    lines: List[str] = []
    lines.append("=" * 100)
    lines.append("RRC STEP-LEVEL STATISTICS (per-destroy/repair iteration)")
    lines.append("=" * 100)

    if not has_baseline and not has_advanced:
        lines.append("Нет доступных RRC-логов для baseline или advanced запусков.")
    else:
        stats_baseline = aggregate_rrc_stats(baseline_rows) if has_baseline else {}
        stats_advanced = aggregate_rrc_stats(advanced_rows) if has_advanced else {}

        if has_baseline:
            s = stats_baseline
            lines.append("\nBaseline (обычный выбор области перестройки):")
            lines.append(f"  Кол-во шагов: {int(s.get('count_steps', 0.0))}")
            lines.append(
                f"  Среднее abs_delta по всем шагам: {s.get('mean_abs_delta', float('nan')):.6f}"
            )
            lines.append(
                f"  Среднее abs_delta по удачным шагам (improved=1): {s.get('mean_abs_delta_improved', float('nan')):.6f}"
            )
            lines.append(
                f"  Среднее abs_delta по неудачным шагам (improved=0): {s.get('mean_abs_delta_worsened', float('nan')):.6f}"
            )
            lines.append(
                f"  Средний rel_delta по всем шагам (%): {s.get('mean_rel_delta_pct', float('nan')):.6f}"
            )
            lines.append(
                "  Средний rel_delta по удачным шагам (%): "
                f"{s.get('mean_rel_delta_improved_pct', float('nan')):.6f}"
            )
            lines.append(
                "  Средний rel_delta по неудачным шагам (%): "
                f"{s.get('mean_rel_delta_worsened_pct', float('nan')):.6f}"
            )
            lines.append(
                f"  Процент удачных улучшений: {s.get('success_rate_pct', float('nan')):.2f}%"
            )

        if has_advanced:
            s = stats_advanced
            lines.append("\nAdvanced sampling (RTDL-ориентированный выбор области):")
            lines.append(f"  Кол-во шагов: {int(s.get('count_steps', 0.0))}")
            lines.append(
                f"  Среднее abs_delta по всем шагам: {s.get('mean_abs_delta', float('nan')):.6f}"
            )
            lines.append(
                f"  Среднее abs_delta по удачным шагам (improved=1): {s.get('mean_abs_delta_improved', float('nan')):.6f}"
            )
            lines.append(
                f"  Среднее abs_delta по неудачным шагам (improved=0): {s.get('mean_abs_delta_worsened', float('nan')):.6f}"
            )
            lines.append(
                f"  Средний rel_delta по всем шагам (%): {s.get('mean_rel_delta_pct', float('nan')):.6f}"
            )
            lines.append(
                "  Средний rel_delta по удачным шагам (%): "
                f"{s.get('mean_rel_delta_improved_pct', float('nan')):.6f}"
            )
            lines.append(
                "  Средний rel_delta по неудачным шагам (%): "
                f"{s.get('mean_rel_delta_worsened_pct', float('nan')):.6f}"
            )
            lines.append(
                f"  Процент удачных улучшений: {s.get('success_rate_pct', float('nan')):.2f}%"
            )

        if has_baseline and has_advanced:
            b = stats_baseline
            a = stats_advanced
            lines.append("\nРазница методов (advanced - baseline):")
            def _delta(key: str) -> float:
                return float(a.get(key, float("nan"))) - float(b.get(key, float("nan")))

            lines.append(
                f"  Δ mean_abs_delta (все шаги): {_delta('mean_abs_delta'):.6f}"
            )
            lines.append(
                "  Δ mean_abs_delta_improved (удачные шаги): "
                f"{_delta('mean_abs_delta_improved'):.6f}"
            )
            lines.append(
                "  Δ mean_abs_delta_worsened (неудачные шаги): "
                f"{_delta('mean_abs_delta_worsened'):.6f}"
            )
            lines.append(
                "  Δ mean_rel_delta_pct (все шаги, п.п.): "
                f"{_delta('mean_rel_delta_pct'):.6f}"
            )
            lines.append(
                "  Δ mean_rel_delta_improved_pct (удачные шаги, п.п.): "
                f"{_delta('mean_rel_delta_improved_pct'):.6f}"
            )
            lines.append(
                "  Δ mean_rel_delta_worsened_pct (неудачные шаги, п.п.): "
                f"{_delta('mean_rel_delta_worsened_pct'):.6f}"
            )
            lines.append(
                "  Δ success_rate_pct (разница в проценте удачных шагов, п.п.): "
                f"{_delta('success_rate_pct'):.2f}"
            )

    lines.append("=" * 100)

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


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


def save_gap_comparison_csv(rows: List[Dict[str, str]], out_path: str) -> None:
    if not rows:
        return
    fieldnames = [
        "instance_id",
        "problem_size",
        "reference_cost",
        "tour_length_baseline",
        "tour_length_advanced",
        "gap_baseline_pct",
        "gap_advanced_pct",
        "delta_gap_pct_points",
        "baseline_gap_quartile",
        "is_hard_instance",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def save_stats_txt(
    stats: Dict[str, Dict[str, float]],
    out_path: str,
    context_lines: Optional[List[str]] = None,
) -> None:
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("COMPARISON STATISTICS (baseline vs advanced_sampling)")
    lines.append("=" * 80)
    if context_lines:
        lines.extend(context_lines)
        lines.append("-" * 80)
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


def save_gap_stats_txt(
    stats: Dict[str, Dict[str, float]],
    out_path: str,
    context_lines: Optional[List[str]] = None,
) -> None:
    lines: List[str] = []
    lines.append("=" * 100)
    lines.append("GAP TO CONCORDE BY PROBLEM SIZE (lower gap is better)")
    lines.append("=" * 100)
    if context_lines:
        lines.extend(context_lines)
        lines.append("-" * 100)
    if not stats:
        lines.append("No overlapping instances with valid reference_cost.")
    else:
        lines.append(
            f"{'Size':<8} {'Count':<8} "
            f"{'Baseline avg gap %':<20} {'Advanced avg gap %':<20} "
            f"{'Delta (pp)':<14} {'Delta min':<12} {'Delta max':<12} {'Adv win rate %':<15}"
        )
        lines.append("-" * 100)
        for size in sorted(stats.keys(), key=lambda x: int(x)):
            s = stats[size]
            lines.append(
                f"{size:<8} {int(s['count']):<8d} "
                f"{s['baseline_gap_mean']:<20.6f} {s['advanced_gap_mean']:<20.6f} "
                f"{s['delta_gap_mean']:<14.6f} {s['delta_gap_min']:<12.6f} {s['delta_gap_max']:<12.6f} {s['adv_win_rate']:<15.2f}"
            )
        lines.append("-" * 100)
        lines.append("Delta (pp) = baseline_gap - advanced_gap. Positive => advanced is closer to Concorde.")
    lines.append("=" * 100)

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
    parser.add_argument(
        "--baseline_fingerprint",
        type=str,
        default=None,
        help="Optional fingerprint/id of the baseline cache used for this comparison.",
    )
    parser.add_argument(
        "--experiment_params_json",
        type=str,
        default=None,
        help="Optional JSON string with experiment params (topk/temp/etc.).",
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
    baseline_ref_json = os.path.join(out_dir, "baseline_ref.json")
    experiment_params_json = os.path.join(out_dir, "experiment_params.json")

    baseline = load_results(baseline_csv)
    advanced = load_results(advanced_csv)
    rows = compute_comparison(baseline, advanced)
    gap_rows = compute_gap_to_concorde_comparison(baseline, advanced)

    if not rows:
        print("No common instances between baseline and advanced runs.")
        return

    comparison_csv = os.path.join(out_dir, "comparison_baseline_vs_advanced.csv")
    stats_txt = os.path.join(out_dir, "comparison_stats.txt")
    gap_comparison_csv = os.path.join(out_dir, "comparison_gap_to_concorde.csv")
    gap_stats_txt = os.path.join(out_dir, "comparison_gap_to_concorde.txt")
    rrc_stats_txt = os.path.join(out_dir, "rrc_step_stats.txt")
    plots_dir = os.path.join(out_dir, "combined_plots")

    parsed_params = None
    context_lines: List[str] = []
    if args.experiment_params_json:
        try:
            parsed_params = json.loads(args.experiment_params_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid --experiment_params_json: {e}") from e
        sampling_variant = parsed_params.get("sampling_variant", "N/A")
        context_lines.append(f"sampling_variant: {sampling_variant}")
        for key in (
            "sampling_window",
            "temperature",
            "topk_frac",
            "topk_min",
            "multi_local_k_min",
            "multi_local_k_max",
        ):
            if key in parsed_params:
                context_lines.append(f"{key}: {parsed_params[key]}")

    save_comparison_csv(rows, comparison_csv)
    stats = aggregate_by_size(rows)
    save_stats_txt(stats, stats_txt, context_lines=context_lines or None)
    save_gap_comparison_csv(gap_rows, gap_comparison_csv)
    gap_stats = aggregate_gap_by_size(gap_rows)
    save_gap_stats_txt(gap_stats, gap_stats_txt, context_lines=context_lines or None)

    # RRC step-level statistics, если есть соответствующие логи.
    baseline_rrc_rows = load_rrc_step_logs(args.baseline_dir)
    advanced_rrc_rows = load_rrc_step_logs(args.advanced_dir)
    if baseline_rrc_rows or advanced_rrc_rows:
        save_rrc_stats_txt(baseline_rrc_rows, advanced_rrc_rows, rrc_stats_txt)

    combine_plots(args.baseline_dir, args.advanced_dir, plots_dir)

    baseline_ref_payload = {
        "baseline_dir": os.path.abspath(args.baseline_dir),
        "advanced_dir": os.path.abspath(args.advanced_dir),
    }
    if args.baseline_fingerprint:
        baseline_ref_payload["baseline_fingerprint"] = args.baseline_fingerprint
    with open(baseline_ref_json, "w") as f:
        json.dump(baseline_ref_payload, f, indent=2)

    if args.experiment_params_json:
        assert parsed_params is not None
        with open(experiment_params_json, "w") as f:
            json.dump(parsed_params, f, indent=2)

    print(
        "Сравнение сохранено в:\n"
        f"  CSV (length): {comparison_csv}\n"
        f"  TXT (length): {stats_txt}\n"
        f"  CSV (gap to Concorde): {gap_comparison_csv}\n"
        f"  TXT (gap to Concorde): {gap_stats_txt}\n"
        f"  TXT (RRC step stats): {rrc_stats_txt if (baseline_rrc_rows or advanced_rrc_rows) else 'нет RRC-логов'}\n"
        f"  baseline_ref.json: {baseline_ref_json}\n"
        f"  experiment_params.json: {experiment_params_json if args.experiment_params_json else 'не передан'}\n"
        f"  Папка с картинками: {plots_dir}"
    )


if __name__ == "__main__":
    main()

