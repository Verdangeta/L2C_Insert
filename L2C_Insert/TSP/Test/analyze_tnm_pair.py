#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, List

import numpy as np


def _load_results(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return data


def _index_by_instance(rows: List[Dict]) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for row in rows:
        instance = row.get("instance")
        if isinstance(instance, str):
            out[instance] = row
    return out


def _mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare TNM baseline vs advanced results")
    parser.add_argument("--baseline_json", required=True, type=str)
    parser.add_argument("--advanced_json", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    args = parser.parse_args()

    baseline = _index_by_instance(_load_results(args.baseline_json))
    advanced = _index_by_instance(_load_results(args.advanced_json))
    shared = sorted(set(baseline.keys()) & set(advanced.keys()))

    baseline_gaps = []
    advanced_gaps = []
    delta_gaps = []
    improved = 0
    worsened = 0
    equal = 0

    for name in shared:
        base_gap = baseline[name].get("gap_percent")
        adv_gap = advanced[name].get("gap_percent")
        if base_gap is None or adv_gap is None:
            continue
        base_gap = float(base_gap)
        adv_gap = float(adv_gap)
        delta = adv_gap - base_gap
        baseline_gaps.append(base_gap)
        advanced_gaps.append(adv_gap)
        delta_gaps.append(delta)
        if delta < 0:
            improved += 1
        elif delta > 0:
            worsened += 1
        else:
            equal += 1

    lines = []
    lines.append("TNM baseline vs advanced comparison")
    lines.append("=" * 80)
    lines.append(f"Baseline JSON: {os.path.abspath(args.baseline_json)}")
    lines.append(f"Advanced JSON: {os.path.abspath(args.advanced_json)}")
    lines.append(f"Shared instances: {len(shared)}")
    lines.append("")

    if delta_gaps:
        lines.append(f"Average baseline gap (%): {_mean(baseline_gaps):.6f}")
        lines.append(f"Average advanced gap (%): {_mean(advanced_gaps):.6f}")
        lines.append(f"Average delta advanced-baseline (%): {_mean(delta_gaps):+.6f}")
        lines.append(f"Median delta advanced-baseline (%): {float(np.median(delta_gaps)):+.6f}")
        lines.append(f"Min/Max delta (%): {float(np.min(delta_gaps)):+.6f} / {float(np.max(delta_gaps)):+.6f}")
        lines.append(f"Improved/Worsened/Equal: {improved}/{worsened}/{equal}")
    else:
        lines.append("No comparable rows with non-null gap_percent were found.")

    output_path = os.path.abspath(args.output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved report to: {output_path}")


if __name__ == "__main__":
    main()
