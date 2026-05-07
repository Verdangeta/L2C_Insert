#!/usr/bin/env python3
import argparse
import csv
import hashlib
import itertools
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Dict, List, Optional

# Baseline (Proximity, no RTDL sampling) is built once into result/baselines/<fp>.
# Do not duplicate it as a sweep variant — compare RTDL runs against that baseline.
VALID_SAMPLING_VARIANTS = ("single", "multi", "multi_edge", "rtdl_edge")
DEFAULT_SAMPLING_VARIANT = "single"
DEFAULT_MULTI_LOCAL_K_MIN = 4
DEFAULT_MULTI_LOCAL_K_MAX = 20


@dataclass
class ExperimentConfig:
    model_path: str
    problem_sizes: str
    num_instances: int
    layout: str
    num_centers: int
    range_min: float
    range_max: float
    rate: float
    pool_root: str
    pool_name: str
    baseline_model_name: str
    adv_model_name: str
    cuda_device_num: int
    rrc_budget: int
    rrc_range: int
    temperature: float
    topk_frac: float
    topk_min: int
    sampling_window: int
    cluster_score_reduction: str
    cluster_boundary_gap: float = 0.005


PRESETS: Dict[str, ExperimentConfig] = {
    "explosion_2k_default": ExperimentConfig(
        model_path="/trinity/home/alexander.mironenko/TDA_tsp/L2C_Insert/L2C_Insert/TSP/Test/result/pretrain/tsp_model.pt",
        problem_sizes="2000",
        num_instances=15,
        layout="explosion",
        num_centers=6,
        range_min=0.1,
        range_max=0.5,
        rate=10.0,
        pool_root="./shared_instances",
        pool_name="",
        baseline_model_name="baseline",
        adv_model_name="advance_sampling_baseline",
        cuda_device_num=0,
        rrc_budget=1000,
        rrc_range=300,
        temperature=1.0,
        topk_frac=0.05,
        topk_min=20,
        sampling_window=0,
        cluster_score_reduction="sum",
    ),
    "explosion_500_default": ExperimentConfig(
        model_path="/trinity/home/alexander.mironenko/TDA_tsp/L2C_Insert/L2C_Insert/TSP/Test/result/pretrain/tsp_model.pt",
        problem_sizes="500",
        num_instances=50,
        layout="explosion",
        num_centers=6,
        range_min=0.1,
        range_max=0.5,
        rate=10.0,
        pool_root="./shared_instances",
        pool_name="",
        baseline_model_name="baseline",
        adv_model_name="advance_sampling_baseline",
        cuda_device_num=0,
        rrc_budget=100,
        rrc_range=200,
        temperature=1.0,
        topk_frac=1.0,
        topk_min=20,
        sampling_window=0,
        cluster_score_reduction="mean",
    ),
        "explosion_1000_default": ExperimentConfig(
        model_path="/trinity/home/alexander.mironenko/TDA_tsp/L2C_Insert/L2C_Insert/TSP/Test/result/pretrain/tsp_model.pt",
        problem_sizes="1000",
        num_instances=30,
        layout="explosion",
        num_centers=6,
        range_min=0.1,
        range_max=0.5,
        rate=10.0,
        pool_root="./shared_instances",
        pool_name="",
        baseline_model_name="baseline",
        adv_model_name="advance_sampling_baseline",
        cuda_device_num=0,
        rrc_budget=1000,
        rrc_range=300,
        temperature=1.0,
        topk_frac=1.0,
        topk_min=20,
        sampling_window=0,
        cluster_score_reduction="sum",
    ),
    "explosion_500_temperature_04": ExperimentConfig(
        model_path="/trinity/home/alexander.mironenko/TDA_tsp/L2C_Insert/L2C_Insert/TSP/Test/result/pretrain/tsp_model.pt",
        problem_sizes="500",
        num_instances=50,
        layout="explosion",
        num_centers=6,
        range_min=0.1,
        range_max=0.5,
        rate=10.0,
        pool_root="./shared_instances",
        pool_name="",
        baseline_model_name="baseline",
        adv_model_name="advance_sampling_baseline",
        cuda_device_num=0,
        rrc_budget=100,
        rrc_range=50,
        temperature=0.4,
        topk_frac=1.0,
        topk_min=20,
        sampling_window=0,
        cluster_score_reduction="sum",
    ),
    "explosion_500_topk_80": ExperimentConfig(
        model_path="/trinity/home/alexander.mironenko/TDA_tsp/L2C_Insert/L2C_Insert/TSP/Test/result/pretrain/tsp_model.pt",
        problem_sizes="500",
        num_instances=50,
        layout="explosion",
        num_centers=6,
        range_min=0.1,
        range_max=0.5,
        rate=10.0,
        pool_root="./shared_instances",
        pool_name="",
        baseline_model_name="baseline",
        adv_model_name="advance_sampling_baseline",
        cuda_device_num=0,
        rrc_budget=100,
        rrc_range=50,
        temperature=1.0,
        topk_frac=0.8,
        topk_min=20,
        sampling_window=0,
        cluster_score_reduction="sum",
    ),
    "explosion_500_topk_80_temperature_04": ExperimentConfig(
        model_path="/trinity/home/alexander.mironenko/TDA_tsp/L2C_Insert/L2C_Insert/TSP/Test/result/pretrain/tsp_model.pt",
        problem_sizes="500",
        num_instances=50,
        layout="explosion",
        num_centers=6,
        range_min=0.1,
        range_max=0.5,
        rate=10.0,
        pool_root="./shared_instances",
        pool_name="",
        baseline_model_name="baseline",
        adv_model_name="advance_sampling_baseline",
        cuda_device_num=0,
        rrc_budget=100,
        rrc_range=50,
        temperature=0.4,
        topk_frac=0.8,
        topk_min=20,
        sampling_window=0,
        cluster_score_reduction="sum",
    ),
    "explosion_2k_more_instances": ExperimentConfig(
        model_path="/trinity/home/alexander.mironenko/TDA_tsp/L2C_Insert/L2C_Insert/TSP/Test/result/pretrain/tsp_model.pt",
        problem_sizes="2000",
        num_instances=50,
        layout="explosion",
        num_centers=6,
        range_min=0.1,
        range_max=0.5,
        rate=10.0,
        pool_root="./shared_instances",
        pool_name="",
        baseline_model_name="baseline_50inst",
        adv_model_name="advance_sampling_50inst",
        cuda_device_num=0,
        rrc_budget=1000,
        rrc_range=300,
        temperature=1.0,
        topk_frac=0.05,
        topk_min=20,
        sampling_window=0,
        cluster_score_reduction="sum",
    ),
    "explosion_2k_wider_clusters": ExperimentConfig(
        model_path="/trinity/home/alexander.mironenko/TDA_tsp/L2C_Insert/L2C_Insert/TSP/Test/result/pretrain/tsp_model.pt",
        problem_sizes="2000",
        num_instances=15,
        layout="explosion",
        num_centers=8,
        range_min=0.05,
        range_max=0.7,
        rate=12.0,
        pool_root="./shared_instances",
        pool_name="",
        baseline_model_name="baseline_wide",
        adv_model_name="advance_sampling_wide",
        cuda_device_num=0,
        rrc_budget=1000,
        rrc_range=300,
        temperature=1.0,
        topk_frac=0.05,
        topk_min=20,
        sampling_window=0,
        cluster_score_reduction="sum",
    ),
    "custom_manual": ExperimentConfig(
        model_path="/trinity/home/alexander.mironenko/TDA_tsp/L2C_Insert/L2C_Insert/TSP/Test/result/pretrain/tsp_model.pt",
        problem_sizes="2000",
        num_instances=15,
        layout="explosion",
        num_centers=6,
        range_min=0.1,
        range_max=0.5,
        rate=10.0,
        pool_root="./shared_instances",
        pool_name="",
        baseline_model_name="baseline_custom",
        adv_model_name="advance_sampling_custom",
        cuda_device_num=0,
        rrc_budget=1000,
        rrc_range=300,
        temperature=1.0,
        topk_frac=0.05,
        topk_min=20,
        sampling_window=0,
        cluster_score_reduction="sum",
    ),
}


def _register_layout_presets() -> None:
    """Auto-register layout variants so new layouts can reuse the same matrix."""
    generated: Dict[str, ExperimentConfig] = {}
    for preset_name, preset_cfg in list(PRESETS.items()):
        if not preset_name.startswith("explosion_"):
            continue
        implosion_name = preset_name.replace("explosion_", "implosion_", 1)
        if implosion_name in PRESETS:
            continue
        generated[implosion_name] = replace(
            preset_cfg,
            layout="implosion",
            rate=0.0,  # implosion generator ignores exponential explosion rate
        )
    PRESETS.update(generated)


_register_layout_presets()


def _float_slug(value: float) -> str:
    s = f"{value:.6g}"
    return s.replace("-", "m").replace(".", "p")


def _sanitize_slug(value: str) -> str:
    safe = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def _build_sweep_dirname(config_name: str, cfg: ExperimentConfig, timestamp: str) -> str:
    config_slug = _sanitize_slug(config_name)
    if (
        str(config_name).startswith(("explosion_", "implosion_"))
        and not str(config_name).startswith(f"{cfg.layout}_")
    ):
        # If layout is overridden via --set, avoid misleading preset naming in paths.
        config_slug = "custom"
    layout_slug = _sanitize_slug(str(cfg.layout))
    size_slug = _sanitize_slug(str(cfg.problem_sizes).replace(",", "-"))
    return f"sweep_{layout_slug}_{config_slug}_s{size_slug}_n{cfg.num_instances}_{timestamp}"


def _parse_kv_list(values: Optional[List[str]]) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    if not values:
        return parsed
    for token in values:
        if "=" not in token:
            raise ValueError(f"Expected key=value, got: {token}")
        key, raw = token.split("=", 1)
        key = key.strip()
        raw = raw.strip()
        if not key:
            raise ValueError(f"Invalid key in token: {token}")
        parsed[key] = raw
    return parsed


def _parse_sweep_expr(expr: str) -> List[Dict[str, str]]:
    tokens = [t.strip() for t in expr.split() if t.strip()]
    if not tokens:
        raise ValueError("Sweep expression is empty")
    keys: List[str] = []
    values_grid: List[List[str]] = []
    for token in tokens:
        if ":" not in token:
            raise ValueError(f"Invalid sweep token (expected key:v1,v2): {token}")
        key, raw_vals = token.split(":", 1)
        vals = [v.strip() for v in raw_vals.split(",") if v.strip()]
        if not vals:
            raise ValueError(f"No values provided for sweep key: {key}")
        keys.append(key.strip())
        values_grid.append(vals)

    variants: List[Dict[str, str]] = []
    for combo in itertools.product(*values_grid):
        variant = {k: v for k, v in zip(keys, combo)}
        variants.append(variant)
    return variants


def _apply_overrides(cfg: ExperimentConfig, overrides: Dict[str, str]) -> ExperimentConfig:
    merged = ExperimentConfig(**cfg.__dict__)
    for key, raw in overrides.items():
        if not hasattr(merged, key):
            raise ValueError(f"Unknown override key: {key}")
        current = getattr(merged, key)
        if isinstance(current, int):
            setattr(merged, key, int(raw))
        elif isinstance(current, float):
            setattr(merged, key, float(raw))
        else:
            setattr(merged, key, raw)
    return merged


def _build_common_args(cfg: ExperimentConfig) -> List[str]:
    args = [
        "--model_path",
        cfg.model_path,
        "--problem_sizes",
        cfg.problem_sizes,
        "--num_instances",
        str(cfg.num_instances),
        "--layout",
        cfg.layout,
        "--num_centers",
        str(cfg.num_centers),
        "--range_min",
        str(cfg.range_min),
        "--range_max",
        str(cfg.range_max),
        "--rate",
        str(cfg.rate),
        "--cuda_device_num",
        str(cfg.cuda_device_num),
        "--RRC_budget",
        str(cfg.rrc_budget),
        "--RRC_range",
        str(cfg.rrc_range),
        "--instances_pool_root",
        cfg.pool_root,
        "--optimal_cost_method",
        "concorde",
    ]
    if cfg.pool_name:
        args += ["--instances_pool_name", cfg.pool_name]
    if str(cfg.layout) == "clustered":
        args += ["--cluster_boundary_gap", str(cfg.cluster_boundary_gap)]
    return args


def _baseline_fingerprint_payload(cfg: ExperimentConfig) -> Dict[str, str]:
    payload = {
        "model_path": cfg.model_path,
        "problem_sizes": cfg.problem_sizes,
        "num_instances": str(cfg.num_instances),
        "layout": cfg.layout,
        "num_centers": str(cfg.num_centers),
        "range_min": str(cfg.range_min),
        "range_max": str(cfg.range_max),
        "rate": str(cfg.rate),
        "rrc_budget": str(cfg.rrc_budget),
        "rrc_range": str(cfg.rrc_range),
        "pool_root": os.path.abspath(cfg.pool_root),
        "pool_name": cfg.pool_name or "__auto__",
    }
    if str(cfg.layout) == "clustered":
        payload["cluster_boundary_gap"] = str(cfg.cluster_boundary_gap)
    return payload


def _baseline_fingerprint(cfg: ExperimentConfig) -> str:
    payload = _baseline_fingerprint_payload(cfg)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:12]


def _run_cmd(command: List[str], cwd: str) -> None:
    printable = " ".join(shlex_quote(x) for x in command)
    print(f"[CMD] {printable}")
    proc = subprocess.run(command, cwd=cwd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (code={proc.returncode}): {printable}")


def shlex_quote(value: str) -> str:
    if value == "":
        return "''"
    if all(ch.isalnum() or ch in ("@", "%", "+", "=", ":", ",", ".", "/", "-", "_") for ch in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _read_gap_metrics(compare_csv: str) -> Dict[str, float]:
    if not os.path.isfile(compare_csv):
        return {}
    deltas: List[float] = []
    wins = 0
    with open(compare_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                delta = float(row.get("delta_gap_pct_points", "nan"))
            except ValueError:
                continue
            if delta != delta:
                continue
            deltas.append(delta)
            if delta > 0:
                wins += 1
    if not deltas:
        return {}
    return {
        "count": float(len(deltas)),
        "mean_delta_pp": float(sum(deltas) / len(deltas)),
        "win_rate_pct": float(wins / len(deltas) * 100.0),
    }


def _build_variant_slug(variant: Dict[str, str]) -> str:
    sampling_variant = str(variant.get("sampling_variant", DEFAULT_SAMPLING_VARIANT))
    topk = float(variant.get("topk_frac", 0.05))
    temp = float(variant.get("temperature", 1.0))
    reduction = str(variant.get("cluster_score_reduction", "sum"))
    window = int(variant.get("sampling_window", 0))
    topk_min = int(variant.get("topk_min", 20))
    base_slug = _sanitize_slug(
        f"variant{sampling_variant}_"
        f"topk{_float_slug(topk)}_temp{_float_slug(temp)}"
        f"_cluster{reduction}_w{window}_topkmin{topk_min}"
    )
    if sampling_variant not in ("multi", "multi_edge", "rtdl_edge"):
        return base_slug
    local_k_min = int(variant.get("multi_local_k_min", DEFAULT_MULTI_LOCAL_K_MIN))
    local_k_max = int(variant.get("multi_local_k_max", DEFAULT_MULTI_LOCAL_K_MAX))
    edge_selection = str(variant.get("edge_selection", "softmax")).lower()
    if sampling_variant == "rtdl_edge":
        suffix = f"_edge{edge_selection}"
        ess_ratio = float(variant.get("edge_target_ess_ratio", -1.0))
        if ess_ratio > 0:
            suffix += f"_ess{_float_slug(ess_ratio)}"
        forbid_removed = int(variant.get("forbid_removed_edges", 0))
        suffix += f"_noaddrem{forbid_removed}"
    else:
        suffix = f"_k{local_k_min}-{local_k_max}"
        if sampling_variant == "multi_edge":
            suffix += f"_edge{edge_selection}"
            forbid_removed = int(variant.get("forbid_removed_edges", 0))
            suffix += f"_noaddrem{forbid_removed}"
    return _sanitize_slug(f"{base_slug}{suffix}")


def _normalize_sampling_variant(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized == "advanced":
        normalized = "single"
    if normalized in ("proximity", "baseline"):
        raise ValueError(
            "sampling_variant=proximity/baseline is not supported in sweep: "
            "baseline (Proximity destroy) is already computed once and stored under "
            "result/baselines/<fingerprint>. Use sampling_variant:single,multi and "
            "compare against that baseline."
        )
    if normalized not in VALID_SAMPLING_VARIANTS:
        raise ValueError(
            f"Invalid sampling_variant={value!r}. "
            f"Expected one of: {', '.join(VALID_SAMPLING_VARIANTS)}"
        )
    return normalized


def _write_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run baseline/advanced layout experiments with baseline cache and sweep support."
    )
    parser.add_argument("--config", type=str, default="explosion_2k_default", choices=sorted(PRESETS.keys()))
    parser.add_argument("--regenerate", type=int, default=0, help="Regenerate shared instances for baseline run")
    parser.add_argument("--baseline-only", action="store_true", help="Run/check baseline only and exit")
    parser.add_argument("--baseline-dir", type=str, default=None, help="Use external baseline directory instead of cache")
    parser.add_argument("--adv", nargs="*", default=None, help="Single advanced variant as key=value tokens")
    parser.add_argument(
        "--sweep",
        type=str,
        default=None,
        help=(
            'Grid sweep expression, e.g. '
            '"sampling_variant:single,multi topk_frac:0.05,0.8 temperature:0.4,1.0". '
            "Baseline (Proximity) is not a sweep variant — it is built once as result/baselines/<fp>."
        ),
    )
    parser.add_argument("--python-bin", type=str, default=sys.executable, help="Python executable for child scripts")
    parser.add_argument("--baselines-root", type=str, default="./result/baselines")
    parser.add_argument("--experiments-root", type=str, default="./result/experiments")
    parser.add_argument(
        "--set",
        nargs="*",
        default=None,
        help="Override config values using key=value (e.g. --set problem_sizes=500 num_instances=50)",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = PRESETS[args.config]
    base_overrides = _parse_kv_list(args.set)
    cfg = _apply_overrides(cfg, base_overrides)
    cfg.pool_root = os.path.abspath(os.path.join(script_dir, cfg.pool_root))

    baselines_root = os.path.abspath(os.path.join(script_dir, args.baselines_root))
    experiments_root = os.path.abspath(os.path.join(script_dir, args.experiments_root))
    os.makedirs(baselines_root, exist_ok=True)
    os.makedirs(experiments_root, exist_ok=True)

    baseline_fp = _baseline_fingerprint(cfg)
    baseline_cache_dir = os.path.join(baselines_root, baseline_fp)
    baseline_payload = _baseline_fingerprint_payload(cfg)
    baseline_payload["fingerprint"] = baseline_fp

    common_args = _build_common_args(cfg)
    test_layout_py = os.path.join(script_dir, "test_explosion.py")
    analyze_pair_py = os.path.join(script_dir, "analyze_explosion_pair.py")

    if args.baseline_dir:
        baseline_dir = os.path.abspath(args.baseline_dir)
        if not os.path.isfile(os.path.join(baseline_dir, "tour_lengths.csv")):
            raise FileNotFoundError(f"Baseline override missing tour_lengths.csv: {baseline_dir}")
        print(f"[BASELINE] Using provided baseline dir: {baseline_dir}")
    else:
        baseline_dir = baseline_cache_dir
        baseline_csv = os.path.join(baseline_dir, "tour_lengths.csv")
        needs_baseline = not os.path.isfile(baseline_csv) or bool(args.regenerate)
        if needs_baseline:
            print(f"[BASELINE] Building baseline cache: {baseline_dir}")
            if os.path.isdir(baseline_dir):
                shutil.rmtree(baseline_dir)
            os.makedirs(baseline_dir, exist_ok=True)
            baseline_cmd = [
                args.python_bin,
                "-u",
                test_layout_py,
                *common_args,
                "--with_RTDL",
                "0",
                "--use_rtdl_sampling",
                "0",
                "--model_name",
                cfg.baseline_model_name,
                "--regenerate_instances",
                str(args.regenerate),
                "--result_dir",
                baseline_dir,
            ]
            _run_cmd(baseline_cmd, cwd=script_dir)
            _write_json(os.path.join(baseline_dir, "baseline_fingerprint.json"), baseline_payload)
        else:
            print(f"[BASELINE] Reusing cached baseline: {baseline_dir}")

    if args.baseline_only:
        print("[DONE] Baseline-only mode")
        print(f"Baseline dir: {baseline_dir}")
        return

    if args.adv and args.sweep:
        raise ValueError("Use either --adv or --sweep, not both")

    variants: List[Dict[str, str]] = []
    if args.sweep:
        variants = _parse_sweep_expr(args.sweep)
    elif args.adv:
        variants = [_parse_kv_list(args.adv)]
    else:
        variants = [
            {
                "sampling_variant": DEFAULT_SAMPLING_VARIANT,
                "topk_frac": str(cfg.topk_frac),
                "temperature": str(cfg.temperature),
                "topk_min": str(cfg.topk_min),
                "sampling_window": str(cfg.sampling_window),
                "cluster_score_reduction": cfg.cluster_score_reduction,
                "multi_local_k_min": str(DEFAULT_MULTI_LOCAL_K_MIN),
                "multi_local_k_max": str(DEFAULT_MULTI_LOCAL_K_MAX),
            }
        ]

    sweep_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dirname = _build_sweep_dirname(args.config, cfg, sweep_id)
    sweep_dir = os.path.join(experiments_root, sweep_dirname)
    runs_root = os.path.join(sweep_dir, "runs")
    os.makedirs(sweep_dir, exist_ok=True)
    os.makedirs(runs_root, exist_ok=True)

    manifest = {
        "schema_version": 2,
        "layout": "single_root_runs_under_sweep",
        "created_at": datetime.now().isoformat(),
        "sweep_dir": sweep_dir,
        "sweep_dirname": sweep_dirname,
        "config": args.config,
        "baseline_fingerprint": baseline_fp,
        "baseline_dir": baseline_dir,
        "baseline_fingerprint_payload": baseline_payload,
        "runs": [],
    }

    for idx, raw_variant in enumerate(variants, start=1):
        sampling_variant = _normalize_sampling_variant(
            raw_variant.get("sampling_variant", DEFAULT_SAMPLING_VARIANT)
        )
        merged_variant = {
            "sampling_variant": sampling_variant,
            "topk_frac": raw_variant.get("topk_frac", str(cfg.topk_frac)),
            "temperature": raw_variant.get("temperature", str(cfg.temperature)),
            "topk_min": raw_variant.get("topk_min", str(cfg.topk_min)),
            "sampling_window": raw_variant.get("sampling_window", str(cfg.sampling_window)),
            "cluster_score_reduction": raw_variant.get("cluster_score_reduction", cfg.cluster_score_reduction),
            "multi_local_k_min": raw_variant.get("multi_local_k_min", str(DEFAULT_MULTI_LOCAL_K_MIN)),
            "multi_local_k_max": raw_variant.get("multi_local_k_max", str(DEFAULT_MULTI_LOCAL_K_MAX)),
            "edge_selection": raw_variant.get("edge_selection", "softmax"),
            "edge_target_ess_ratio": raw_variant.get("edge_target_ess_ratio", "-1"),
            "forbid_removed_edges": raw_variant.get("forbid_removed_edges", "0"),
        }
        _sw = int(merged_variant["sampling_window"])
        if _sw != 0:
            raise ValueError(
                "sampling_window in sweep must be 0 (cluster RTDL only); "
                f"got {_sw}. Tour-index window mode is no longer supported."
            )
        slug = _build_variant_slug(merged_variant)
        run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx:02d}_{slug}"
        run_dirname = f"{idx:02d}_{slug}"
        run_rel_path = os.path.join("runs", run_dirname)
        advanced_dir = os.path.join(sweep_dir, run_rel_path)
        compare_dir = os.path.join(advanced_dir, "compare")
        os.makedirs(advanced_dir, exist_ok=True)

        adv_model_name = f"{cfg.adv_model_name}_{slug}"
        if sampling_variant == "multi":
            use_rtdl_sampling = "1"
            multi_center = "1"
            edge_multi = "0"
            rtdl_edge = "0"
        elif sampling_variant == "multi_edge":
            use_rtdl_sampling = "1"
            multi_center = "0"
            edge_multi = "1"
            rtdl_edge = "0"
        elif sampling_variant == "rtdl_edge":
            use_rtdl_sampling = "1"
            multi_center = "0"
            edge_multi = "0"
            rtdl_edge = "1"
        else:
            use_rtdl_sampling = "1"
            multi_center = "0"
            edge_multi = "0"
            rtdl_edge = "0"

        print(f"[ADV {idx}/{len(variants)}] {slug} (sampling_variant={sampling_variant})")
        adv_cmd = [
            args.python_bin,
            "-u",
            test_layout_py,
            *common_args,
            "--with_RTDL",
            "0",
            "--use_rtdl_sampling",
            use_rtdl_sampling,
            "--rtdl_sampling_multi_center",
            multi_center,
            "--rtdl_sampling_edge_multi",
            edge_multi,
            "--rtdl_sampling_rtdl_edge",
            rtdl_edge,
            "--rtdl_sampling_edge_selection",
            str(merged_variant["edge_selection"]).lower(),
            "--rtdl_sampling_edge_target_ess_ratio",
            str(merged_variant["edge_target_ess_ratio"]),
            "--rtdl_sampling_forbid_removed_edges",
            str(merged_variant["forbid_removed_edges"]),
            "--rtdl_sampling_window",
            merged_variant["sampling_window"],
            "--rtdl_sampling_temperature",
            merged_variant["temperature"],
            "--rtdl_sampling_topk_frac",
            merged_variant["topk_frac"],
            "--rtdl_sampling_topk_min",
            merged_variant["topk_min"],
            "--rtdl_sampling_cluster_score_reduction",
            merged_variant["cluster_score_reduction"],
            "--rtdl_sampling_multi_local_k_min",
            merged_variant["multi_local_k_min"],
            "--rtdl_sampling_multi_local_k_max",
            merged_variant["multi_local_k_max"],
            "--model_name",
            adv_model_name,
            "--regenerate_instances",
            "0",
            "--result_dir",
            advanced_dir,
        ]
        _run_cmd(adv_cmd, cwd=script_dir)

        analyze_cmd = [
            args.python_bin,
            analyze_pair_py,
            "--baseline_dir",
            baseline_dir,
            "--advanced_dir",
            advanced_dir,
            "--output_dir",
            compare_dir,
            "--baseline_fingerprint",
            baseline_fp,
            "--experiment_params_json",
            json.dumps(merged_variant, sort_keys=True),
        ]
        _run_cmd(analyze_cmd, cwd=script_dir)

        metrics = _read_gap_metrics(os.path.join(compare_dir, "comparison_gap_to_concorde.csv"))
        run_record = {
            "slug": slug,
            "run_id": run_id,
            "run_dirname": run_dirname,
            "run_rel_path": run_rel_path,
            "advanced_dir": advanced_dir,
            "compare_dir": compare_dir,
            "compare_gap_csv": os.path.join(compare_dir, "comparison_gap_to_concorde.csv"),
            "params": merged_variant,
        }
        run_record.update(metrics)
        manifest["runs"].append(run_record)

    _write_json(os.path.join(sweep_dir, "manifest.json"), manifest)
    print("[DONE] All experiments completed.")
    print(f"Baseline dir: {baseline_dir}")
    print(f"Sweep dir: {sweep_dir}")


if __name__ == "__main__":
    main()
