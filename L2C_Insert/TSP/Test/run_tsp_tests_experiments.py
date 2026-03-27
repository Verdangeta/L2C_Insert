#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class PairConfig:
    model_path: str
    cuda_device_num: int
    rrc_budget: int
    rrc_range: int
    temperature: float
    topk_frac: float
    topk_min: int
    sampling_window: int
    cluster_score_reduction: str
    sampling_log_every: int
    tnm_start_idx: int
    tnm_end_idx: Optional[int]


PRESETS: Dict[str, PairConfig] = {
    "tsp_default": PairConfig(
        model_path="/trinity/home/alexander.mironenko/TDA_tsp/L2C_Insert/L2C_Insert/TSP/Test/result/pretrain/tsp_model.pt",
        cuda_device_num=0,
        rrc_budget=1000,
        rrc_range=300,
        temperature=1.0,
        topk_frac=0.05,
        topk_min=20,
        sampling_window=0,
        cluster_score_reduction="sum",
        sampling_log_every=50,
        tnm_start_idx=0,
        tnm_end_idx=None,
    ),
    "tsp_quick_smoke": PairConfig(
        model_path="/trinity/home/alexander.mironenko/TDA_tsp/L2C_Insert/L2C_Insert/TSP/Test/result/pretrain/tsp_model.pt",
        cuda_device_num=0,
        rrc_budget=200,
        rrc_range=100,
        temperature=1.0,
        topk_frac=0.2,
        topk_min=20,
        sampling_window=0,
        cluster_score_reduction="sum",
        sampling_log_every=25,
        tnm_start_idx=0,
        tnm_end_idx=5,
    ),
}

ALLOWED_TASKS = {"tnm", "synthetic", "lib"}
RESULT_DIR_PREFIX = "Results will be saved to:"


def _parse_kv_list(entries: Optional[List[str]]) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    if not entries:
        return parsed
    for item in entries:
        if "=" not in item:
            raise ValueError(f"Invalid --set entry '{item}', expected key=value")
        key, value = item.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _cast_value(field_name: str, raw_value: str):
    if field_name in {"model_path", "cluster_score_reduction"}:
        return raw_value
    if field_name in {"cuda_device_num", "rrc_budget", "rrc_range", "topk_min", "sampling_window", "sampling_log_every", "tnm_start_idx"}:
        return int(raw_value)
    if field_name in {"temperature", "topk_frac"}:
        return float(raw_value)
    if field_name == "tnm_end_idx":
        if raw_value.lower() in {"none", "null", ""}:
            return None
        return int(raw_value)
    raise ValueError(f"Unknown config field: {field_name}")


def _apply_overrides(cfg: PairConfig, overrides: Dict[str, str]) -> PairConfig:
    if not overrides:
        return cfg
    data = asdict(cfg)
    for key, value in overrides.items():
        if key not in data:
            raise ValueError(f"Unsupported --set key: {key}")
        data[key] = _cast_value(key, value)
    return PairConfig(**data)


def _parse_tasks(raw_tasks: str) -> List[str]:
    items = [x.strip() for x in raw_tasks.split(",") if x.strip()]
    if not items:
        raise ValueError("--tasks cannot be empty")
    if items == ["all"]:
        return ["tnm", "synthetic", "lib"]
    bad = [x for x in items if x not in ALLOWED_TASKS]
    if bad:
        raise ValueError(f"Unknown tasks: {bad}. Allowed: tnm,synthetic,lib,all")
    return items


def _run_and_capture_result_dir(cmd: List[str], cwd: str, log_path: str) -> str:
    print(f"[RUN] {shlex.join(cmd)}")
    result_dir = ""
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
            if RESULT_DIR_PREFIX in line:
                result_dir = line.split(RESULT_DIR_PREFIX, 1)[1].strip()
        exit_code = process.wait()
    if exit_code != 0:
        raise RuntimeError(f"Command failed with exit code {exit_code}: {shlex.join(cmd)}")
    if not result_dir:
        raise RuntimeError(f"Could not parse result dir from logs: {log_path}")
    return os.path.abspath(result_dir)


def _run_cmd(cmd: List[str], cwd: str) -> None:
    print(f"[RUN] {shlex.join(cmd)}")
    completed = subprocess.run(cmd, cwd=cwd, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {shlex.join(cmd)}")


def _build_task_base_args(task: str, cfg: PairConfig) -> List[str]:
    args = [
        "--cuda_device_num", str(cfg.cuda_device_num),
        "--RRC_budget", str(cfg.rrc_budget),
        "--RRC_range", str(cfg.rrc_range),
        "--model_path", cfg.model_path,
        "--rtdl_sampling_window", str(cfg.sampling_window),
        "--rtdl_sampling_temperature", str(cfg.temperature),
        "--rtdl_sampling_topk_frac", str(cfg.topk_frac),
        "--rtdl_sampling_topk_min", str(cfg.topk_min),
        "--rtdl_sampling_cluster_score_reduction", cfg.cluster_score_reduction,
        "--rtdl_sampling_log_every", str(cfg.sampling_log_every),
    ]
    if task == "tnm":
        args.extend(["--start_idx", str(cfg.tnm_start_idx)])
        if cfg.tnm_end_idx is not None:
            args.extend(["--end_idx", str(cfg.tnm_end_idx)])
    return args


def _task_script(task: str, script_dir: str) -> str:
    return os.path.join(script_dir, f"test_{task}.py")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _run_pair_analysis(
    task: str,
    python_bin: str,
    script_dir: str,
    baseline_dir: str,
    advanced_dir: str,
    baseline_log: str,
    advanced_log: str,
) -> Dict[str, str]:
    compare_dir = os.path.join(advanced_dir, "compare_with_baseline")
    _ensure_dir(compare_dir)

    if task in {"lib", "synthetic"}:
        analyzer = os.path.join(script_dir, "result", "analyze_tsplib_rtdl_compare.py")
        report_path = os.path.join(compare_dir, f"{task}_comparison_use_rtdl_sampling.txt")
        cmd = [
            python_bin,
            analyzer,
            "--log_with_rtdl",
            os.path.join(advanced_dir, "log.txt"),
            "--log_without_rtdl",
            os.path.join(baseline_dir, "log.txt"),
            "--name_with_rtdl",
            f"{task}_use_rtdl_sampling=ON",
            "--name_without_rtdl",
            f"{task}_use_rtdl_sampling=OFF",
            "--out_path",
            report_path,
        ]
        _run_cmd(cmd, cwd=script_dir)
        return {
            "compare_dir": compare_dir,
            "compare_report": report_path,
            "compare_type": "log_parser",
            "baseline_log": baseline_log,
            "advanced_log": advanced_log,
        }

    if task == "tnm":
        analyzer = os.path.join(script_dir, "analyze_tnm_pair.py")
        report_path = os.path.join(compare_dir, "tnm_pair_summary.txt")
        baseline_json = os.path.join(baseline_dir, "tnm_results.json")
        advanced_json = os.path.join(advanced_dir, "tnm_results.json")
        cmd = [
            python_bin,
            analyzer,
            "--baseline_json",
            baseline_json,
            "--advanced_json",
            advanced_json,
            "--output_path",
            report_path,
        ]
        _run_cmd(cmd, cwd=script_dir)
        return {
            "compare_dir": compare_dir,
            "compare_report": report_path,
            "compare_type": "tnm_json",
            "baseline_json": baseline_json,
            "advanced_json": advanced_json,
        }

    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline/advanced pairs for TSP test scripts")
    parser.add_argument("--config", type=str, default="tsp_default", choices=sorted(PRESETS.keys()))
    parser.add_argument("--tasks", type=str, default="all", help="Comma-separated list: tnm,synthetic,lib,all")
    parser.add_argument("--regenerate", type=int, default=0, help="Reserved flag for compatibility with pair wrappers")
    parser.add_argument("--baseline-only", type=int, default=0, help="Run only baseline if set to 1")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--logs-root", type=str, default="./pair_run_logs")
    parser.add_argument("--set", nargs="*", default=None, help="Override config values: key=value")
    args = parser.parse_args()

    if args.regenerate not in (0, 1):
        raise ValueError("--regenerate must be 0 or 1")
    if args.baseline_only not in (0, 1):
        raise ValueError("--baseline-only must be 0 or 1")

    cfg = _apply_overrides(PRESETS[args.config], _parse_kv_list(args.set))
    tasks = _parse_tasks(args.tasks)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_root = os.path.abspath(os.path.join(script_dir, args.logs_root))
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(logs_root, f"pair_{run_ts}_{args.config}")
    _ensure_dir(run_dir)

    manifest = {
        "created_at": datetime.now().isoformat(),
        "config_name": args.config,
        "config": asdict(cfg),
        "tasks": tasks,
        "baseline_only": bool(args.baseline_only),
        "runs": {},
    }

    for task in tasks:
        print(f"\n=== Task: {task} ===")
        base_args = _build_task_base_args(task, cfg)
        script_path = _task_script(task, script_dir)
        if not os.path.isfile(script_path):
            raise FileNotFoundError(f"Task script not found: {script_path}")

        baseline_log = os.path.join(run_dir, f"{task}_baseline.log")
        baseline_cmd = [
            args.python_bin,
            "-u",
            script_path,
            *base_args,
            "--with_RTDL", "0",
            "--use_rtdl_sampling", "0",
        ]
        baseline_dir = _run_and_capture_result_dir(baseline_cmd, cwd=script_dir, log_path=baseline_log)

        advanced_log = None
        advanced_dir = None
        if not args.baseline_only:
            advanced_log = os.path.join(run_dir, f"{task}_advanced.log")
            advanced_cmd = [
                args.python_bin,
                "-u",
                script_path,
                *base_args,
                "--with_RTDL", "0",
                "--use_rtdl_sampling", "1",
            ]
            advanced_dir = _run_and_capture_result_dir(advanced_cmd, cwd=script_dir, log_path=advanced_log)

        task_record = {
            "baseline_log": baseline_log,
            "baseline_dir": baseline_dir,
            "advanced_log": advanced_log,
            "advanced_dir": advanced_dir,
        }
        if advanced_dir is not None and advanced_log is not None:
            task_record.update(
                _run_pair_analysis(
                    task=task,
                    python_bin=args.python_bin,
                    script_dir=script_dir,
                    baseline_dir=baseline_dir,
                    advanced_dir=advanced_dir,
                    baseline_log=baseline_log,
                    advanced_log=advanced_log,
                )
            )
        manifest["runs"][task] = task_record

    manifest_path = os.path.join(run_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as out:
        json.dump(manifest, out, indent=2, ensure_ascii=False)

    print("\n[DONE] Pair run complete.")
    print(f"Run dir: {run_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
