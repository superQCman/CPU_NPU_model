from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_LOGS_ROOT = PROJECT_ROOT / "data" / "case_dataset_scheduler_logs"

LOG_PATTERN = re.compile(
    r"case_(?P<case>\d+)_bs1024_2048_nip1000_2000_t(?P<num_threads>\d+)_i(?P<inter_threads>\d+)\.log$"
)

CASE_ARCH_FIELDS: dict[str, dict[str, str]] = {
    "case_7": {
        "arch_embedding_size": "25600-25600-25600-25600-25600-25600-25600-25600",
        "arch_mlp_bot": "400-300-200",
        "arch_mlp_top": "1600-800-1",
    },
    "case_8": {
        "arch_embedding_size": "25600-25600-25600-25600-25600-25600-25600-25600",
        "arch_mlp_bot": "800-800-400",
        "arch_mlp_top": "1600-800-1",
    },
    "case_9": {
        "arch_embedding_size": "51200-51200-51200-51200-51200-51200-51200-51200",
        "arch_mlp_bot": "400-300-200",
        "arch_mlp_top": "1600-800-1",
    },
    "case_10": {
        "arch_embedding_size": "51200-51200-51200-51200-51200-51200-51200-51200",
        "arch_mlp_bot": "800-800-400",
        "arch_mlp_top": "1600-800-1",
    },
}


def case_family(case_number: int | str) -> str:
    return f"case_{int(case_number)}"


def build_case_id(case_number: int | str, num_threads: int | str, inter_threads: int | str) -> str:
    return f"{case_family(case_number)}_{int(num_threads)}_{int(inter_threads)}"


def build_source_name(case_number: int | str, num_threads: int | str, inter_threads: int | str) -> str:
    return f"{case_family(case_number)}_run_{int(num_threads)}_{int(inter_threads)}"


def arch_fields_for_case_family(case_name: str) -> dict[str, str]:
    if case_name not in CASE_ARCH_FIELDS:
        raise KeyError(f"Unsupported case family for architecture mapping: {case_name}")
    return dict(CASE_ARCH_FIELDS[case_name])


def _load_scheduler_summary(root: Path) -> dict[str, dict[str, str]]:
    summary_path = root / "scheduler_summary.csv"
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        return {
            str(row.get("job_name", "")).strip(): row
            for row in csv.DictReader(handle)
            if str(row.get("job_name", "")).strip()
        }


def _fallback_data_root(root: Path) -> Path:
    if root.parent.name == "case_dataset_scheduler_logs":
        return root.parent.parent
    return PROJECT_ROOT / "data"


def parse_scheduler_logs(logs_root: Path | str = DEFAULT_LOGS_ROOT) -> list[dict[str, Any]]:
    root = Path(logs_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Scheduler logs root does not exist: {root}")
    if not any(root.glob("*.log")):
        candidates = [
            path.parent
            for path in root.glob("*/scheduler_summary.csv")
            if path.is_file()
        ]
        if candidates:
            root = max(candidates, key=lambda path: path.stat().st_mtime)

    specs: list[dict[str, Any]] = []
    summary_by_job = _load_scheduler_summary(root)
    data_root = _fallback_data_root(root)
    for log_path in sorted(root.glob("*.log")):
        match = LOG_PATTERN.match(log_path.name)
        if not match:
            continue
        case_number = int(match.group("case"))
        num_threads = int(match.group("num_threads"))
        inter_threads = int(match.group("inter_threads"))
        family = case_family(case_number)
        job_name = log_path.stem
        summary = summary_by_job.get(job_name, {})
        feature_root_raw = str(summary.get("feature_dataset_root") or "").strip()
        if feature_root_raw:
            feature_root = Path(feature_root_raw).expanduser()
        else:
            feature_root = data_root / (
                f"features_extensible_case_{case_number}_bs1024_2048_nip1000_2000_t{num_threads}_i{inter_threads}"
            )
        sweep_dir_raw = str(summary.get("out_root") or "").strip()
        if sweep_dir_raw:
            sweep_dir = Path(sweep_dir_raw).expanduser()
        else:
            sweep_dir = data_root / (
                f"sweep_runs_extensible_case_{case_number}_bs1024_2048_nip1000_2000_t{num_threads}_i{inter_threads}"
            )
        if not feature_root.exists():
            raise FileNotFoundError(f"Feature root referenced by {log_path.name} does not exist: {feature_root}")
        if not sweep_dir.exists():
            raise FileNotFoundError(f"Sweep dir referenced by {log_path.name} does not exist: {sweep_dir}")
        spec = {
            "case_family": family,
            "case_id": build_case_id(case_number, num_threads, inter_threads),
            "source_name": build_source_name(case_number, num_threads, inter_threads),
            "source_mode": "concurrent" if inter_threads > 1 else "serial_control",
            "num_threads": num_threads,
            "inter_threads": inter_threads,
            "feature_root": feature_root.resolve(),
            "sweep_dir": sweep_dir.resolve(),
            "log_path": log_path.resolve(),
        }
        spec.update(arch_fields_for_case_family(family))
        specs.append(spec)
    if not specs:
        raise ValueError(f"No scheduler log entries matched the expected pattern under {root}")
    return specs


def filter_specs_by_case_families(specs: list[dict[str, Any]], families: set[str]) -> list[dict[str, Any]]:
    allowed = {str(value) for value in families}
    return [dict(spec) for spec in specs if str(spec["case_family"]) in allowed]


def manifest_sources_from_specs(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "source_name": str(spec["source_name"]),
            "case_id": str(spec["case_id"]),
            "source_mode": str(spec["source_mode"]),
            "feature_root": str(Path(spec["feature_root"]).resolve()),
            "feature_glob": "bs*_nip*.csv",
            "sweep_dir": str(Path(spec["sweep_dir"]).resolve()),
        }
        for spec in specs
    ]


def static_source_metadata_map(specs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(spec["source_name"]): {
            "case_id": str(spec["case_id"]),
            "num_threads": int(spec["num_threads"]),
            "inter_threads": int(spec["inter_threads"]),
            "arch_embedding_size": str(spec["arch_embedding_size"]),
            "arch_mlp_bot": str(spec["arch_mlp_bot"]),
            "arch_mlp_top": str(spec["arch_mlp_top"]),
            "case_family": str(spec["case_family"]),
            "source_mode": str(spec["source_mode"]),
            "log_path": str(Path(spec["log_path"]).resolve()),
            "feature_root": str(Path(spec["feature_root"]).resolve()),
            "sweep_dir": str(Path(spec["sweep_dir"]).resolve()),
        }
        for spec in specs
    }
