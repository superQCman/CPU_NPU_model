from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .feature_engineering import normalize_node_name
from .graph_contract import ComboArtifactPaths, ComboSpec
from .paths import DEFAULT_CLASSED_MODEL_ROOT, ORT_ROOT

DEFAULT_ORT_ROOT = ORT_ROOT
DEFAULT_ARTIFACT_ROOT = DEFAULT_CLASSED_MODEL_ROOT

COMBO_PATTERN = re.compile(r"^bs(?P<batch_size>\d+)_nip(?P<nip>\d+)$")


def format_combo(batch_size: int, num_indices_per_lookup: int) -> str:
    return f"bs{int(batch_size)}_nip{int(num_indices_per_lookup)}"


def parse_combo(combo: str) -> tuple[int, int]:
    match = COMBO_PATTERN.fullmatch(combo)
    if not match:
        raise ValueError(f"Unexpected combo format: {combo}")
    return int(match.group("batch_size")), int(match.group("nip"))


def _prediction_path(artifact_root: Path, split: str) -> Path:
    return artifact_root / "models" / "combined" / f"combined_predictions_{split}.csv"


def load_prediction_frame(artifact_root: Path = DEFAULT_ARTIFACT_ROOT, split: str = "test") -> pd.DataFrame:
    artifact_root = Path(artifact_root)
    prediction_df = pd.read_csv(_prediction_path(artifact_root, split))
    prediction_df["op_idx"] = prediction_df["row_uid"].str.extract(r"opidx:(\d+)").astype(int)

    dataset_df = pd.read_csv(
        artifact_root / "classed_dataset_full.csv",
        usecols=[
            "row_uid",
            "node_name",
            "batch_size",
            "num_indices_per_lookup",
            "inter_threads",
            "profile_last2_range_ratio",
        ],
    )
    merged = prediction_df.merge(dataset_df, on="row_uid", how="left", validate="one_to_one")
    if merged["node_name"].isna().any():
        missing = int(merged["node_name"].isna().sum())
        raise ValueError(f"Failed to recover node metadata for {missing} prediction rows")
    merged["inter_threads"] = merged["inter_threads"].astype(int)
    merged["batch_size"] = merged["batch_size"].astype(int)
    merged["num_indices_per_lookup"] = merged["num_indices_per_lookup"].astype(int)
    return merged


@lru_cache(maxsize=None)
def load_case_sweep_summary(case_id: str, ort_root: Path = DEFAULT_ORT_ROOT) -> pd.DataFrame:
    sweep_summary_path = Path(ort_root) / f"sweep_runs_extensible_{case_id}" / "sweep_summary.csv"
    sweep_df = pd.read_csv(sweep_summary_path)
    sweep_df["combo"] = sweep_df.apply(
        lambda row: format_combo(row["batch_size"], row["num_indices_per_lookup"]),
        axis=1,
    )
    return sweep_df


def load_combo_artifact_index(
    case_ids: list[str] | tuple[str, ...] | set[str],
    ort_root: Path = DEFAULT_ORT_ROOT,
) -> dict[tuple[str, str], ComboArtifactPaths]:
    artifact_index: dict[tuple[str, str], ComboArtifactPaths] = {}
    for case_id in sorted(case_ids):
        sweep_df = load_case_sweep_summary(case_id, Path(ort_root))
        for row in sweep_df.itertuples(index=False):
            combo = format_combo(row.batch_size, row.num_indices_per_lookup)
            profile_dir = Path(row.profile_dir)
            artifact_index[(case_id, combo)] = ComboArtifactPaths(
                case_id=case_id,
                combo=combo,
                shape_csv=Path(row.shape_csv),
                profile_dir=profile_dir,
                timeline_csv=profile_dir / "branch_parallel_op_timeline.csv",
            )
    return artifact_index


def build_combo_specs(
    prediction_df: pd.DataFrame,
    ort_root: Path = DEFAULT_ORT_ROOT,
) -> list[ComboSpec]:
    case_ids = sorted(prediction_df["case_id"].unique())
    artifact_index = load_combo_artifact_index(case_ids, Path(ort_root))

    combo_specs: list[ComboSpec] = []
    grouped = (
        prediction_df[
            ["case_id", "combo", "batch_size", "num_indices_per_lookup", "inter_threads"]
        ]
        .drop_duplicates()
        .sort_values(["case_id", "batch_size", "num_indices_per_lookup"])
    )
    for row in grouped.itertuples(index=False):
        key = (row.case_id, row.combo)
        if key not in artifact_index:
            raise KeyError(f"Missing sweep metadata for {row.case_id} / {row.combo}")
        combo_specs.append(
            ComboSpec(
                case_id=row.case_id,
                combo=row.combo,
                batch_size=int(row.batch_size),
                num_indices_per_lookup=int(row.num_indices_per_lookup),
                inter_threads=int(row.inter_threads),
                artifact_paths=artifact_index[key],
            )
        )
    return combo_specs


@lru_cache(maxsize=None)
def load_op_shapes_frame(shape_csv: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(shape_csv))


def normalize_timeline_node_name(node_name: str | float | int | None) -> str:
    normalized = normalize_node_name(node_name)
    if normalized.startswith("fused "):
        normalized = normalized[len("fused ") :]
    if normalized == "/MatMul/MatmulTransposeFusion/":
        return "/MatMul"
    return normalized


@lru_cache(maxsize=None)
def load_timeline_frame(timeline_csv: str | Path) -> pd.DataFrame:
    timeline_df = pd.read_csv(Path(timeline_csv))
    timeline_df["normalized_node_name"] = timeline_df["node_name"].map(normalize_timeline_node_name)
    return timeline_df
