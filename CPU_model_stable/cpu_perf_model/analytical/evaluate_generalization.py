from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import analytical_helpers as analytical_eval
from .contracts import DEFAULT_INPUT_CSV, DEFAULT_OUTPUT_DIR


DEFAULT_GROUP_COLUMN = "sample_group"
DEFAULT_SEED = 42
DEFAULT_TEST_RATIO = 0.30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate analytical calibration with a fixed 7:3 grouped holdout split.",
    )
    parser.add_argument(
        "--input-csv",
        default=str(DEFAULT_INPUT_CSV),
        help="Input dataset_full.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for grouped-holdout evaluation artifacts.",
    )
    parser.add_argument(
        "--hardware-profile",
        default="",
        help="Optional hardware profile YAML used when rebuilding analytical features.",
    )
    parser.add_argument(
        "--group-column",
        default=DEFAULT_GROUP_COLUMN,
        help="Column grouped as an inseparable unit during the train/test split.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_TEST_RATIO,
        help="Grouped holdout ratio for the test split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed used for grouped split assignment.",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=3,
        help="Coordinate-descent passes used during parameter calibration.",
    )
    return parser.parse_args()


def _normalized_group_values(frame: pd.DataFrame, group_column: str) -> pd.Series:
    if group_column in frame.columns:
        values = frame[group_column]
    else:
        values = pd.Series("", index=frame.index, dtype=str)
    normalized = values.fillna("").astype(str).str.strip()
    combo_fallback = frame.get("combo", pd.Series("", index=frame.index)).fillna("").astype(str).str.strip()
    row_fallback = frame.get("row_uid", pd.Series(range(len(frame)), index=frame.index)).astype(str)
    normalized = normalized.where(normalized.ne(""), combo_fallback)
    normalized = normalized.where(normalized.ne(""), row_fallback)
    return normalized


def _build_group_table(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    work = frame.copy()
    work["_group_value"] = _normalized_group_values(work, group_column)
    work["_group_key"] = work["case_id"].fillna("").astype(str) + "::" + work["_group_value"]

    rows: list[dict[str, Any]] = []
    for group_key, group in work.groupby("_group_key", sort=True):
        case_id = str(group["case_id"].iloc[0])
        group_value = str(group["_group_value"].iloc[0])
        rows.append(
            {
                "group_key": str(group_key),
                "case_id": case_id,
                "group_value": group_value,
                "row_count": int(len(group)),
                "actual_sum_us": float(pd.to_numeric(group["actual_us"], errors="coerce").fillna(0.0).sum()),
                "families": sorted(group["family"].dropna().astype(str).unique().tolist()),
            }
        )
    return pd.DataFrame(rows).sort_values(["case_id", "group_value"]).reset_index(drop=True)


def _assign_initial_splits(group_df: pd.DataFrame, test_ratio: float, seed: int) -> pd.DataFrame:
    assigned = group_df.copy()
    assigned["split"] = "train"

    for case_offset, (case_id, case_groups) in enumerate(assigned.groupby("case_id", sort=True)):
        case_indices = case_groups.index.to_numpy(dtype=int)
        if len(case_indices) <= 1:
            assigned.loc[case_indices, "split"] = "train"
            continue
        rng = np.random.default_rng(seed + case_offset)
        shuffled = case_indices.copy()
        rng.shuffle(shuffled)
        target_test = int(round(len(shuffled) * float(test_ratio)))
        target_test = min(max(target_test, 1), len(shuffled) - 1)
        test_indices = shuffled[:target_test]
        assigned.loc[test_indices, "split"] = "test"
    return assigned


def _case_split_count(group_df: pd.DataFrame, case_id: str, split_name: str) -> int:
    return int(((group_df["case_id"] == case_id) & (group_df["split"] == split_name)).sum())


def _family_group_count(group_df: pd.DataFrame, family: str, split_name: str) -> int:
    mask = group_df["split"].eq(split_name) & group_df["families"].map(lambda families: family in families)
    return int(mask.sum())


def _candidate_order(group_df: pd.DataFrame, family: str, source_split: str) -> list[int]:
    candidates = group_df[
        group_df["split"].eq(source_split) & group_df["families"].map(lambda families: family in families)
    ].copy()
    if candidates.empty:
        return []
    candidates = candidates.sort_values(["row_count", "actual_sum_us", "group_key"], ascending=[True, True, True])
    return candidates.index.astype(int).tolist()


def _maybe_move_group(
    group_df: pd.DataFrame,
    row_index: int,
    *,
    target_split: str,
) -> bool:
    current_split = str(group_df.at[row_index, "split"])
    if current_split == target_split:
        return False
    source_split = current_split
    case_id = str(group_df.at[row_index, "case_id"])
    if _case_split_count(group_df, case_id, source_split) <= 1:
        return False
    group_df.at[row_index, "split"] = target_split
    return True


def _enforce_family_coverage(group_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]], list[str]]:
    adjusted = group_df.copy()
    adjustments: list[dict[str, Any]] = []
    unresolved: list[str] = []
    families = sorted({family for row in adjusted["families"] for family in row})

    for family in families:
        family_group_total = int(adjusted["families"].map(lambda row: family in row).sum())
        if family_group_total < 2:
            unresolved.append(family)
            continue

        if _family_group_count(adjusted, family, "test") == 0:
            moved = False
            for row_index in _candidate_order(adjusted, family, "train"):
                if _maybe_move_group(adjusted, row_index, target_split="test"):
                    adjustments.append(
                        {
                            "family": family,
                            "group_key": str(adjusted.at[row_index, "group_key"]),
                            "case_id": str(adjusted.at[row_index, "case_id"]),
                            "from_split": "train",
                            "to_split": "test",
                            "reason": "family_missing_in_test",
                        }
                    )
                    moved = True
                    break
            if not moved:
                unresolved.append(family)

        if _family_group_count(adjusted, family, "train") == 0:
            moved = False
            for row_index in _candidate_order(adjusted, family, "test"):
                if _maybe_move_group(adjusted, row_index, target_split="train"):
                    adjustments.append(
                        {
                            "family": family,
                            "group_key": str(adjusted.at[row_index, "group_key"]),
                            "case_id": str(adjusted.at[row_index, "case_id"]),
                            "from_split": "test",
                            "to_split": "train",
                            "reason": "family_missing_in_train",
                        }
                    )
                    moved = True
                    break
            if not moved and family not in unresolved:
                unresolved.append(family)

    return adjusted, adjustments, sorted(set(unresolved))


def _split_manifest(
    heavy_df: pd.DataFrame,
    group_df: pd.DataFrame,
    *,
    input_csv: Path,
    group_column: str,
    seed: int,
    test_ratio: float,
    passes: int,
) -> dict[str, Any]:
    family_presence: list[dict[str, Any]] = []
    families = sorted({family for family in heavy_df["family"].dropna().astype(str).tolist()})
    for family in families:
        family_presence.append(
            {
                "family": family,
                "train_group_count": _family_group_count(group_df, family, "train"),
                "test_group_count": _family_group_count(group_df, family, "test"),
                "train_row_count": int(len(heavy_df[(heavy_df["family"] == family) & (heavy_df["_split"] == "train")])),
                "test_row_count": int(len(heavy_df[(heavy_df["family"] == family) & (heavy_df["_split"] == "test")])),
            }
        )

    per_case: list[dict[str, Any]] = []
    for case_id, case_groups in group_df.groupby("case_id", sort=True):
        per_case.append(
            {
                "case_id": str(case_id),
                "group_count": int(len(case_groups)),
                "train_group_count": int(case_groups["split"].eq("train").sum()),
                "test_group_count": int(case_groups["split"].eq("test").sum()),
            }
        )

    return {
        "input_csv": str(input_csv),
        "evaluation_protocol": "train_test_7_3_grouped",
        "group_column": group_column,
        "seed": int(seed),
        "test_ratio": float(test_ratio),
        "passes": int(passes),
        "row_count": int(len(heavy_df)),
        "group_count": int(len(group_df)),
        "train_row_count": int((heavy_df["_split"] == "train").sum()),
        "test_row_count": int((heavy_df["_split"] == "test").sum()),
        "train_group_count": int(group_df["split"].eq("train").sum()),
        "test_group_count": int(group_df["split"].eq("test").sum()),
        "per_case_group_counts": per_case,
        "family_presence": family_presence,
    }


def render_markdown(
    *,
    input_csv: Path,
    split_manifest: dict[str, Any],
    train_summary: dict[str, Any],
    test_summary: dict[str, Any],
    parameters: dict[str, float],
) -> str:
    lines = [
        "# Analytical Grouped Holdout Evaluation",
        "",
        f"- Input dataset: `{input_csv}`",
        f"- Protocol: `{split_manifest['evaluation_protocol']}`",
        f"- Group column: `{split_manifest['group_column']}`",
        f"- Seed: `{split_manifest['seed']}`",
        f"- Group split: train/test = `{split_manifest['train_group_count']}/{split_manifest['test_group_count']}`",
        f"- Row split: train/test = `{split_manifest['train_row_count']}/{split_manifest['test_row_count']}`",
        "",
        "## Overall Metrics",
        "",
        "| split | weighted family MAPE | duration-weighted RE |",
        "| --- | ---: | ---: |",
        f"| train | {train_summary.get('weighted_family_mape', 0.0) * 100.0:.2f}% | {train_summary.get('duration_weighted_relative_error', 0.0) * 100.0:.2f}% |",
        f"| test | {test_summary.get('weighted_family_mape', 0.0) * 100.0:.2f}% | {test_summary.get('duration_weighted_relative_error', 0.0) * 100.0:.2f}% |",
        "",
        "## Test Family Metrics",
        "",
        "| family | rows | mean MAPE | mean DWRE |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in test_summary.get("family_summary", []):
        lines.append(
            f"| `{row['family']}` | {int(row['total_rows'])} | "
            f"{row['mean_mape'] * 100.0:.2f}% | {row['mean_dwre'] * 100.0:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Parameters",
            "",
            "| name | value |",
            "| --- | ---: |",
        ]
    )
    for key in sorted(parameters):
        lines.append(f"| `{key}` | {float(parameters[key]):.6f} |")
    lines.append("")
    return "\n".join(lines)


def evaluate_generalization(
    input_csv: Path,
    output_dir: Path,
    *,
    passes: int,
    hardware_profile_path: Path | None,
    group_column: str,
    test_ratio: float,
    seed: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    heavy_df = analytical_eval.prepare_heavy_slice(input_csv, hardware_profile_path=hardware_profile_path)
    if heavy_df.empty:
        raise RuntimeError(f"No analytical heavy-family rows were found in {input_csv}")

    group_df = _build_group_table(heavy_df, group_column)
    group_df = _assign_initial_splits(group_df, test_ratio=float(test_ratio), seed=int(seed))
    group_df, adjustments, unresolved_families = _enforce_family_coverage(group_df)

    heavy_df = heavy_df.copy()
    heavy_df["_group_value"] = _normalized_group_values(heavy_df, group_column)
    heavy_df["_group_key"] = heavy_df["case_id"].fillna("").astype(str) + "::" + heavy_df["_group_value"]
    heavy_df = heavy_df.merge(group_df[["group_key", "split"]], left_on="_group_key", right_on="group_key", how="left", validate="many_to_one")
    heavy_df["_split"] = heavy_df["split"].fillna("train").astype(str)
    heavy_df = heavy_df.drop(columns=["group_key", "split"])

    train_df = heavy_df[heavy_df["_split"] == "train"].copy()
    test_df = heavy_df[heavy_df["_split"] == "test"].copy()
    if train_df.empty or test_df.empty:
        raise RuntimeError("Grouped holdout split produced an empty train or test partition")

    params = analytical_eval.calibrate_params(train_df, passes=passes)
    train_metrics = pd.DataFrame(analytical_eval.family_metric_rows(train_df, params, "train"))
    test_metrics = pd.DataFrame(analytical_eval.family_metric_rows(test_df, params, "test"))
    metrics_df = pd.concat([train_metrics, test_metrics], ignore_index=True)

    split_manifest = _split_manifest(
        heavy_df,
        group_df,
        input_csv=input_csv,
        group_column=group_column,
        seed=seed,
        test_ratio=test_ratio,
        passes=passes,
    )
    split_manifest["coverage_adjustments"] = adjustments
    split_manifest["families_without_dual_split_coverage"] = unresolved_families

    train_summary = analytical_eval.summarize_metrics(metrics_df, "train")
    test_summary = analytical_eval.summarize_metrics(metrics_df, "test")
    evaluation_metrics = {
        "evaluation_protocol": "train_test_7_3_grouped",
        "train": train_summary,
        "test": test_summary,
    }
    summary_payload = {
        **evaluation_metrics,
        "split_manifest": split_manifest,
        "parameters": params,
    }

    evaluation_family_metrics_path = output_dir / "evaluation_family_metrics.csv"
    metrics_df.to_csv(evaluation_family_metrics_path, index=False)

    split_manifest_path = output_dir / "evaluation_split_manifest.json"
    split_manifest_path.write_text(json.dumps(split_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    parameters_path = output_dir / "evaluation_parameters.json"
    parameters_path.write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")

    evaluation_metrics_path = output_dir / "evaluation_metrics.json"
    evaluation_metrics_path.write_text(json.dumps(evaluation_metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    generalization_summary_path = output_dir / "generalization_summary.json"
    generalization_summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Keep a neutral compatibility alias for existing readers.
    summary_json_path = output_dir / "summary.json"
    summary_json_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_md_path = output_dir / "generalization_summary.md"
    summary_md_path.write_text(
        render_markdown(
            input_csv=input_csv,
            split_manifest=split_manifest,
            train_summary=train_summary,
            test_summary=test_summary,
            parameters=params,
        ),
        encoding="utf-8",
    )

    manifest = {
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "evaluation_protocol": "train_test_7_3_grouped",
        "group_column": group_column,
        "hardware_profile_path": "" if hardware_profile_path is None else str(hardware_profile_path),
        "passes": int(passes),
        "artifacts": {
            "evaluation_split_manifest_json": str(split_manifest_path),
            "evaluation_metrics_json": str(evaluation_metrics_path),
            "evaluation_parameters_json": str(parameters_path),
            "evaluation_family_metrics_csv": str(evaluation_family_metrics_path),
            "generalization_summary_json": str(generalization_summary_path),
            "summary_json": str(summary_json_path),
            "generalization_summary_md": str(summary_md_path),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    hardware_profile_path = Path(args.hardware_profile) if args.hardware_profile else None
    manifest = evaluate_generalization(
        Path(args.input_csv),
        Path(args.output_dir),
        passes=args.passes,
        hardware_profile_path=hardware_profile_path,
        group_column=args.group_column,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
