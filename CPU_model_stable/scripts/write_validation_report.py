from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import pandas as pd


REFERENCE = {
    "v5_reference_e2e_mape": 0.0961,
    "v5_reference_e2e_p90": 0.2442,
    "current_no_descriptor_sync_reorder_reference_e2e_mape": 0.0955,
    "current_no_descriptor_sync_reorder_reference_e2e_p90": 0.2430,
}


def _load(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def _fmt_percent(value: float | int | str) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return str(value)


def _route_row(frame: pd.DataFrame, route: str) -> pd.Series | None:
    if frame.empty or "route" not in frame.columns:
        return None
    rows = frame[frame["route"].astype(str) == route]
    if rows.empty:
        return None
    return rows.iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Write CPU_model_stable validation report.")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--route", default="whitebox_family_prior_plus_gather_slice_rf_v5")
    args = parser.parse_args()

    root = args.artifact_root.resolve()
    e2e = _load(root / "e2e_pipeline_summary.csv")
    summary = _load(root / "summary.csv")
    op_audit = _load(root / "singleop_optype_threshold_audit.csv")
    scorecard = _load(root / "analytical_quality_scorecard.csv")

    e2e_row = _route_row(e2e, args.route)
    single_row = _route_row(summary, args.route)

    lines: list[str] = []
    lines.append("# CPU_model_stable 1/16 Validation Report")
    lines.append("")
    lines.append(f"- artifact_root: `{root}`")
    lines.append(f"- route: `{args.route}`")
    lines.append(f"- python: `{platform.python_version()}`")
    lines.append("")

    lines.append("## E2E Metrics")
    if e2e_row is None:
        lines.append("- E2E summary row not found.")
    else:
        mape = float(e2e_row.get("e2e_mape", float("nan")))
        p90 = float(e2e_row.get("e2e_p90", float("nan")))
        median = float(e2e_row.get("e2e_median", float("nan")))
        lines.append(f"- MAPE: {_fmt_percent(mape)}")
        lines.append(f"- P90 APE: {_fmt_percent(p90)}")
        lines.append(f"- Median APE: {_fmt_percent(median)}")
        lines.append(f"- Combo count: `{int(e2e_row.get('combo_count_full', 0))}`")
        lines.append(f"- Delta vs original v5 MAPE reference: {(mape - REFERENCE['v5_reference_e2e_mape']) * 100:+.2f}pp")
        lines.append(f"- Delta vs original v5 P90 reference: {(p90 - REFERENCE['v5_reference_e2e_p90']) * 100:+.2f}pp")
    lines.append("")

    lines.append("## Single-Op Overall")
    if single_row is None:
        lines.append("- Single-op summary row not found.")
    else:
        lines.append(f"- MAPE: {_fmt_percent(single_row.get('mape', float('nan')))}")
        lines.append(f"- P90 APE: {_fmt_percent(single_row.get('p90', float('nan')))}")
        lines.append(f"- Median APE: {_fmt_percent(single_row.get('median', float('nan')))}")
        lines.append(f"- Eval rows: `{int(single_row.get('eval_rows', 0))}`")
    lines.append("")

    lines.append("## Op-Type Metrics")
    if op_audit.empty:
        lines.append("- Op-type audit not found.")
    else:
        cols = [c for c in ["op_type", "route", "mape", "p90", "eval_rows", "status"] if c in op_audit.columns]
        route_audit = op_audit
        if "route" in route_audit.columns:
            route_audit = route_audit[route_audit["route"].astype(str) == args.route]
        lines.append("| op_type | MAPE | P90 APE | rows |")
        lines.append("|---|---:|---:|---:|")
        for _, row in route_audit.sort_values(cols[:1] or route_audit.columns[:1].tolist()).iterrows():
            lines.append(
                f"| {row.get('op_type', '')} | {_fmt_percent(row.get('mape', float('nan')))} | "
                f"{_fmt_percent(row.get('p90', float('nan')))} | "
                f"{int(row.get('eval_rows', row.get('row_count', row.get('rows', 0))) or 0)} |"
            )
    lines.append("")

    lines.append("## Analytical Quality")
    if scorecard.empty:
        lines.append("- Analytical quality scorecard not found.")
    else:
        lines.append("| metric | value | target | status |")
        lines.append("|---|---:|---:|---|")
        route_scorecard = scorecard
        if "route" in route_scorecard.columns:
            route_scorecard = route_scorecard[route_scorecard["route"].astype(str) == args.route]
        for _, row in route_scorecard.iterrows():
            metric = row.get("metric_name", row.get("metric", row.get("name", "")))
            value = row.get("value", row.get("actual", ""))
            target = row.get("target", "")
            status = row.get("status", row.get("pass_target", row.get("pass", "")))
            lines.append(f"| {metric} | {value} | {target} | {status} |")
    lines.append("")

    lines.append("## Generated Files")
    for name in [
        "e2e_pipeline_summary.csv",
        "e2e_pipeline_combo_metrics.csv",
        "summary.csv",
        "singleop_optype_threshold_audit.csv",
        "analytical_quality_scorecard.csv",
    ]:
        lines.append(f"- `{name}`: `{(root / name).exists()}`")
    lines.append("")

    (root / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "validation_reference.json").write_text(json.dumps(REFERENCE, indent=2), encoding="utf-8")
    print(root / "validation_report.md")


if __name__ == "__main__":
    main()
