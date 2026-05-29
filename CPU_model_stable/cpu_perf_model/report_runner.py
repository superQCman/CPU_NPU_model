from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .paths import (
    DEFAULT_CLASSED_MODEL_ROOT,
    DEFAULT_PIPELINE_ROOT,
    DEFAULT_REPORT_DRAFT_PATH,
    DEFAULT_REPORT_ROOT,
)
from .report_config import ONLY_CHOICES
from .report_shared import (
    PREDICTION_SOURCE_SINGLE_FAIR,
    SectionResult,
    build_figures_catalog,
    build_report_draft,
    build_run_manifest,
    run_e2e_core,
    run_e2e_sum_baseline,
    run_platform_summary,
    run_single_op_ablation,
    run_single_op_core,
    run_single_op_fair_baseline,
    run_single_op_ood,
    run_timeline_cases,
)


def _section_results_to_json(results: list[SectionResult]) -> list[dict[str, object]]:
    return [{"name": result.name, "outputs": result.outputs} for result in results]


def run_cpu_single_only_report(
    *,
    output_root: Path = DEFAULT_REPORT_ROOT,
    draft_path: Path = DEFAULT_REPORT_DRAFT_PATH,
    single_op_artifact_root: Path | None = None,
    e2e_artifact_root: Path | None = None,
    skip_ood: bool = False,
    skip_ablation: bool = False,
    skip_timelines: bool = False,
    only: str = "all",
) -> dict[str, Any]:
    resolved_single_op_root = single_op_artifact_root or DEFAULT_CLASSED_MODEL_ROOT
    resolved_e2e_root = e2e_artifact_root or DEFAULT_PIPELINE_ROOT
    sections: list[SectionResult] = []
    sections.append(
        run_platform_summary(
            output_root,
            single_op_artifact_root=resolved_single_op_root,
            e2e_artifact_root=resolved_e2e_root,
        )
    )

    if only in {"all", "single_op"}:
        sections.append(
            run_single_op_fair_baseline(
                output_root,
                single_op_artifact_root=resolved_single_op_root,
            )
        )
        sections.append(
            run_single_op_core(
                output_root,
                single_op_artifact_root=resolved_single_op_root,
                baseline_model_root=None,
                e2e_artifact_root=resolved_e2e_root,
                prediction_source=PREDICTION_SOURCE_SINGLE_FAIR,
            )
        )
        if not skip_ood:
            sections.append(
                run_single_op_ood(
                    output_root,
                    single_op_artifact_root=resolved_single_op_root,
                    ood_artifact_root=None,
                    prediction_source=PREDICTION_SOURCE_SINGLE_FAIR,
                )
            )
        if not skip_ablation:
            sections.append(
                run_single_op_ablation(
                    output_root,
                    ablation_artifact_root=resolved_e2e_root,
                    single_op_artifact_root=resolved_single_op_root,
                    e2e_artifact_root=resolved_e2e_root,
                    prediction_source=PREDICTION_SOURCE_SINGLE_FAIR,
                )
            )

    if only in {"all", "e2e"}:
        sections.append(
            run_e2e_core(
                output_root,
                e2e_artifact_root=resolved_e2e_root,
                single_op_artifact_root=resolved_single_op_root,
                prediction_source=PREDICTION_SOURCE_SINGLE_FAIR,
            )
        )
        sections.append(
            run_e2e_sum_baseline(
                output_root,
                e2e_artifact_root=resolved_e2e_root,
                single_op_artifact_root=resolved_single_op_root,
                prediction_source=PREDICTION_SOURCE_SINGLE_FAIR,
            )
        )
        if not skip_timelines:
            sections.append(
                run_timeline_cases(
                    output_root,
                    single_op_artifact_root=resolved_single_op_root,
                    prediction_source=PREDICTION_SOURCE_SINGLE_FAIR,
                )
            )

    sections.append(build_figures_catalog(output_root))
    sections.append(
        build_report_draft(
            output_root,
            draft_path=draft_path,
            single_op_artifact_root=resolved_single_op_root,
        )
    )

    manifest_path = build_run_manifest(
        output_root=output_root,
        sections=sections,
        single_op_artifact_root=resolved_single_op_root,
        e2e_artifact_root=resolved_e2e_root,
    )
    return {
        "manifest_path": str(manifest_path),
        "sections": _section_results_to_json(sections),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CPU single-only report suite from CPU_Perf_Model artifacts.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--draft-path", type=Path, default=DEFAULT_REPORT_DRAFT_PATH)
    parser.add_argument("--single-op-artifact-root", type=Path, default=None)
    parser.add_argument("--e2e-artifact-root", type=Path, default=None)
    parser.add_argument("--skip-ood", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-timelines", action="store_true")
    parser.add_argument("--only", choices=ONLY_CHOICES, default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_cpu_single_only_report(
        output_root=args.output_root,
        draft_path=args.draft_path,
        single_op_artifact_root=args.single_op_artifact_root,
        e2e_artifact_root=args.e2e_artifact_root,
        skip_ood=args.skip_ood,
        skip_ablation=args.skip_ablation,
        skip_timelines=args.skip_timelines,
        only=args.only,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
