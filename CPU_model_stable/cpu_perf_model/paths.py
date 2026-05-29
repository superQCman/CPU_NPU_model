from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
ORT_ROOT = PROJECT_ROOT.parent
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "latest"

DEFAULT_DATASET_ROOT = ARTIFACT_ROOT / "dataset_all_78910_nodrop"
DEFAULT_ANALYTICAL_ROOT = ARTIFACT_ROOT / "analytical_model"
DEFAULT_SINGLE_MODEL_ROOT = ARTIFACT_ROOT / "single_mlp_fair"
DEFAULT_CLASSED_MODEL_ROOT = ARTIFACT_ROOT / "classed_mlp_with_analytical"
DEFAULT_PIPELINE_ROOT = ARTIFACT_ROOT / "cpu_pipeline_eval"
DEFAULT_REPORT_ROOT = ARTIFACT_ROOT / "cpu_single_only_report"

DEFAULT_REPORT_DRAFT_PATH = PROJECT_ROOT / "cpu_single_only_report_draft.md"
DEFAULT_REPORT_DATA_SYNC_PATH = PROJECT_ROOT / "cpu_single_only_report_data_sync.md"

DEFAULT_REFERENCE_SINGLE_OP_ROOT = (
    ORT_ROOT
    / "single_op_stage1_mlp"
    / "artifacts"
    / "latest"
    / "classed_op_mlp_test_78910_analytical_5_300_iter_quick_nodrop"
)
DEFAULT_REFERENCE_BASELINE_ROOT = (
    ORT_ROOT / "single_op_stage1_mlp" / "artifacts" / "latest" / "model_all_no_trace"
)
