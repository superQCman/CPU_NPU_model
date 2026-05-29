from __future__ import annotations

from pathlib import Path

from .paths import (
    DEFAULT_ANALYTICAL_ROOT,
    DEFAULT_CLASSED_MODEL_ROOT,
    DEFAULT_PIPELINE_ROOT,
    DEFAULT_REFERENCE_BASELINE_ROOT,
    DEFAULT_REPORT_DATA_SYNC_PATH,
    DEFAULT_REPORT_DRAFT_PATH,
    DEFAULT_REPORT_ROOT,
    ORT_ROOT,
)


CPU_REPORT_OUTPUT_ROOT = DEFAULT_REPORT_ROOT
CPU_REPORT_DRAFT_PATH = DEFAULT_REPORT_DRAFT_PATH
CPU_REPORT_DATA_SYNC_PATH = DEFAULT_REPORT_DATA_SYNC_PATH
CPU_REPORT_SINGLE_ONLY_DATA_SYNC_PATH = CPU_REPORT_DATA_SYNC_PATH

SINGLE_OP_ARTIFACT_ROOT = DEFAULT_CLASSED_MODEL_ROOT
E2E_ARTIFACT_ROOT = DEFAULT_PIPELINE_ROOT
OOD_ARTIFACT_ROOT = DEFAULT_ANALYTICAL_ROOT
ABLATION_ARTIFACT_ROOT = DEFAULT_PIPELINE_ROOT
BASELINE_MODEL_ROOT = DEFAULT_REFERENCE_BASELINE_ROOT

FAIR_SINGLE_MLP_HIDDEN_LAYERS = (128, 128, 128, 128, 128, 128)
FAIR_SINGLE_MLP_MAX_ITER = 200
FAIR_SINGLE_MLP_BATCH_SIZE = 1024
FAIR_SINGLE_MLP_ALPHA = 1e-4
FAIR_SINGLE_MLP_LEARNING_RATE_INIT = 1e-3
FAIR_SINGLE_MLP_SEED = 42

REPORT_FIGURE_FONT_SCALE = 2.0
CPU_REPORT_FIGURE_FONT_SCALE = REPORT_FIGURE_FONT_SCALE

OOD_BATCH_HOLDS = (1856, 1920, 1984, 2016, 2048)
OOD_NUM_THREADS_HOLD = 3

REPRESENTATIVE_OP_TYPES = ("Gather", "ReduceSum", "Transpose", "Concat", "Gemm")

TIMELINE_CASES: tuple[tuple[str, str], ...] = (
    ("case_10_3_3", "bs2048_nip2000"),
    ("case_10_4_6", "bs1856_nip1700"),
    ("case_8_1_1", "bs2048_nip2000"),
)

MODEL_GROUP_ORDER = ("gather", "layout_move", "view_meta", "mixed_balanced", "compute_dominant")

SECTION_ORDER = (
    "platform",
    "single_op_core",
    "single_op_ood",
    "single_op_ablation",
    "e2e_core",
    "e2e_sum_baseline",
    "timelines",
    "figures",
    "draft",
)

ONLY_CHOICES = ("all", "single_op", "e2e")

TABLE_FILENAMES = {
    "4-1": "table_4_1_platform_dataset.csv",
    "4-2": "table_4_2_single_op_group_metrics.csv",
    "4-3": "table_4_3_single_op_optype_metrics.csv",
    "4-4": "table_4_4_single_op_representative_ops.csv",
    "4-5": "table_4_5_e2e_static_summary.csv",
    "4-6": "table_4_6_e2e_sum_baseline.csv",
    "4-7": "table_4_7_single_op_ablation_summary.csv",
}

FIGURE_FILENAMES = {
    "4-1": "fig_4_1_platform_dataset_overview.png",
    "4-2": "fig_4_2_single_op_group_metrics.png",
    "4-3": "fig_4_3_single_op_group_baseline_compare.png",
    "4-4": "fig_4_4_single_op_optype_metrics.png",
    "4-5": "fig_4_5_single_op_representative_graph.png",
    "4-6": "fig_4_6_single_op_prediction_scatter.png",
    "4-7": "fig_4_7_single_op_training_history.png",
    "4-8": "fig_4_8_single_op_residual_distribution.png",
    "4-9": "fig_4_9_single_op_ood_slices.png",
    "4-10": "fig_4_10_single_op_ood_generalization.png",
    "4-11": "fig_4_11_e2e_predicted_vs_actual.png",
    "4-12": "fig_4_12_e2e_error_heatmap.png",
    "4-13": "fig_4_13_e2e_error_distribution.png",
    "4-14": "fig_4_14_timeline_cases.png",
    "4-15": "fig_4_15_critical_path_breakdown.png",
    "4-16": "fig_4_16_ablation_delta_bars.png",
    "4-17": "fig_4_17_ablation_delta_heatmap.png",
    "4-18": "fig_4_18_sum_baseline_compare.png",
    "4-19": "fig_4_19_best_ablation_summary.png",
}

REPRESENTATIVE_PANEL_FILENAME = "fig_4_5_single_op_representative_panel.png"


def output_subdir(root: Path, name: str) -> Path:
    return Path(root) / name


DEFAULT_ORT_ROOT = ORT_ROOT
