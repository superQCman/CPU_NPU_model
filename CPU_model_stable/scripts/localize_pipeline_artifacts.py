from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


INDEX_NAME = "pipeline_artifact_index.csv"


def _copy_if_present(src_text: str, dst: Path) -> str:
    src = Path(str(src_text)).expanduser()
    if not src.is_file():
        return ""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        shutil.copy2(src, dst)
    return str(dst.resolve())


def _write_index_json(index_csv: Path, frame: pd.DataFrame) -> None:
    out = index_csv.with_suffix(".json")
    payload = {
        "row_count": int(len(frame)),
        "rows": frame.to_dict(orient="records"),
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def localize_one(index_csv: Path, artifact_root: Path) -> dict[str, int]:
    frame = pd.read_csv(index_csv)
    copied = 0
    missing = 0

    for idx, row in frame.iterrows():
        source_name = str(row.get("source_name", "")).strip() or "unknown_source"
        combo = str(row.get("combo", "")).strip() or str(idx)
        combo_root = artifact_root / source_name / combo

        shape_dst = combo_root / "op_shapes.csv"
        timeline_dst = combo_root / "branch_parallel_op_timeline.csv"

        shape_local = _copy_if_present(str(row.get("shape_csv", "")), shape_dst)
        timeline_local = _copy_if_present(str(row.get("timeline_csv", "")), timeline_dst)

        if shape_local:
            copied += 1
            frame.at[idx, "shape_csv"] = shape_local
            frame.at[idx, "shape_csv_exists"] = True
        else:
            missing += 1
            frame.at[idx, "shape_csv_exists"] = False

        if timeline_local:
            copied += 1
            frame.at[idx, "timeline_csv"] = timeline_local
            frame.at[idx, "timeline_csv_exists"] = True
            frame.at[idx, "profile_dir"] = str(timeline_dst.parent.resolve())
        else:
            missing += 1
            frame.at[idx, "timeline_csv_exists"] = False

        frame.at[idx, "sweep_dir"] = str((artifact_root / source_name).resolve())
        frame.at[idx, "log_dir"] = str((artifact_root / source_name / combo / "logs").resolve())
        if "effective_onnx_path" in frame.columns:
            frame.at[idx, "effective_onnx_path"] = ""
        if "effective_onnx_exists" in frame.columns:
            frame.at[idx, "effective_onnx_exists"] = False

    frame.to_csv(index_csv, index=False)
    _write_index_json(index_csv, frame)
    return {"rows": int(len(frame)), "copied": copied, "missing": missing}


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy referenced E2E pipeline artifacts into CPU_model_stable.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    artifact_root = args.artifact_root.resolve()
    summaries = []
    for index_csv in sorted(data_root.glob(f"source_*/{INDEX_NAME}")):
        summary = localize_one(index_csv, artifact_root / index_csv.parent.name)
        summary["index_csv"] = str(index_csv)
        summaries.append(summary)

    manifest = artifact_root / "localization_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"datasets": summaries}, indent=2), encoding="utf-8")
    print(json.dumps({"dataset_count": len(summaries), "manifest": str(manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
