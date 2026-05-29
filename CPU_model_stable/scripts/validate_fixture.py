from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_FILES = [
    "dataset_full.csv",
    "train.csv",
    "test.csv",
    "train_raw.csv",
    "test_raw.csv",
    "e2e_train.csv",
    "e2e_test_raw.csv",
    "joint_6d_tuple_audit.csv",
]


def _read_csv(root: Path, name: str) -> pd.DataFrame:
    path = root / name
    if not path.is_file():
        raise SystemExit(f"missing fixture file: {path}")
    return pd.read_csv(path)


def _combo_row_counts(frame: pd.DataFrame) -> pd.Series:
    key_cols = ["source_name", "case_id", "combo"]
    missing = [col for col in key_cols if col not in frame.columns]
    if missing:
        raise SystemExit(f"missing combo key columns in fixture: {missing}")
    return frame.groupby(key_cols, dropna=False).size()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the CPU_model_stable joint_6d_tuple fixture.")
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--expected-manifest", type=Path, required=True)
    args = parser.parse_args()

    root = args.fixture_root.resolve()
    expected = json.loads(args.expected_manifest.read_text(encoding="utf-8"))

    for name in REQUIRED_FILES:
        if not (root / name).is_file():
            raise SystemExit(f"missing required fixture file: {root / name}")

    train = _read_csv(root, "train.csv")
    test = _read_csv(root, "test.csv")
    e2e_train = _read_csv(root, "e2e_train.csv")
    e2e_test = _read_csv(root, "e2e_test_raw.csv")

    required_columns = set(expected.get("required_columns", []))
    if required_columns:
        missing = sorted(required_columns.difference(train.columns))
        if missing:
            raise SystemExit(f"fixture train.csv missing expected columns: {missing}")

    joint_col = "joint_6d_tuple_key"
    if joint_col not in train.columns or joint_col not in test.columns:
        raise SystemExit("fixture is missing joint_6d_tuple_key")
    overlap = set(train[joint_col].dropna().astype(str)) & set(test[joint_col].dropna().astype(str))
    if overlap:
        raise SystemExit(f"train/test joint tuple overlap is not zero: {len(overlap)}")

    for name, frame in [("e2e_train.csv", e2e_train), ("e2e_test_raw.csv", e2e_test)]:
        counts = _combo_row_counts(frame)
        if counts.empty:
            raise SystemExit(f"{name} has no E2E combo rows")
        if counts.nunique() != 1:
            raise SystemExit(f"{name} has non-uniform rows per combo: {counts.value_counts().to_dict()}")
        rows_per_combo = int(counts.iloc[0])
        expected_rows = int(expected.get("rows_per_e2e_combo", rows_per_combo))
        if rows_per_combo != expected_rows:
            raise SystemExit(f"{name} rows per combo={rows_per_combo}, expected={expected_rows}")

    print(
        json.dumps(
            {
                "fixture_root": str(root),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "e2e_train_rows": int(len(e2e_train)),
                "e2e_test_rows": int(len(e2e_test)),
                "joint_overlap": 0,
                "status": "ok",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
