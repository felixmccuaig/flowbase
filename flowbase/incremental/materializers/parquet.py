"""Parquet-backed incremental materializers."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import pandas as pd

from flowbase.incremental.materializers.base import IncrementalMaterializer, MaterializationResult


def _normalize_columns(values: Sequence[str] | None) -> List[str]:
    return [str(v) for v in (values or [])]


def _partition_dir(root: Path, partition_column: str, value: Any) -> Path:
    return root / f"{partition_column}={value}"


def _read_partitioned_dataset(root: Path, partition_column: str) -> pd.DataFrame:
    files = list(root.glob(f"{partition_column}=*/*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat((pd.read_parquet(file) for file in files), ignore_index=True)


def _drop_duplicate_keys(df: pd.DataFrame, primary_key: Sequence[str]) -> pd.DataFrame:
    if not primary_key or df.empty:
        return df
    return df.drop_duplicates(subset=list(primary_key), keep="last").reset_index(drop=True)


class PartitionReplaceMaterializer(IncrementalMaterializer):
    """Replace touched partitions while leaving untouched partitions intact."""

    strategy_name = "partition_replace"

    def execute(
        self,
        df: pd.DataFrame,
        *,
        output_path: str,
        partition_column: str,
        sort_by: Sequence[str] | None = None,
    ) -> MaterializationResult:
        root = Path(output_path)
        root.mkdir(parents=True, exist_ok=True)
        sort_columns = _normalize_columns(sort_by)
        work = df.copy()
        touched: List[str] = []

        for raw_value in work[partition_column].drop_duplicates().tolist():
            value = str(raw_value)
            touched.append(value)
            part_dir = _partition_dir(root, partition_column, value)
            if part_dir.exists():
                shutil.rmtree(part_dir)
            part_dir.mkdir(parents=True, exist_ok=True)
            part_df = work[work[partition_column] == raw_value].copy()
            if sort_columns:
                part_df = part_df.sort_values(sort_columns)
            part_df.to_parquet(part_dir / "data.parquet", index=False)

        return MaterializationResult(
            strategy=self.strategy_name,
            output_path=str(root),
            rows_written=len(work),
            partitions_touched=sorted(touched),
            metadata={"partition_column": partition_column},
        )


class KeyUpsertMaterializer(IncrementalMaterializer):
    """Upsert rows by key, optionally scoped to partitions."""

    strategy_name = "key_upsert"

    def execute(
        self,
        df: pd.DataFrame,
        *,
        output_path: str,
        primary_key: Sequence[str],
        partition_column: str | None = None,
        sort_by: Sequence[str] | None = None,
    ) -> MaterializationResult:
        root = Path(output_path)
        key_columns = _normalize_columns(primary_key)
        sort_columns = _normalize_columns(sort_by)
        incoming = _drop_duplicate_keys(df.copy(), key_columns)

        if partition_column:
            root.mkdir(parents=True, exist_ok=True)
            existing = _read_partitioned_dataset(root, partition_column)
            touched_values = incoming[partition_column].drop_duplicates().tolist()
            existing_touched = (
                existing[existing[partition_column].isin(touched_values)].copy()
                if not existing.empty else pd.DataFrame(columns=incoming.columns)
            )
            merged = self._merge(existing_touched, incoming, key_columns)

            replacer = PartitionReplaceMaterializer()
            result = replacer.execute(
                merged,
                output_path=str(root),
                partition_column=partition_column,
                sort_by=sort_columns,
            )
            result.strategy = self.strategy_name
            result.metadata["primary_key"] = key_columns
            return result

        target = root if root.suffix == ".parquet" else (root / "data.parquet")
        target.parent.mkdir(parents=True, exist_ok=True)
        existing = pd.read_parquet(target) if target.exists() else pd.DataFrame(columns=incoming.columns)
        merged = self._merge(existing, incoming, key_columns)
        if sort_columns:
            merged = merged.sort_values(sort_columns)
        merged.to_parquet(target, index=False)
        return MaterializationResult(
            strategy=self.strategy_name,
            output_path=str(target),
            rows_written=len(merged),
            partitions_touched=[],
            metadata={"primary_key": key_columns},
        )

    def _merge(
        self,
        existing: pd.DataFrame,
        incoming: pd.DataFrame,
        primary_key: Sequence[str],
    ) -> pd.DataFrame:
        if existing.empty:
            return _drop_duplicate_keys(incoming, primary_key)
        if not primary_key:
            return incoming.reset_index(drop=True)

        incoming_index = set(tuple(row) for row in incoming[list(primary_key)].itertuples(index=False, name=None))
        preserved = existing[
            ~existing[list(primary_key)].apply(lambda row: tuple(row) in incoming_index, axis=1)
        ].copy()
        return _drop_duplicate_keys(pd.concat([preserved, incoming], ignore_index=True), primary_key)
