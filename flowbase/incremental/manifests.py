"""Generic change manifest contracts and helpers for incremental pipelines."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

try:
    import boto3
except ModuleNotFoundError:  # pragma: no cover
    boto3 = None


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_text(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def is_s3_uri(value: str) -> bool:
    return value.startswith("s3://")


def split_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


@dataclass(frozen=True)
class ManifestInput:
    normalized_uri: str
    content_hash: str
    record_count: int
    raw_uri: str | None = None


@dataclass(frozen=True)
class ChangeRecord:
    source: str
    change_type: str
    grain: str
    primary_key: dict[str, Any]
    partition_keys: dict[str, Any]
    entity_keys: dict[str, Any]
    race_key: dict[str, Any] | None = None
    event_time: str | None = None
    observed_at: str | None = None
    version: str | None = None
    idempotency_key: str | None = None
    payload_ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return {key: value for key, value in data.items() if value is not None}


@dataclass(frozen=True)
class ChangeManifest:
    manifest_version: str
    pipeline: str
    source: str
    normalized_dataset: str
    run_id: str
    produced_at: str
    mode: str
    input: ManifestInput
    changes: list[ChangeRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "pipeline": self.pipeline,
            "source": self.source,
            "normalized_dataset": self.normalized_dataset,
            "run_id": self.run_id,
            "produced_at": self.produced_at,
            "mode": self.mode,
            "input": asdict(self.input),
            "changes": [change.to_dict() for change in self.changes],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChangeManifest":
        manifest_input = ManifestInput(**dict(data["input"]))
        changes = [ChangeRecord(**dict(change)) for change in data.get("changes", [])]
        return cls(
            manifest_version=str(data["manifest_version"]),
            pipeline=str(data["pipeline"]),
            source=str(data["source"]),
            normalized_dataset=str(data["normalized_dataset"]),
            run_id=str(data["run_id"]),
            produced_at=str(data["produced_at"]),
            mode=str(data["mode"]),
            input=manifest_input,
            changes=changes,
        )


@dataclass(frozen=True)
class ManifestEvent:
    manifest_version: str
    pipeline: str
    source: str
    normalized_dataset: str
    run_id: str
    produced_at: str
    mode: str
    manifest_uri: str
    change_count: int
    partition_values: list[str]
    workflow_name: str
    workflow_config_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_manifest(manifest: ChangeManifest, output_path: Path) -> dict[str, Any]:
    payload = manifest.to_dict()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_manifest(manifest_uri: str | Path) -> dict[str, Any]:
    manifest_uri = str(manifest_uri)
    if is_s3_uri(manifest_uri):
        if boto3 is None:  # pragma: no cover
            raise RuntimeError("boto3 is required to load S3 manifests")
        bucket, key = split_s3_uri(manifest_uri)
        body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
        return json.loads(body)
    return json.loads(Path(manifest_uri).read_text(encoding="utf-8"))


def upload_manifest(local_path: Path, bucket: str, key: str) -> str:
    if boto3 is None:  # pragma: no cover
        raise RuntimeError("boto3 is required to upload manifests")
    boto3.client("s3").upload_file(str(local_path), bucket, key)
    return f"s3://{bucket}/{key}"


def extract_partition_values(
    manifest: dict[str, Any],
    partition_field: str,
    container_keys: Iterable[str] = ("primary_key", "partition_keys", "race_key"),
) -> list[str]:
    values: set[str] = set()
    for change in manifest.get("changes", []):
        for container_key in container_keys:
            container = change.get(container_key) or {}
            field_value = container.get(partition_field)
            if field_value:
                values.add(str(field_value))
    return sorted(values)


def build_manifest_event(
    *,
    manifest: ChangeManifest | dict[str, Any],
    manifest_uri: str,
    workflow_name: str,
    workflow_config_path: str,
    partition_values: list[str] | None = None,
) -> ManifestEvent:
    payload = manifest.to_dict() if isinstance(manifest, ChangeManifest) else manifest
    return ManifestEvent(
        manifest_version=str(payload["manifest_version"]),
        pipeline=str(payload["pipeline"]),
        source=str(payload["source"]),
        normalized_dataset=str(payload["normalized_dataset"]),
        run_id=str(payload["run_id"]),
        produced_at=str(payload["produced_at"]),
        mode=str(payload["mode"]),
        manifest_uri=manifest_uri,
        change_count=len(payload.get("changes", [])),
        partition_values=sorted({str(value) for value in (partition_values or [])}),
        workflow_name=workflow_name,
        workflow_config_path=workflow_config_path,
    )


def publish_manifest_event(
    *,
    manifest_event: ManifestEvent,
    event_bus_name: str | None = None,
    source: str = "flowbase.manifest",
    detail_type: str = "change_manifest.created",
) -> dict[str, Any]:
    if boto3 is None:  # pragma: no cover
        raise RuntimeError("boto3 is required to publish manifest events")
    entry: dict[str, Any] = {
        "Source": source,
        "DetailType": detail_type,
        "Detail": json.dumps(manifest_event.to_dict()),
    }
    if event_bus_name:
        entry["EventBusName"] = event_bus_name
    response = boto3.client("events").put_events(Entries=[entry])
    if int(response.get("FailedEntryCount", 0)):
        raise RuntimeError(f"Failed to publish manifest event: {json.dumps(response, default=str)}")
    return response
