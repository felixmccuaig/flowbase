from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from flowbase.incremental import (
    ChangeManifest,
    ChangeRecord,
    ManifestInput,
    build_manifest_event,
    extract_partition_values,
    load_manifest,
    write_manifest,
)


class IncrementalManifestTest(TestCase):
    def test_write_and_load_manifest_round_trip(self) -> None:
        manifest = ChangeManifest(
            manifest_version="1",
            pipeline="example",
            source="source_a",
            normalized_dataset="norm_source_a",
            run_id="run-1",
            produced_at="2026-03-23T00:00:00Z",
            mode="realtime",
            input=ManifestInput(
                normalized_uri="s3://bucket/path.parquet",
                content_hash="sha256:abc",
                record_count=2,
            ),
            changes=[
                ChangeRecord(
                    source="source_a",
                    change_type="upsert",
                    grain="entity_record",
                    primary_key={"record_id": "r1"},
                    partition_keys={"partition_date": "2026-03-23"},
                    entity_keys={"entity_id": "e1"},
                )
            ],
        )
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            write_manifest(manifest, path)
            loaded = load_manifest(path)
        self.assertEqual(loaded["source"], "source_a")
        self.assertEqual(len(loaded["changes"]), 1)

    def test_build_manifest_event_and_dates(self) -> None:
        manifest = {
            "manifest_version": "1",
            "pipeline": "example",
            "source": "source_a",
            "normalized_dataset": "norm_source_a",
            "run_id": "run-1",
            "produced_at": "2026-03-23T00:00:00Z",
            "mode": "realtime",
            "input": {"normalized_uri": "x", "content_hash": "y", "record_count": 1},
            "changes": [
                {"primary_key": {"record_id": "r1"}},
                {"partition_keys": {"partition_date": "2026-03-24"}},
            ],
        }
        event = build_manifest_event(
            manifest=manifest,
            manifest_uri="s3://bucket/manifest.json",
            workflow_name="wf",
            workflow_config_path="workflows/wf/workflow.yaml",
            partition_values=["2026-03-23", "2026-03-24"],
        )
        self.assertEqual(extract_partition_values(manifest, "partition_date"), ["2026-03-24"])
        self.assertEqual(event.manifest_uri, "s3://bucket/manifest.json")
        self.assertEqual(event.partition_values, ["2026-03-23", "2026-03-24"])
