"""Tests for rollup runner functionality."""

import gzip
import io
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List

import boto3
import pandas as pd
import pytest
import yaml
from moto import mock_aws

from flowbase.rollup.runner import RollupRunner


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def s3_mock():
    """Create mock S3 environment."""
    with mock_aws():
        s3_client = boto3.client('s3', region_name='us-east-1')
        s3_client.create_bucket(Bucket='test-bucket')
        yield s3_client


def create_test_jsonl_files(temp_dir: Path, file_count: int, records_per_file: int) -> List[Path]:
    """Create test JSONL files with known data."""
    files = []
    for i in range(file_count):
        file_path = temp_dir / f"test_data_{i}.jsonl"
        with open(file_path, 'w') as f:
            for j in range(records_per_file):
                record = {
                    "id": f"file_{i}_record_{j}",
                    "timestamp": f"2025-01-01T{i:02d}:{j:02d}:00Z",
                    "value": i * records_per_file + j,
                    "file_index": i,
                    "record_index": j
                }
                f.write(json.dumps(record) + '\n')
        files.append(file_path)
    return files


def create_test_csv_files(temp_dir: Path, file_count: int, records_per_file: int) -> List[Path]:
    """Create test CSV files with known data."""
    files = []
    for i in range(file_count):
        file_path = temp_dir / f"test_data_{i}.csv"
        records = []
        for j in range(records_per_file):
            records.append({
                "id": f"file_{i}_record_{j}",
                "timestamp": f"2025-01-01T{i:02d}:{j:02d}:00Z",
                "value": i * records_per_file + j,
                "file_index": i,
                "record_index": j
            })
        df = pd.DataFrame(records)
        df.to_csv(file_path, index=False)
        files.append(file_path)
    return files


def create_test_parquet_files(temp_dir: Path, file_count: int, records_per_file: int) -> List[Path]:
    """Create test Parquet files with known data."""
    files = []
    for i in range(file_count):
        file_path = temp_dir / f"test_data_{i}.parquet"
        records = []
        for j in range(records_per_file):
            records.append({
                "id": f"file_{i}_record_{j}",
                "timestamp": f"2025-01-01T{i:02d}:{j:02d}:00Z",
                "value": i * records_per_file + j,
                "file_index": i,
                "record_index": j
            })
        df = pd.DataFrame(records)
        df.to_parquet(file_path, index=False)
        files.append(file_path)
    return files


def create_compressed_jsonl_file(temp_dir: Path, record_count: int) -> Path:
    """Create a gzip compressed JSONL file."""
    file_path = temp_dir / "compressed.jsonl.gz"
    with gzip.open(file_path, 'wt') as f:
        for i in range(record_count):
            record = {
                "id": f"compressed_record_{i}",
                "timestamp": f"2025-01-01T00:{i:02d}:00Z",
                "value": i,
                "compressed": True
            }
            f.write(json.dumps(record) + '\n')
    return file_path


class TestRollupRunnerBasic:
    """Test basic rollup runner functionality."""

    def test_initialization(self):
        """Test rollup runner initialization."""
        runner = RollupRunner()
        assert runner.s3_bucket is None
        assert runner.s3_prefix == ""
        assert runner.temp_dir.exists()

    def test_initialization_with_params(self):
        """Test rollup runner initialization with parameters."""
        runner = RollupRunner(
            s3_bucket="test-bucket",
            s3_prefix="test-prefix/",
            temp_dir="/tmp/test"
        )
        assert runner.s3_bucket == "test-bucket"
        assert runner.s3_prefix == "test-prefix/"
        assert str(runner.temp_dir) == "/tmp/test"

    def test_template_substitution_simple(self):
        """Test simple template variable substitution."""
        runner = RollupRunner()

        text = "data/{{ year }}/{{ month }}/{{ day }}/file.jsonl"
        params = {"year": "2025", "month": "01", "day": "15"}

        result = runner._substitute_templates(text, params)
        assert result == "data/2025/01/15/file.jsonl"

    def test_template_substitution_nested_config(self):
        """Test template substitution in nested config structures."""
        runner = RollupRunner()

        config = {
            "source": {
                "bucket": "test-bucket",
                "prefix": "data/{{ year }}/{{ month }}/",
                "pattern": "*.jsonl"
            },
            "target": {
                "bucket": "test-bucket",
                "key": "output/{{ year }}-{{ month }}.parquet"
            },
            "list_param": ["item_{{ year }}", "item_{{ month }}"]
        }

        params = {"year": "2025", "month": "01"}
        result = runner._substitute_templates_in_config(config, params)

        assert result["source"]["prefix"] == "data/2025/01/"
        assert result["target"]["key"] == "output/2025-01.parquet"
        assert result["list_param"] == ["item_2025", "item_01"]


class TestRollupRunnerSimple:
    """Test simple rollup operations."""

    def test_simple_rollup_jsonl_to_parquet(self, temp_dir, s3_mock):
        """Test rolling up JSONL files to Parquet format."""
        # Create test files (3 files, 10 records each = 30 total)
        source_files = create_test_jsonl_files(temp_dir, 3, 10)

        # Upload to mock S3
        for file_path in source_files:
            s3_mock.upload_file(str(file_path), 'test-bucket', f"source/{file_path.name}")

        # Configure rollup
        config = {
            "rollup_type": "simple",
            "source": {
                "bucket": "test-bucket",
                "prefix": "source/",
                "pattern": "*.jsonl",
                "format": "jsonl"
            },
            "target": {
                "bucket": "test-bucket",
                "key": "target/rollup.parquet",
                "format": "parquet"
            }
        }

        # Run rollup
        runner = RollupRunner(s3_bucket='test-bucket')
        result = runner.run(config_path=None, config_dict=config)

        # Verify results
        assert result["type"] == "rollup"
        assert result["source_files"] == 3
        assert result["rows_processed"] == 30
        assert result["output_format"] == "parquet"
        assert "target_s3_url" in result

        # Verify output file exists and has correct data
        response = s3_mock.get_object(Bucket='test-bucket', Key='target/rollup.parquet')
        output_df = pd.read_parquet(io.BytesIO(response['Body'].read()))

        assert len(output_df) == 30

        # Verify all records preserved with correct data
        expected_ids = {f"file_{i}_record_{j}" for i in range(3) for j in range(10)}
        actual_ids = set(output_df['id'])
        assert actual_ids == expected_ids

        # Verify other fields
        assert output_df['value'].tolist() == list(range(30))
        assert all(output_df['file_index'] >= 0)
        assert all(output_df['record_index'] >= 0)

    def test_simple_rollup_csv_to_jsonl(self, temp_dir, s3_mock):
        """Test rolling up CSV files to JSONL format."""
        # Create test CSV files
        source_files = create_test_csv_files(temp_dir, 2, 5)  # 2 files, 5 records each = 10 total

        # Upload to mock S3
        for file_path in source_files:
            s3_mock.upload_file(str(file_path), 'test-bucket', f"source/{file_path.name}")

        config = {
            "rollup_type": "simple",
            "source": {
                "bucket": "test-bucket",
                "prefix": "source/",
                "pattern": "*.csv",
                "format": "csv"
            },
            "target": {
                "bucket": "test-bucket",
                "key": "target/rollup.jsonl",
                "format": "jsonl"
            }
        }

        runner = RollupRunner(s3_bucket='test-bucket')
        result = runner.run(config_path=None, config_dict=config)

        assert result["source_files"] == 2
        assert result["rows_processed"] == 10

        # Verify JSONL output
        response = s3_mock.get_object(Bucket='test-bucket', Key='target/rollup.jsonl')
        content = response['Body'].read().decode('utf-8')
        lines = content.strip().split('\n')
        assert len(lines) == 10

        # Verify first record
        first_record = json.loads(lines[0])
        assert "id" in first_record
        assert "value" in first_record
        assert first_record["value"] == 0

    def test_simple_rollup_parquet_to_csv(self, temp_dir, s3_mock):
        """Test rolling up Parquet files to CSV format."""
        # Create test Parquet files
        source_files = create_test_parquet_files(temp_dir, 2, 3)  # 2 files, 3 records each = 6 total

        # Upload to mock S3
        for file_path in source_files:
            s3_mock.upload_file(str(file_path), 'test-bucket', f"source/{file_path.name}")

        config = {
            "rollup_type": "simple",
            "source": {
                "bucket": "test-bucket",
                "prefix": "source/",
                "pattern": "*.parquet",
                "format": "parquet"
            },
            "target": {
                "bucket": "test-bucket",
                "key": "target/rollup.csv",
                "format": "csv"
            }
        }

        runner = RollupRunner(s3_bucket='test-bucket')
        result = runner.run(config_path=None, config_dict=config)

        assert result["source_files"] == 2
        assert result["rows_processed"] == 6

        # Verify CSV output
        response = s3_mock.get_object(Bucket='test-bucket', Key='target/rollup.csv')
        content = response['Body'].read().decode('utf-8')
        lines = content.strip().split('\n')
        assert len(lines) == 7  # 6 data rows + 1 header

        # Verify header contains expected columns (order may vary)
        header = lines[0]
        assert "id" in header
        assert "value" in header
        assert "timestamp" in header
        assert "file_index" in header
        assert "record_index" in header

    def test_gzip_compression_handling(self, temp_dir, s3_mock):
        """Test that gzip compressed files are properly handled."""
        # Create gzip compressed file
        compressed_file = create_compressed_jsonl_file(temp_dir, 50)
        s3_mock.upload_file(str(compressed_file), 'test-bucket', 'source/compressed.jsonl.gz')

        config = {
            "rollup_type": "simple",
            "source": {
                "bucket": "test-bucket",
                "prefix": "source/",
                "pattern": "*.gz",
                "format": "jsonl",
                "compression": "gzip"
            },
            "target": {
                "bucket": "test-bucket",
                "key": "target/output.parquet",
                "format": "parquet"
            }
        }

        runner = RollupRunner(s3_bucket='test-bucket')
        result = runner.run(config_path=None, config_dict=config)

        assert result["rows_processed"] == 50

        # Verify output
        response = s3_mock.get_object(Bucket='test-bucket', Key='target/output.parquet')
        output_df = pd.read_parquet(io.BytesIO(response['Body'].read()))
        assert len(output_df) == 50
        assert all(output_df['compressed'] == True)

    def test_output_compression(self, temp_dir, s3_mock):
        """Test output file compression."""
        # Create test files
        source_files = create_test_jsonl_files(temp_dir, 1, 20)

        # Upload to mock S3
        for file_path in source_files:
            s3_mock.upload_file(str(file_path), 'test-bucket', f"source/{file_path.name}")

        config = {
            "rollup_type": "simple",
            "source": {
                "bucket": "test-bucket",
                "prefix": "source/",
                "pattern": "*.jsonl",
                "format": "jsonl"
            },
            "target": {
                "bucket": "test-bucket",
                "key": "target/rollup.jsonl",
                "format": "jsonl",
                "compression": "gzip"
            }
        }

        runner = RollupRunner(s3_bucket='test-bucket')
        result = runner.run(config_path=None, config_dict=config)

        assert result["rows_processed"] == 20
        assert result["compression"] == "gzip"

        # Verify compressed output exists (check the actual target key from result)
        target_key = result["target_key"]
        response = s3_mock.get_object(Bucket='test-bucket', Key=target_key)
        compressed_data = response['Body'].read()

        # Decompress and verify
        with gzip.open(io.BytesIO(compressed_data), 'rt') as f:
            content = f.read()
            lines = content.strip().split('\n')
            assert len(lines) == 20


class TestRollupRunnerHierarchical:
    """Test hierarchical rollup operations."""

    def test_hierarchical_rollup_daily_stage(self, temp_dir, s3_mock):
        """Test daily hierarchical rollup stage."""
        # Create hourly files for a day (simulate 3 hours for testing)
        hourly_files = []
        total_records = 0

        for hour in range(3):
            file_path = temp_dir / f"hour_{hour:02d}.jsonl"
            with open(file_path, 'w') as f:
                for record in range(5):  # 5 records per hour
                    data = {
                        "id": f"2025-01-15-{hour:02d}-{record}",
                        "hour": hour,
                        "day": 15,
                        "month": 1,
                        "year": 2025,
                        "value": hour * 5 + record
                    }
                    f.write(json.dumps(data) + '\n')
                    total_records += 1

            hourly_files.append(file_path)
            # Upload to S3 with hierarchical path structure
            s3_mock.upload_file(str(file_path), 'test-bucket', f"trades/2025/01/15/data_{hour:02d}.jsonl")

        config = {
            "rollup_type": "hierarchical",
            "stage": "daily",
            "source": {
                "bucket": "test-bucket",
                "prefix": "trades/{{ year }}/{{ month }}/{{ day }}/",
                "pattern": "*.jsonl",  # Simple pattern to find all files
                "format": "jsonl"
            },
            "target": {
                "bucket": "test-bucket",
                "key": "daily/{{ year }}/{{ month }}/{{ day }}.parquet",
                "format": "parquet"
            }
        }

        runner = RollupRunner(s3_bucket='test-bucket')
        result = runner.run(
            config_path=None,
            config_dict=config,
            params={"year": "2025", "month": "01", "day": "15"}
        )

        # Verify all records are in output (3 hours * 5 records = 15)
        assert result["rows_processed"] == 15
        assert result["stage"] == "daily"

        # Verify output file
        response = s3_mock.get_object(Bucket='test-bucket', Key='daily/2025/01/15.parquet')
        output_df = pd.read_parquet(io.BytesIO(response['Body'].read()))
        assert len(output_df) == 15

        # Verify data integrity
        assert set(output_df['hour']) == {0, 1, 2}
        assert all(output_df['day'] == 15)
        assert all(output_df['month'] == 1)
        assert all(output_df['year'] == 2025)

    def test_hierarchical_rollup_template_substitution(self, temp_dir, s3_mock):
        """Test template variable substitution in hierarchical rollup."""
        # Create a test file
        file_path = temp_dir / "test.jsonl"
        with open(file_path, 'w') as f:
            for i in range(10):
                data = {
                    "id": f"test_record_{i}",
                    "custom_field": "test_value_{{ year }}"
                }
                f.write(json.dumps(data) + '\n')

        s3_mock.upload_file(str(file_path), 'test-bucket', 'source/test.jsonl')

        config = {
            "rollup_type": "hierarchical",
            "stage": "monthly",
            "source": {
                "bucket": "test-bucket",
                "prefix": "source/",
                "pattern": "*.jsonl",
                "format": "jsonl"
            },
            "target": {
                "bucket": "test-bucket",
                "key": "monthly/{{ year }}/{{ month }}.parquet",
                "format": "parquet"
            },
            "post_process": {
                "archive_source": True,
                "archive_location": "s3://test-bucket/archive/{{ year }}/{{ month }}.tar.gz"
            }
        }

        runner = RollupRunner(s3_bucket='test-bucket')
        result = runner.run(
            config_path=None,
            config_dict=config,
            params={"year": "2025", "month": "02"}
        )

        # Verify template substitution worked
        assert result["rows_processed"] == 10

        # Verify output file has correct path
        response = s3_mock.get_object(Bucket='test-bucket', Key='monthly/2025/02.parquet')
        output_df = pd.read_parquet(io.BytesIO(response['Body'].read()))
        assert len(output_df) == 10

    def test_hierarchical_rollup_unknown_stage(self):
        """Test error handling for unknown hierarchical stage."""
        config = {
            "rollup_type": "hierarchical",
            "stage": "unknown_stage",
            "source": {
                "bucket": "test-bucket",
                "prefix": "source/",
                "pattern": "*.jsonl",
                "format": "jsonl"
            },
            "target": {
                "bucket": "test-bucket",
                "key": "target/output.parquet",
                "format": "parquet"
            }
        }

        runner = RollupRunner()
        result = runner.run(config_path=None, config_dict=config)

        assert "error" in result
        assert "unknown_stage" in result["error"]

    def test_recursive_glob_pattern_matching(self, temp_dir, s3_mock):
        """Test that **/*.gz pattern correctly matches files in subdirectories (simulates hourly data structure)."""
        # Create test files in subdirectories (simulating time-based partitioning)
        test_files = []

        # Create files in different subdirectories
        for hour in ["00", "01", "02"]:
            subdir = temp_dir / hour
            subdir.mkdir(exist_ok=True)

            file_path = subdir / f"data-stream-1-2025-11-15-{hour}-test.gz"
            with gzip.open(file_path, 'wt') as f:
                f.write('{"id": "test_record", "hour": "' + hour + '", "value": 1}\n')
            test_files.append(file_path)

            # Upload to S3 with the expected structure
            s3_key = f"data/2025/11/15/{hour}/data-stream-1-2025-11-15-{hour}-test.gz"
            s3_mock.upload_file(str(file_path), 'test-bucket', s3_key)

        # Test the rollup with **/*.gz pattern
        config = {
            "rollup_type": "hierarchical",
            "stage": "daily",
            "source": {
                "bucket": "test-bucket",
                "prefix": "data/2025/11/15/",
                "pattern": "**/*.gz",  # This should now work with the fixed glob matching
                "format": "jsonl",
                "compression": "gzip"
            },
            "target": {
                "bucket": "test-bucket",
                "key": "data/processed/daily/data_2025_11_15.parquet",
                "format": "parquet"
            }
        }

        runner = RollupRunner(s3_bucket='test-bucket')
        result = runner.run(config_path=None, config_dict=config)

        # Verify the rollup found and processed all 3 files
        assert result["source_files"] == 3
        assert result["rows_processed"] == 3
        assert result["stage"] == "daily"

        # Verify output file contains data from all hours
        response = s3_mock.get_object(Bucket='test-bucket', Key='data/processed/daily/data_2025_11_15.parquet')
        output_df = pd.read_parquet(io.BytesIO(response['Body'].read()))

        assert len(output_df) == 3
        assert set(output_df['hour']) == {"00", "01", "02"}
        assert all(output_df['id'] == 'test_record')


class TestRollupRunnerErrorHandling:
    """Test error handling in rollup operations."""

    def test_no_source_files_found(self, s3_mock):
        """Test handling when no source files match pattern."""
        config = {
            "rollup_type": "simple",
            "source": {
                "bucket": "test-bucket",
                "prefix": "nonexistent/",
                "pattern": "*.jsonl",
                "format": "jsonl"
            },
            "target": {
                "bucket": "test-bucket",
                "key": "target/output.parquet",
                "format": "parquet"
            }
        }

        runner = RollupRunner(s3_bucket='test-bucket')
        result = runner.run(config_path=None, config_dict=config)

        assert result["source_files"] == 0
        assert result["status"] == "no_files"

    def test_invalid_output_format(self, temp_dir, s3_mock):
        """Test error handling for invalid output format."""
        # Create and upload test file
        file_path = temp_dir / "test.jsonl"
        with open(file_path, 'w') as f:
            f.write('{"id": "test", "value": 1}\n')

        s3_mock.upload_file(str(file_path), 'test-bucket', 'source/test.jsonl')

        config = {
            "rollup_type": "simple",
            "source": {
                "bucket": "test-bucket",
                "prefix": "source/",
                "pattern": "*.jsonl",
                "format": "jsonl"
            },
            "target": {
                "bucket": "test-bucket",
                "key": "target/output.invalid",
                "format": "invalid_format"
            }
        }

        runner = RollupRunner(s3_bucket='test-bucket')

        # Should raise ValueError for invalid format
        with pytest.raises(ValueError, match="Unsupported output format"):
            runner.run(config_path=None, config_dict=config)

    def test_s3_access_error(self):
        """Test handling of S3 access errors."""
        config = {
            "rollup_type": "simple",
            "source": {
                "bucket": "nonexistent-bucket",
                "prefix": "source/",
                "pattern": "*.jsonl",
                "format": "jsonl"
            },
            "target": {
                "bucket": "nonexistent-bucket",
                "key": "target/output.parquet",
                "format": "parquet"
            }
        }

        runner = RollupRunner(s3_bucket='nonexistent-bucket')

        # Should raise an exception for S3 access issues
        with pytest.raises(Exception):  # Could be ClientError or other S3-related exception
            runner.run(config_path=None, config_dict=config)


class TestRollupRunnerIntegration:
    """Integration tests for complete rollup workflows."""

    def test_end_to_end_workflow_simulation(self, temp_dir, s3_mock):
        """Test a complete rollup workflow simulation with multiple hourly files."""
        # Simulate a full day's worth of hourly files
        total_files = 24
        records_per_file = 100
        total_records = total_files * records_per_file

        # Create and upload hourly files
        for hour in range(total_files):
            file_path = temp_dir / f"hour_{hour:02d}.jsonl"
            with open(file_path, 'w') as f:
                for record in range(records_per_file):
                    data = {
                        "id": f"2025-01-15-{hour:02d}-{record:04d}",
                        "timestamp": f"2025-01-15T{hour:02d}:{record % 60:02d}:{record // 60:02d}Z",
                        "hour": hour,
                        "value": hour * records_per_file + record,
                        "source": "test_data"
                    }
                    f.write(json.dumps(data) + '\n')

            s3_mock.upload_file(str(file_path), 'test-bucket', f"data/2025/01/15/data_{hour:02d}.jsonl")

        # Run daily rollup
        config = {
            "rollup_type": "hierarchical",
            "stage": "daily",
            "source": {
                "bucket": "test-bucket",
                "prefix": "data/{{ year }}/{{ month }}/{{ day }}/",
                "pattern": "*.jsonl",
                "format": "jsonl"
            },
            "target": {
                "bucket": "test-bucket",
                "key": "daily/{{ year }}/{{ month }}/{{ day }}.parquet",
                "format": "parquet",
                "compression": "gzip"
            },
            "post_process": {
                "archive_source": True,
                "archive_location": "s3://test-bucket/archive/daily/{{ year }}/{{ month }}/{{ day }}.tar.gz"
            }
        }

        runner = RollupRunner(s3_bucket='test-bucket')
        result = runner.run(
            config_path=None,
            config_dict=config,
            params={"year": "2025", "month": "01", "day": "15"}
        )

        # Verify complete rollup
        assert result["rows_processed"] == total_records
        assert result["source_files"] == total_files
        assert result["stage"] == "daily"
        # Note: archiving is not fully implemented in this test version

        # Verify compressed output (key should include .gz extension when compression is applied)
        response = s3_mock.get_object(Bucket='test-bucket', Key='daily/2025/01/15.parquet.gz')
        compressed_data = response['Body'].read()

        # Decompress and verify data
        import gzip
        decompressed_data = gzip.decompress(compressed_data)
        with io.BytesIO(decompressed_data) as buffer:
            output_df = pd.read_parquet(buffer)

        assert len(output_df) == total_records
        assert set(output_df['hour']) == set(range(24))
        assert all(output_df['source'] == 'test_data')

        # Verify data integrity - all IDs should be unique
        assert len(output_df['id'].unique()) == total_records