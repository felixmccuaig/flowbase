"""Rollup runner for combining and archiving data files."""

from __future__ import annotations

import gzip
import json
import logging
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class RollupRunner:
    """Runs rollup operations to combine and archive data files."""

    def __init__(
        self,
        s3_bucket: Optional[str] = None,
        s3_prefix: Optional[str] = "",
        temp_dir: Optional[str] = None,
        data_root: Optional[str] = None
    ):
        """Initialize rollup runner.

        Args:
            s3_bucket: Default S3 bucket for operations
            s3_prefix: Default S3 prefix
            temp_dir: Directory for temporary files
            data_root: Root directory override for Lambda support
        """
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix or ""
        self.temp_dir = Path(temp_dir) if temp_dir else Path("data/rollup/.temp")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.data_root = Path(data_root) if data_root else None
        self.s3_sync = None

        # Initialize S3 sync if bucket provided
        if s3_bucket:
            try:
                from flowbase.storage.s3_sync import S3Sync
                self.s3_sync = S3Sync(bucket=s3_bucket, prefix=self.s3_prefix)
            except ImportError:
                logger.warning("boto3 not installed. S3 operations disabled.")

    def run(
        self,
        config_path: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute a rollup operation.

        Args:
            config_path: Path to rollup configuration YAML
            config_dict: Configuration dictionary (for testing)
            params: Runtime parameters for templating
            **kwargs: Additional parameters

        Returns:
            Dictionary with rollup results
        """
        # Load configuration
        if config_dict:
            config = config_dict
        elif config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            raise ValueError("Either config_path or config_dict must be provided")

        # Merge parameters
        all_params = {**(params or {}), **kwargs}

        rollup_type = config.get("rollup_type", "simple")

        if rollup_type == "hierarchical":
            return self._run_hierarchical_rollup(config, all_params)
        elif rollup_type == "archive":
            return self._run_archive_operation(config, all_params)
        else:
            return self._run_simple_rollup(config, all_params)

    def _run_simple_rollup(self, config: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Run a simple rollup operation."""
        return self._execute_rollup(config, params)

    def _run_hierarchical_rollup(self, config: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Run a hierarchical rollup with archiving."""
        return self._execute_hierarchical_rollup(config, params)

    def _run_archive_operation(self, config: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Run an archive operation."""
        return self._execute_archive_operation(config, params)

    def _execute_rollup(self, config: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the core rollup logic."""
        # Extract configuration
        source_config = config.get("source", {})
        target_config = config.get("target", {})

        source_bucket = source_config.get("bucket")
        source_prefix = source_config.get("prefix", "")
        source_pattern = source_config.get("pattern", "*")
        input_format = source_config.get("format", "jsonl")

        target_bucket = target_config.get("bucket")
        target_key = target_config.get("key")
        output_format = target_config.get("format", "parquet")
        compression = target_config.get("compression")

        # Initialize S3 sync for source and target
        source_s3 = self._get_s3_sync(source_bucket, source_prefix)
        target_s3 = self._get_s3_sync(target_bucket, "")

        if not source_s3 or not target_s3:
            return {"error": "S3 sync not available"}

        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_temp_dir = temp_path / "source"
            target_temp_file = temp_path / "target"

            source_temp_dir.mkdir()

            # Download source files matching pattern
            logger.info(f"Downloading files from s3://{source_bucket}/{source_prefix} matching {source_pattern}")

            matching_keys = self._find_matching_files(source_s3, source_prefix, source_pattern)
            if not matching_keys:
                logger.warning(f"No files found matching pattern {source_pattern}")
                return {
                    "type": "rollup",
                    "source_files": 0,
                    "status": "no_files"
                }

            logger.info(f"Found {len(matching_keys)} files to rollup")

            # Download all matching files
            downloaded_files = []
            for key in matching_keys:
                local_file = source_temp_dir / Path(key).name
                if source_s3.download_file(key[len(source_prefix):].lstrip("/"), local_file):
                    downloaded_files.append(local_file)
                else:
                    logger.warning(f"Failed to download {key}")

            if not downloaded_files:
                return {"error": "Failed to download any source files"}

            # Combine files based on input format
            logger.info(f"Combining {len(downloaded_files)} files")
            combined_data = self._combine_files(downloaded_files, input_format)

            if not combined_data:
                logger.warning("No data found in source files")
                return {"error": "No data in source files"}

            # Convert to DataFrame
            df = pd.DataFrame(combined_data)

            # Clean DataFrame to handle string 'NaN' values and type conversion issues
            df = self._clean_dataframe_for_output(df)

            # Write combined data to target format
            logger.info(f"Writing combined data ({len(df)} rows) to {output_format} format")
            final_temp_file = self._write_output_file(df, target_temp_file, output_format, compression)

            # Update target key if compression was applied
            upload_key = target_key
            if compression == "gzip" and not target_key.endswith('.gz'):
                upload_key = target_key + '.gz'

            # Upload to target S3 location
            logger.info(f"Uploading rolled-up file to s3://{target_bucket}/{upload_key}")
            if target_s3.upload_file(final_temp_file, upload_key):
                s3_url = f"s3://{target_bucket}/{upload_key}"
                logger.info(f"Successfully uploaded rollup to {s3_url}")
            else:
                logger.error("Failed to upload rolled-up file to S3")
                return {"error": "Upload failed"}

            return {
                "type": "rollup",
                "source_files": len(matching_keys),
                "rows_processed": len(df),
                "output_format": output_format,
                "compression": compression,
                "target_s3_url": s3_url,
                "source_bucket": source_bucket,
                "target_bucket": target_bucket,
                "target_key": upload_key
            }

    def _execute_hierarchical_rollup(self, config: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute hierarchical rollup with archiving support."""
        stage = config.get("stage")

        # Validate stage first, before any S3 operations
        if stage not in ["daily", "monthly", "yearly"]:
            return {"error": f"Unknown hierarchical stage: {stage}"}

        strategy = config.get("strategy", {})
        post_process = config.get("post_process", {})

        # Use the source and target configs from the YAML, substituting template variables
        source_config = config.get("source", {})
        target_config = config.get("target", {})

        # Substitute template variables in source and target configs
        source_config = self._substitute_templates_in_config(source_config, params)
        target_config = self._substitute_templates_in_config(target_config, params)

        # Override the config for this execution
        rollup_config = config.copy()
        rollup_config["source"] = source_config
        rollup_config["target"] = target_config

        # Execute the rollup
        result = self._execute_rollup(rollup_config, params)

        if result.get("error"):
            return result

        # Get the source files that were processed (this would need to be returned from _execute_rollup)
        # For now, we'll need to reconstruct this information
        source_files = result.get("source_files", [])
        if isinstance(source_files, int):
            # If it's just a count, we can't get the actual file list
            # This is a limitation of the current design
            logger.warning("Cannot archive/delete source files: file list not available")
            source_files = []

        # Post-processing: archiving
        if post_process.get("archive_source", False) and source_files:
            archive_location = post_process.get("archive_location")
            if archive_location:
                # Substitute templates in archive location
                archive_location = self._substitute_templates(archive_location, params)
                self._archive_files(source_files, archive_location, params)

        if post_process.get("delete_source", False) and source_files:
            self._delete_source_files(source_files)

        result.update({
            "stage": stage,
            "archived": post_process.get("archive_source", False),
            "deleted_source": post_process.get("delete_source", False)
        })

        return result

    def _execute_archive_operation(self, config: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an archive operation."""
        archive_config = config.get("archive", {})
        source_config = config.get("source", {})

        source_bucket = source_config.get("bucket")
        source_prefix = source_config.get("prefix", "")
        source_pattern = source_config.get("pattern", "*")

        target_bucket = archive_config.get("target_bucket")
        target_prefix = archive_config.get("target_prefix", "")
        compression = archive_config.get("compression", "tar.gz")
        retention_days = archive_config.get("retention_days")

        # Find files to archive
        source_s3 = self._get_s3_sync(source_bucket, source_prefix)
        if not source_s3:
            return {"error": "S3 sync not available"}

        matching_keys = self._find_matching_files(source_s3, source_prefix, source_pattern)
        if not matching_keys:
            return {"status": "no_files_to_archive"}

        # Create archive
        archive_key = f"{target_prefix.rstrip('/')}/archive_{params.get('year', 'unknown')}_{params.get('month', 'unknown')}.tar.gz"
        target_s3 = self._get_s3_sync(target_bucket, "")

        if self._create_archive(matching_keys, source_s3, archive_key, target_s3):
            # Optionally delete source files after archiving
            if config.get("post_process", {}).get("delete_source", False):
                self._delete_source_files(matching_keys)

            return {
                "type": "archive",
                "files_archived": len(matching_keys),
                "archive_key": archive_key,
                "compression": compression,
                "retention_days": retention_days
            }
        else:
            return {"error": "Archive creation failed"}

    def _get_s3_sync(self, bucket: str, prefix: str) -> Optional[Any]:
        """Get S3 sync instance for bucket."""
        if bucket == self.s3_bucket and prefix == self.s3_prefix:
            return self.s3_sync

        try:
            from flowbase.storage.s3_sync import S3Sync
            return S3Sync(bucket=bucket, prefix=prefix)
        except ImportError:
            return None

    def _find_matching_files(self, s3_sync: Any, prefix: str, pattern: str) -> List[str]:
        """Find files matching pattern in S3."""
        import boto3
        import fnmatch
        from pathlib import Path

        s3_client = boto3.client('s3')
        paginator = s3_client.get_paginator('list_objects_v2')

        matching_keys = []

        # Handle recursive patterns (**)
        if "**" in pattern:
            # For recursive patterns, we need to search more broadly
            # Remove the ** part and search from the base prefix
            base_prefix = prefix.rstrip("/")
            search_prefix = base_prefix + "/" if base_prefix else ""

            for page in paginator.paginate(Bucket=s3_sync.bucket, Prefix=search_prefix):
                if "Contents" not in page:
                    continue

                for obj in page["Contents"]:
                    key = obj["Key"]
                    # Get path relative to the search prefix
                    if search_prefix and key.startswith(search_prefix):
                        rel_key = key[len(search_prefix):]
                    else:
                        rel_key = key

                    # Convert S3 key to path-like structure for glob matching
                    path_obj = Path(rel_key)

                    # Simple glob matching - check if pattern matches
                    # This is a simplified version; for full glob support, consider using pathlib.Path.match()
                    if self._matches_glob_pattern(path_obj, pattern):
                        matching_keys.append(key)
        else:
            # Simple pattern matching without **
            for page in paginator.paginate(Bucket=s3_sync.bucket, Prefix=prefix):
                if "Contents" not in page:
                    continue

                for obj in page["Contents"]:
                    key = obj["Key"]
                    rel_key = key[len(prefix):].lstrip("/")
                    if fnmatch.fnmatch(rel_key, pattern):
                        matching_keys.append(key)

        return matching_keys

    def _matches_glob_pattern(self, path_obj: Path, pattern: str) -> bool:
        """Check if a path matches a glob pattern with ** support."""
        # Use pathlib's built-in glob matching which properly supports **
        return path_obj.match(pattern)

    def _combine_files(self, file_list: List[Path], input_format: str) -> List[Dict[str, Any]]:
        """Combine multiple files into a single data structure."""
        combined_data = []

        for file_path in file_list:
            try:
                if input_format == "jsonl":
                    if str(file_path).endswith('.gz'):
                        with gzip.open(file_path, 'rt') as f:
                            for line in f:
                                if line.strip():
                                    combined_data.append(json.loads(line))
                    else:
                        with open(file_path, 'r') as f:
                            for line in f:
                                if line.strip():
                                    combined_data.append(json.loads(line))

                elif input_format == "csv":
                    if str(file_path).endswith('.gz'):
                        with gzip.open(file_path, 'rt') as f:
                            df = pd.read_csv(f)
                    else:
                        df = pd.read_csv(file_path)
                    combined_data.extend(df.to_dict('records'))

                elif input_format == "parquet":
                    df = pd.read_parquet(file_path)
                    combined_data.extend(df.to_dict('records'))

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue

        return combined_data

    def _write_output_file(self, df: pd.DataFrame, temp_file: Path, output_format: str, compression: Optional[str]) -> Path:
        """Write DataFrame to output file with optional compression."""
        if output_format == "parquet":
            df.to_parquet(temp_file.with_suffix('.parquet'), index=False)
            final_file = temp_file.with_suffix('.parquet')

        elif output_format == "jsonl":
            with open(temp_file.with_suffix('.jsonl'), 'w') as f:
                for record in df.to_dict('records'):
                    f.write(json.dumps(record) + '\n')
            final_file = temp_file.with_suffix('.jsonl')

        elif output_format == "csv":
            df.to_csv(temp_file.with_suffix('.csv'), index=False)
            final_file = temp_file.with_suffix('.csv')

        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        # Apply compression if specified
        if compression == "gzip":
            import gzip
            compressed_file = final_file.with_suffix(final_file.suffix + '.gz')
            with gzip.open(compressed_file, 'wb') as f_out:
                with open(final_file, 'rb') as f_in:
                    f_out.write(f_in.read())
            final_file = compressed_file

        return final_file

    def _clean_dataframe_for_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean DataFrame to handle string 'NaN' values and type conversion issues.

        This method ensures that string representations of NaN are converted to proper
        pandas NaN values, and attempts to convert columns to appropriate numeric types
        where possible, making the rollup more fault-tolerant.
        """
        # Replace common string representations of NaN with proper pandas NaN
        df = df.replace(['NaN', 'nan', 'null', 'NULL', 'None', ''], pd.NA)

        # Try to convert object columns to numeric types where appropriate
        for col in df.columns:
            if df[col].dtype == 'object':
                # Attempt numeric conversion, coercing errors to NaN
                try:
                    # Use pd.to_numeric with errors='coerce' to convert invalid values to NaN
                    numeric_series = pd.to_numeric(df[col], errors='coerce')

                    # Only convert if more than 50% of values are successfully converted to numeric
                    # This avoids converting columns that are actually meant to be strings
                    non_na_count = numeric_series.notna().sum()
                    if non_na_count > len(numeric_series) * 0.5:
                        df[col] = numeric_series
                        logger.info(f"Converted column '{col}' to numeric type ({non_na_count}/{len(numeric_series)} values)")
                    else:
                        logger.debug(f"Kept column '{col}' as object type (only {non_na_count}/{len(numeric_series)} numeric values)")
                except Exception as e:
                    logger.debug(f"Could not convert column '{col}' to numeric: {e}")

        return df

    def _archive_files(self, file_list: List[str], archive_location: str, params: Dict[str, Any]) -> bool:
        """Archive files to compressed tar.gz in S3."""
        import io

        # Parse archive location
        if not archive_location.startswith("s3://"):
            return False

        s3_path = archive_location.replace("s3://", "")
        if "/" in s3_path:
            archive_bucket, archive_key = s3_path.split("/", 1)
        else:
            return False

        # Substitute template variables
        archive_key = self._substitute_templates(archive_key, params)

        # Create tar.gz archive in memory
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode='w:gz') as tar:
            for s3_key in file_list:
                # For S3 files, we need to download them first
                # This is a simplified version - in practice you'd want to stream
                temp_file = self.temp_dir / Path(s3_key).name
                temp_file.parent.mkdir(parents=True, exist_ok=True)

                # Download file to temp location
                source_s3 = self._get_s3_sync(self.s3_bucket or archive_bucket, "")
                if source_s3 and source_s3.download_file(s3_key, temp_file):
                    tar.add(temp_file, arcname=Path(s3_key).name)
                    temp_file.unlink()  # Clean up

        tar_buffer.seek(0)

        # Upload archive to S3
        archive_s3 = self._get_s3_sync(archive_bucket, "")
        if archive_s3:
            temp_archive_path = self.temp_dir / f"temp_archive_{hash(archive_key)}.tar.gz"
            with open(temp_archive_path, 'wb') as f:
                f.write(tar_buffer.getvalue())

            success = archive_s3.upload_file(temp_archive_path, archive_key)
            temp_archive_path.unlink()
            return success

        return False

    def _delete_source_files(self, file_list: List[str]) -> None:
        """Delete source files after successful rollup."""
        import boto3

        s3_client = boto3.client('s3')
        for s3_key in file_list:
            try:
                # Extract bucket from key (assuming same bucket for now)
                bucket = self.s3_bucket
                if bucket:
                    s3_client.delete_object(Bucket=bucket, Key=s3_key)
                    logger.info(f"Deleted source file: s3://{bucket}/{s3_key}")
            except Exception as e:
                logger.warning(f"Failed to delete {s3_key}: {e}")

    def _create_archive(self, file_list: List[str], source_s3: Any, archive_key: str, target_s3: Any) -> bool:
        """Create a compressed archive of files."""
        import io

        # Create tar.gz archive in memory
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode='w:gz') as tar:
            for s3_key in file_list:
                # Download and add each file
                temp_file = self.temp_dir / f"temp_{hash(s3_key)}"
                if source_s3.download_file(s3_key[len(source_s3.prefix):].lstrip("/"), temp_file):
                    tar.add(temp_file, arcname=Path(s3_key).name)
                    temp_file.unlink()

        tar_buffer.seek(0)

        # Upload archive
        temp_archive_path = self.temp_dir / f"temp_archive_{hash(archive_key)}.tar.gz"
        with open(temp_archive_path, 'wb') as f:
            f.write(tar_buffer.getvalue())

        success = target_s3.upload_file(temp_archive_path, archive_key)
        temp_archive_path.unlink()

        return success

    def _substitute_templates(self, text: str, params: Dict[str, Any]) -> str:
        """Substitute template variables in text."""
        result = text
        for key, value in params.items():
            result = result.replace(f"{{{{ {key} }}}}", str(value))
        return result

    def _substitute_templates_in_config(self, config: Any, params: Dict[str, Any]) -> Any:
        """Recursively substitute template variables in config values."""
        if isinstance(config, str):
            return self._substitute_templates(config, params)
        elif isinstance(config, dict):
            return {k: self._substitute_templates_in_config(v, params) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_templates_in_config(item, params) for item in config]
        else:
            return config