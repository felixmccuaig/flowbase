"""Workflow orchestration runner for scheduled ML pipelines."""

from __future__ import annotations

import logging
import re
import traceback
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class TaskType(str, Enum):
    """Supported task types in workflows."""

    SCRAPER = "scraper"
    FEATURES = "features"
    INFERENCE = "inference"
    CUSTOM = "custom"


class TaskStatus(str, Enum):
    """Task execution status."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TaskResult:
    """Result of a task execution."""

    name: str
    status: TaskStatus
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_traceback: Optional[str] = None
    duration_seconds: Optional[float] = None


@dataclass
class WorkflowTask:
    """A task within a workflow."""

    name: str
    task_type: TaskType
    config: str
    depends_on: List[str]
    params: Dict[str, Any]
    condition: Optional[str] = None


class WorkflowRunner:
    """Runs declarative workflow configurations."""

    DEFAULT_CONFIG_FILENAMES = ("workflow.yaml", "config.yaml")

    def __init__(self, base_dir: str = "workflows", logs_dir: str = "logs/workflows"):
        self.base_dir = Path(base_dir)
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.logger: Optional[logging.Logger] = None
        self.log_file: Optional[Path] = None
        self.project_config: Optional[Dict[str, Any]] = None
        self.s3_sync = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def _load_project_config(self, project_root: Path) -> None:
        """Load project config (flowbase.yaml) and initialize S3 sync if enabled."""
        flowbase_config_path = project_root / "flowbase.yaml"

        if not flowbase_config_path.exists():
            self.logger.debug("No flowbase.yaml found, S3 sync disabled")
            return

        try:
            with open(flowbase_config_path, 'r') as f:
                self.project_config = yaml.safe_load(f)

            # Check if S3 sync is enabled
            storage_config = self.project_config.get('storage', {})
            sync_enabled = self.project_config.get('sync_artifacts', False)

            if sync_enabled and isinstance(storage_config, dict):
                s3_bucket = storage_config.get('bucket')
                s3_prefix = storage_config.get('prefix', '')

                if s3_bucket:
                    try:
                        from flowbase.storage.s3_sync import S3Sync
                        self.s3_sync = S3Sync(bucket=s3_bucket, prefix=s3_prefix)
                        self.logger.info(f"S3 sync enabled: s3://{s3_bucket}/{s3_prefix}")
                    except ImportError:
                        self.logger.warning("boto3 not installed. S3 sync disabled.")

        except Exception as e:
            self.logger.warning(f"Failed to load project config: {e}")

    def _setup_logging(self, workflow_name: str) -> None:
        """Set up logging for a workflow run."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.logs_dir / f"{workflow_name}_{timestamp}.log"

        # Create logger
        self.logger = logging.getLogger(f"workflow.{workflow_name}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        # File handler with detailed formatting
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self.logger.info(f"Starting workflow: {workflow_name}")
        self.logger.info(f"Log file: {self.log_file}")

    def load_config(
        self,
        workflow_name: str,
        config_path: Optional[str] = None
    ) -> tuple[Dict[str, Any], Path]:
        """Load a workflow config."""
        if config_path:
            candidate = Path(config_path)
        else:
            candidate = self._discover_default_config(workflow_name)

        if not candidate.exists():
            raise FileNotFoundError(f"Workflow config not found: {candidate}")

        with candidate.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        return config, candidate

    def run(
        self,
        workflow_name: str,
        params: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute a workflow.

        Args:
            workflow_name: Name of the workflow to run
            params: Runtime parameters (override config params)
            config_path: Optional explicit path to workflow config
            dry_run: If True, only validate and plan, don't execute

        Returns:
            Dictionary with workflow execution results
        """
        # Set up logging
        self._setup_logging(workflow_name)

        runtime_params = params or {}
        config, config_file = self.load_config(workflow_name, config_path=config_path)
        config_dir = config_file.parent

        self.logger.info(f"Loaded config from: {config_file}")
        self.logger.info(f"Runtime params: {runtime_params}")

        # Resolve template variables
        template_vars = self._build_template_vars(config, runtime_params)

        # Parse tasks
        tasks = self._parse_tasks(config.get("tasks", []))

        # Build execution plan (topological sort)
        execution_order = self._build_execution_plan(tasks)

        if dry_run:
            return {
                "workflow": config.get("name", workflow_name),
                "config_path": str(config_file),
                "template_vars": template_vars,
                "execution_plan": [t.name for t in execution_order],
                "tasks": [self._task_to_dict(t) for t in execution_order],
                "dry_run": True,
            }

        # Execute tasks in order
        results: List[TaskResult] = []
        task_outputs: Dict[str, Any] = {}

        self.logger.info(f"Executing {len(execution_order)} tasks")

        for task in execution_order:
            self.logger.info(f"Starting task: {task.name} (type: {task.task_type})")

            # Check dependencies
            if not self._check_dependencies(task, results):
                self.logger.warning(f"Task {task.name} skipped: dependencies failed")
                results.append(TaskResult(
                    name=task.name,
                    status=TaskStatus.SKIPPED,
                    error="Dependencies failed"
                ))
                continue

            # Evaluate condition
            if task.condition and not self._evaluate_condition(
                task.condition, template_vars, task_outputs
            ):
                self.logger.warning(f"Task {task.name} skipped: condition not met: {task.condition}")
                results.append(TaskResult(
                    name=task.name,
                    status=TaskStatus.SKIPPED,
                    error=f"Condition not met: {task.condition}"
                ))
                continue

            # Execute task
            result = self._execute_task(
                task, config_dir, template_vars, runtime_params, task_outputs
            )
            results.append(result)

            if result.status == TaskStatus.SUCCESS:
                self.logger.info(f"Task {task.name} completed successfully in {result.duration_seconds:.2f}s")
            else:
                self.logger.error(f"Task {task.name} failed: {result.error}")
                if result.error_traceback:
                    self.logger.error(f"Traceback:\n{result.error_traceback}")

            if result.output:
                task_outputs[task.name] = result.output

        success = all(
            r.status in {TaskStatus.SUCCESS, TaskStatus.SKIPPED}
            for r in results
        )

        if success:
            self.logger.info("Workflow completed successfully")
        else:
            self.logger.error("Workflow completed with errors")

        return {
            "workflow": config.get("name", workflow_name),
            "config_path": str(config_file),
            "log_file": str(self.log_file) if self.log_file else None,
            "template_vars": template_vars,
            "results": [self._result_to_dict(r) for r in results],
            "success": success,
        }

    def list_workflows(self) -> List[str]:
        """List available workflow configs."""
        if not self.base_dir.exists():
            return []

        workflows: List[str] = []
        for directory in sorted(self.base_dir.iterdir()):
            if not directory.is_dir():
                continue
            for filename in self.DEFAULT_CONFIG_FILENAMES:
                candidate = directory / filename
                if candidate.exists():
                    workflows.append(str(candidate))
                    break
        return workflows

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _discover_default_config(self, workflow_name: str) -> Path:
        base = self.base_dir / workflow_name
        for filename in self.DEFAULT_CONFIG_FILENAMES:
            candidate = base / filename
            if candidate.exists():
                return candidate
        return base / self.DEFAULT_CONFIG_FILENAMES[0]

    def _build_template_vars(
        self,
        config: Dict[str, Any],
        runtime_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build template variables for substitution."""
        now = datetime.utcnow()

        template_vars = {
            # Date/time variables
            "today": now.strftime("%Y-%m-%d"),
            "yesterday": (now.replace(day=now.day - 1)).strftime("%Y-%m-%d"),
            "now": now.isoformat(),
            "year": now.strftime("%Y"),
            "month": now.strftime("%m"),
            "day": now.strftime("%d"),
            "hour": now.strftime("%H"),

            # Workflow metadata
            "workflow_name": config.get("name", "unknown"),
        }

        # Merge config-level params (substitute templates first)
        config_params = config.get("params", {})
        for key, value in config_params.items():
            if key not in template_vars:
                # Substitute any templates in config param values
                if isinstance(value, str):
                    template_vars[key] = self._substitute_templates(value, template_vars)
                else:
                    template_vars[key] = value

        # Merge runtime params (override config params)
        template_vars.update(runtime_params)

        return template_vars

    def _parse_tasks(self, tasks_cfg: List[Any]) -> List[WorkflowTask]:
        """Parse task definitions from config."""
        tasks: List[WorkflowTask] = []

        for entry in tasks_cfg:
            if not isinstance(entry, dict):
                raise ValueError("Each task must be a mapping")

            name = entry.get("name")
            if not name:
                raise ValueError("Task must have a 'name' field")

            task_type_str = entry.get("type", "custom")
            try:
                task_type = TaskType(task_type_str)
            except ValueError:
                raise ValueError(
                    f"Invalid task type '{task_type_str}'. "
                    f"Must be one of: {[t.value for t in TaskType]}"
                )

            config = entry.get("config", "")
            depends_on = entry.get("depends_on", [])
            if isinstance(depends_on, str):
                depends_on = [depends_on]

            params = entry.get("params", {})
            condition = entry.get("condition")

            tasks.append(WorkflowTask(
                name=str(name),
                task_type=task_type,
                config=str(config),
                depends_on=list(depends_on),
                params=params,
                condition=condition,
            ))

        return tasks

    def _build_execution_plan(self, tasks: List[WorkflowTask]) -> List[WorkflowTask]:
        """
        Build execution plan using topological sort.
        Raises ValueError if there are circular dependencies.
        """
        task_map = {t.name: t for t in tasks}
        visited = set()
        visiting = set()
        order: List[WorkflowTask] = []

        def visit(task_name: str) -> None:
            if task_name in visited:
                return
            if task_name in visiting:
                raise ValueError(f"Circular dependency detected involving task: {task_name}")

            if task_name not in task_map:
                raise ValueError(f"Task '{task_name}' referenced but not defined")

            visiting.add(task_name)
            task = task_map[task_name]

            for dep in task.depends_on:
                visit(dep)

            visiting.remove(task_name)
            visited.add(task_name)
            order.append(task)

        for task in tasks:
            visit(task.name)

        return order

    def _check_dependencies(
        self,
        task: WorkflowTask,
        results: List[TaskResult]
    ) -> bool:
        """Check if all dependencies succeeded."""
        if not task.depends_on:
            return True

        results_map = {r.name: r for r in results}

        for dep_name in task.depends_on:
            dep_result = results_map.get(dep_name)
            if not dep_result or dep_result.status != TaskStatus.SUCCESS:
                return False

        return True

    def _evaluate_condition(
        self,
        condition: str,
        template_vars: Dict[str, Any],
        task_outputs: Dict[str, Any]
    ) -> bool:
        """
        Evaluate a simple condition expression.

        Examples:
            "{{ hour >= 9 and hour < 17 }}"
            "{{ task_name.row_count > 0 }}"
        """
        # Simple template substitution for now
        # In production, use a proper expression evaluator
        try:
            # Replace {{ var }} with values
            expr = condition
            for key, value in template_vars.items():
                expr = expr.replace(f"{{{{ {key} }}}}", str(value))

            # Evaluate simple boolean expressions (UNSAFE - for demo only)
            # In production, use ast.literal_eval or a safe expression parser
            return bool(eval(expr))
        except Exception:
            return False

    def _execute_task(
        self,
        task: WorkflowTask,
        config_dir: Path,
        template_vars: Dict[str, Any],
        runtime_params: Dict[str, Any],
        task_outputs: Dict[str, Any],
    ) -> TaskResult:
        """Execute a single task."""
        start_time = datetime.utcnow()

        try:
            # Substitute template variables in params
            resolved_params = self._substitute_templates(task.params, template_vars)

            # Merge with runtime params
            all_params = {**resolved_params, **runtime_params}

            self.logger.info(f"Task {task.name}: Resolved params: {all_params}")

            # Dispatch to appropriate runner
            if task.task_type == TaskType.SCRAPER:
                output = self._run_scraper(task, config_dir, all_params)
            elif task.task_type == TaskType.FEATURES:
                output = self._run_features(task, config_dir, all_params)
            elif task.task_type == TaskType.INFERENCE:
                output = self._run_inference(task, config_dir, all_params)
            elif task.task_type == TaskType.CUSTOM:
                output = self._run_custom(task, config_dir, all_params)
            else:
                raise ValueError(f"Unsupported task type: {task.task_type}")

            duration = (datetime.utcnow() - start_time).total_seconds()

            return TaskResult(
                name=task.name,
                status=TaskStatus.SUCCESS,
                output=output,
                duration_seconds=duration,
            )

        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            error_tb = traceback.format_exc()

            self.logger.error(f"Task {task.name} failed with exception: {e}")
            self.logger.error(f"Full traceback:\n{error_tb}")

            return TaskResult(
                name=task.name,
                status=TaskStatus.FAILED,
                error=str(e),
                error_traceback=error_tb,
                duration_seconds=duration,
            )

    def _substitute_templates(
        self,
        obj: Any,
        template_vars: Dict[str, Any]
    ) -> Any:
        """Recursively substitute {{ var }} templates in config values."""
        if isinstance(obj, str):
            # Replace {{ var }} with values
            result = obj
            for match in re.finditer(r'\{\{\s*(\w+)\s*\}\}', obj):
                var_name = match.group(1)
                if var_name in template_vars:
                    value = template_vars[var_name]
                    result = result.replace(match.group(0), str(value))
            return result

        elif isinstance(obj, dict):
            return {k: self._substitute_templates(v, template_vars) for k, v in obj.items()}

        elif isinstance(obj, list):
            return [self._substitute_templates(item, template_vars) for item in obj]

        else:
            return obj

    def _run_scraper(
        self,
        task: WorkflowTask,
        config_dir: Path,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a scraper task."""
        from flowbase.scrapers.runner import ScraperRunner

        # Resolve config path relative to project root
        config_path = self._resolve_task_config_path(task.config, config_dir)

        # Find project root and load config for S3 sync
        search_dir = Path(config_path).parent.resolve()
        project_root = None
        while search_dir != search_dir.parent:
            if (search_dir / "data").exists() or (search_dir / "models").exists():
                project_root = search_dir
                break
            search_dir = search_dir.parent

        if not project_root:
            project_root = Path(config_path).parent.parent

        # Load project config if not already loaded
        if not self.project_config:
            self._load_project_config(project_root)

        # Extract date parameter
        date = params.get("date")

        # Check for environment variable overrides for Lambda/cloud environments
        import os
        metadata_db = os.environ.get('FLOWBASE_METADATA_DB')
        temp_dir = os.environ.get('FLOWBASE_TEMP_DIR', '/tmp/data/scrapers/.temp')
        data_root = os.environ.get('FLOWBASE_DATA_ROOT')

        runner = ScraperRunner(metadata_db=metadata_db, temp_dir=temp_dir, data_root=data_root)
        result = runner.run(config_path, date=date)

        # Sync ingested file to S3 if enabled
        s3_url = None
        if self.s3_sync and result and result.get("destination"):
            destination_str = result["destination"]
            # Convert to absolute path if it's relative
            destination = Path(destination_str)
            if not destination.is_absolute():
                destination = project_root / destination

            if destination.exists():
                self.logger.info(f"Syncing scraped data to S3: {destination}")
                # Construct S3 key based on the local path structure
                # If using data_root (Lambda /tmp), strip it to get the original relative path
                import os
                data_root = os.environ.get('FLOWBASE_DATA_ROOT')
                if data_root and destination.is_relative_to(Path(data_root)):
                    relative_path = destination.relative_to(Path(data_root))
                else:
                    relative_path = destination.relative_to(project_root)
                s3_key = str(relative_path)
                if self.s3_sync.upload_file(destination, s3_key):
                    s3_url = f"s3://{self.s3_sync.bucket}/{self.s3_sync.prefix}/{s3_key}".replace("//", "/")
                    self.logger.info(f"Scraped data synced to S3: {s3_url}")
                else:
                    self.logger.warning("Failed to sync scraped data to S3")

        return {
            "type": "scraper",
            "config": config_path,
            "date": date,
            "row_count": result.get("rows", 0) if result else 0,
            "destination": result.get("destination") if result else None,
            "s3_url": s3_url,
        }

    def _run_features(
        self,
        task: WorkflowTask,
        config_dir: Path,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a feature generation task."""
        import yaml
        from flowbase.workflows.dependency_resolver import DependencyResolver
        from flowbase.workflows.table_loader import TableLoader
        from flowbase.query.engines.duckdb_engine import DuckDBEngine
        from flowbase.pipelines.feature_compiler import FeatureCompiler

        # Resolve config path relative to project root
        config_path = self._resolve_task_config_path(task.config, config_dir)

        self.logger.info(f"Running features task with config: {config_path}")

        # Find project root
        search_dir = Path(config_path).parent.resolve()
        project_root = None
        while search_dir != search_dir.parent:
            if (search_dir / "data").exists() or (search_dir / "models").exists():
                project_root = search_dir
                break
            search_dir = search_dir.parent

        if not project_root:
            project_root = Path(config_path).parent.parent

        # Load project config and initialize S3 sync if enabled
        self._load_project_config(project_root)

        # Initialize DuckDB engine
        engine = DuckDBEngine()

        # Resolve and load dependencies
        self.logger.info(f"Resolving data dependencies for features")
        resolver = DependencyResolver(project_root=project_root)

        try:
            # Resolve all dependencies
            feature_dep = resolver.resolve_feature_dependencies(config_path)
            feature_name = feature_dep.config.name

            self.logger.info(f"Feature name: {feature_name}")

            # Determine source table name
            if feature_dep.dataset:
                source_table = feature_dep.dataset.name
            else:
                source_table = "raw_data"

            # Get all table dependencies
            tables = feature_dep.dataset.depends_on_tables if feature_dep.dataset else []
            self.logger.info(f"Found {len(tables)} table dependencies: {[t.name for t in tables]}")

            # Load all tables into DuckDB
            if tables:
                self.logger.info("Loading tables into DuckDB")
                # Check for data_root environment variable for Lambda support
                import os
                data_root = os.environ.get('FLOWBASE_DATA_ROOT')
                loader = TableLoader(engine, project_config=self.project_config, data_root=data_root)

                for table in tables:
                    try:
                        loader.load_table(table)
                    except Exception as e:
                        self.logger.error(f"Failed to load table {table.name}: {e}")
                        # Continue with other tables

                # Now load datasets
                if feature_dep.dataset:
                    self._load_dataset_into_duckdb(engine, feature_dep.dataset, resolver, project_root)

        except Exception as e:
            self.logger.error(f"Dependency resolution/loading failed: {e}")
            raise

        # Convert feature config to dict for compiler
        feature_config_dict = self._feature_config_to_dict(feature_dep.config)

        # Determine which compiler to use based on feature config format
        # If features have 'type' field, use declarative compiler
        # Otherwise use expression-based compiler
        use_declarative = False
        if feature_config_dict.get('features'):
            first_feature = feature_config_dict['features'][0]
            use_declarative = 'type' in first_feature

        # Compile features to SQL
        if use_declarative:
            from flowbase.pipelines.declarative_feature_compiler import DeclarativeFeatureCompiler
            self.logger.info(f"Using declarative feature compiler with source table: {source_table}")

            # Get entity_id and time columns from config, with sensible defaults
            entity_id_column = feature_config_dict.get('entity_id_column', 'entity_id')
            time_column = feature_config_dict.get('time_column', 'timestamp')

            compiler = DeclarativeFeatureCompiler(
                entity_id_column=entity_id_column,
                time_column=time_column,
                source_table=source_table
            )
            sql = compiler.compile(feature_config_dict)
        else:
            self.logger.info(f"Using expression-based compiler with source table: {source_table}")
            compiler = FeatureCompiler(source_table=source_table)
            sql = compiler.compile(feature_config_dict)

        self.logger.info(f"Generated SQL:\n{sql}")

        # Execute feature SQL
        try:
            result_df = engine.execute(sql)
            row_count = len(result_df)

            self.logger.info(f"Generated {row_count:,} rows × {len(result_df.columns)} columns")

            # Save to parquet
            # Use data_root if specified (for Lambda /tmp support)
            import os
            data_root = os.environ.get('FLOWBASE_DATA_ROOT')
            if data_root:
                output_path = Path(data_root) / "data" / "features" / f"{feature_name}.parquet"
            else:
                output_path = project_root / "data" / "features" / f"{feature_name}.parquet"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result_df.to_parquet(output_path)

            self.logger.info(f"Saved features to: {output_path}")

            # Sync to S3 if enabled
            s3_url = None
            if self.s3_sync:
                self.logger.info("Syncing features to S3...")
                s3_key = f"data/features/{feature_name}.parquet"
                if self.s3_sync.upload_file(output_path, s3_key):
                    s3_url = f"s3://{self.s3_sync.bucket}/{self.s3_sync.prefix}/{s3_key}".replace("//", "/")
                    self.logger.info(f"Features synced to S3: {s3_url}")
                else:
                    self.logger.warning("Failed to sync features to S3")

        finally:
            engine.close()

        return {
            "type": "features",
            "config": config_path,
            "feature_name": feature_name,
            "row_count": row_count,
            "output_path": str(output_path),
            "s3_url": s3_url,
        }

    def _load_dataset_into_duckdb(
        self,
        engine: Any,
        dataset: Any,
        resolver: Any,
        project_root: Path
    ) -> None:
        """Load a dataset and its dependencies into DuckDB."""
        from flowbase.pipelines.dataset_compiler import DatasetCompiler

        self.logger.info(f"Loading dataset: {dataset.name}")

        # Use the config from the dataset dependency
        dataset_config_dict = self._dataset_config_to_dict(dataset.config)

        # Check if this is a merged dataset with source datasets
        if dataset.config.sources:
            # Load each source dataset first and create alias views
            for source_ref in dataset.config.sources:
                # Recursively load source dataset
                source_dep = resolver.resolve_dataset_dependencies(source_ref.dataset_config)
                self._load_dataset_into_duckdb(engine, source_dep, resolver, project_root)

                # Create view with the 'name' from the source (used by dataset compiler)
                name_sql = f"CREATE OR REPLACE VIEW {source_ref.name} AS SELECT * FROM {source_dep.name}"
                try:
                    engine.execute(name_sql)
                    self.logger.info(f"Created source view '{source_ref.name}' -> '{source_dep.name}'")
                except Exception as e:
                    self.logger.error(f"Failed to create source view {source_ref.name}: {e}")

                # Also create alias view if specified (for use within the merge SQL)
                if source_ref.alias and source_ref.alias != source_ref.name:
                    alias_sql = f"CREATE OR REPLACE VIEW {source_ref.alias} AS SELECT * FROM {source_dep.name}"
                    try:
                        engine.execute(alias_sql)
                        self.logger.info(f"Created alias view '{source_ref.alias}' -> '{source_dep.name}'")
                    except Exception as e:
                        self.logger.error(f"Failed to create alias view {source_ref.alias}: {e}")

        # Compile and create view for this dataset
        try:
            compiler = DatasetCompiler(source_table="raw_data")
            sql = compiler.compile(dataset_config_dict)

            # Create view
            view_sql = f"CREATE OR REPLACE VIEW {dataset.name} AS {sql}"
            engine.execute(view_sql)

            self.logger.info(f"Created view for dataset: {dataset.name}")

        except Exception as e:
            self.logger.error(f"Failed to load dataset {dataset.name}: {e}")

    def _feature_config_to_dict(self, config: Any) -> Dict[str, Any]:
        """Convert feature config dataclass back to dict format for compiler."""
        result = {
            'name': config.name,
            'description': config.description,
        }

        if config.source:
            result['source'] = {
                'dataset_config': config.source.dataset_config,
                'table': config.source.table,
            }

        # Add declarative feature engineering config if present
        if hasattr(config, 'entity_id_column') and config.entity_id_column:
            result['entity_id_column'] = config.entity_id_column
        if hasattr(config, 'time_column') and config.time_column:
            result['time_column'] = config.time_column

        if config.features:
            # Convert features, preserving all fields
            result['features'] = []
            for f in config.features:
                feature_dict = {'name': f.name}

                # Add all attributes from the feature object
                if hasattr(f, 'type') and f.type:
                    feature_dict['type'] = f.type
                if hasattr(f, 'expression') and f.expression:
                    feature_dict['expression'] = f.expression
                if hasattr(f, 'description') and f.description:
                    feature_dict['description'] = f.description
                if hasattr(f, 'windows') and f.windows:
                    feature_dict['windows'] = f.windows
                if hasattr(f, 'partition_by') and f.partition_by:
                    feature_dict['partition_by'] = f.partition_by
                if hasattr(f, 'filter') and f.filter:
                    feature_dict['filter'] = f.filter
                if hasattr(f, 'value_column') and f.value_column:
                    feature_dict['value_column'] = f.value_column
                if hasattr(f, 'value_expression') and f.value_expression:
                    feature_dict['value_expression'] = f.value_expression
                if hasattr(f, 'lagged') and f.lagged is not None:
                    feature_dict['lagged'] = f.lagged
                if hasattr(f, 'pass_num') and f.pass_num is not None:
                    feature_dict['pass'] = f.pass_num  # Convert back to 'pass' for compiler

                result['features'].append(feature_dict)

        return result

    def _dataset_config_to_dict(self, config: Any) -> Dict[str, Any]:
        """Convert dataset config dataclass back to dict format for compiler."""
        result = {
            'name': config.name,
            'description': config.description,
        }

        if config.source:
            result['source'] = {
                'table': config.source.table,
                'table_config': config.source.table_config,
            }

        if config.sources:
            result['sources'] = [
                {
                    'name': s.name,
                    'dataset_config': s.dataset_config,
                    'alias': s.alias,
                }
                for s in config.sources
            ]

        if config.join:
            result['join'] = {
                'type': config.join.type,
                'conditions': config.join.conditions,
            }

        if config.columns:
            result['columns'] = []
            for c in config.columns:
                col_dict = {
                    'name': c.name,
                    'type': c.type,
                    'required': c.required,
                }
                # Only add optional fields if they have values
                if c.source is not None:
                    col_dict['source'] = c.source
                if c.expression is not None:
                    col_dict['expression'] = c.expression
                if c.validate is not None:
                    col_dict['validate'] = c.validate
                result['columns'].append(col_dict)

        if config.filters:
            result['filters'] = [
                {
                    'column': f.column,
                    'operator': f.operator,
                    'value': f.value,
                }
                for f in config.filters
            ]

        return result

    def _run_inference(
        self,
        task: WorkflowTask,
        config_dir: Path,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run an inference task."""
        from flowbase.inference.runner import InferenceRunner

        # Resolve config path relative to project root
        config_path = self._resolve_task_config_path(task.config, config_dir)

        # Find project root and load config for S3 sync
        search_dir = Path(config_path).parent.resolve()
        project_root = None
        while search_dir != search_dir.parent:
            if (search_dir / "data").exists() or (search_dir / "models").exists():
                project_root = search_dir
                break
            search_dir = search_dir.parent

        if not project_root:
            project_root = Path(config_path).parent.parent

        # Load project config if not already loaded
        if not self.project_config:
            self._load_project_config(project_root)

        # Extract model name from config path or use task name
        model_name = Path(config_path).parent.name

        # Check for data_root environment variable for Lambda support
        import os
        data_root = os.environ.get('FLOWBASE_DATA_ROOT')
        metadata_db = os.environ.get('FLOWBASE_METADATA_DB')

        # Get S3 configuration from project config
        s3_bucket = None
        s3_prefix = ""
        if self.project_config:
            storage_config = self.project_config.get('storage', {})
            if isinstance(storage_config, dict):
                s3_bucket = storage_config.get('bucket')
                s3_prefix = storage_config.get('prefix', '')

        runner = InferenceRunner(
            metadata_db=metadata_db,
            data_root=data_root,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix
        )
        result = runner.run(
            model_name=model_name,
            params=params,
            config_path=config_path,
            skip_outputs=False,
        )

        # Sync prediction outputs to S3 if enabled
        s3_urls = []
        if self.s3_sync and result and result.get("outputs"):
            outputs = result.get("outputs", {})
            # Look for parquet output files
            for output_key, output_path in outputs.items():
                if isinstance(output_path, str):
                    output_file = Path(output_path)
                    if output_file.exists() and output_file.suffix == '.parquet':
                        self.logger.info(f"Syncing prediction output to S3: {output_file}")
                        # Construct S3 key based on the local path structure
                        # If using data_root (Lambda /tmp), strip it to get the original relative path
                        import os
                        data_root_env = os.environ.get('FLOWBASE_DATA_ROOT')
                        if data_root_env and output_file.is_relative_to(Path(data_root_env)):
                            relative_path = output_file.relative_to(Path(data_root_env))
                        else:
                            relative_path = output_file.relative_to(project_root)
                        s3_key = str(relative_path)
                        if self.s3_sync.upload_file(output_file, s3_key):
                            s3_url = f"s3://{self.s3_sync.bucket}/{self.s3_sync.prefix}/{s3_key}".replace("//", "/")
                            s3_urls.append(s3_url)
                            self.logger.info(f"Prediction output synced to S3: {s3_url}")
                        else:
                            self.logger.warning(f"Failed to sync prediction output to S3: {output_file}")

        return {
            "type": "inference",
            "model": result.get("model"),
            "row_count": len(result.get("results", [])),
            "outputs": result.get("outputs"),
            "s3_urls": s3_urls if s3_urls else None,
        }

    def _run_custom(
        self,
        task: WorkflowTask,
        config_dir: Path,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a custom task (user-defined script or binary)."""
        import subprocess
        import json
        import os

        # Resolve config path relative to project root
        config_path = self._resolve_task_config_path(task.config, config_dir)

        # Find project root
        search_dir = Path(config_path).parent.resolve()
        project_root = None
        while search_dir != search_dir.parent:
            if (search_dir / "data").exists() or (search_dir / "models").exists():
                project_root = search_dir
                break
            search_dir = search_dir.parent

        if not project_root:
            project_root = Path(config_path).parent.parent

        # Load project config if not already loaded
        if not self.project_config:
            self._load_project_config(project_root)

        self.logger.info(f"Running custom task with config: {config_path}")

        # Load custom task config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Get command and arguments
        command = config.get('command', [])
        if isinstance(command, str):
            command = [command]
        elif not isinstance(command, list):
            raise ValueError("Command must be a string or list")

        # Substitute template variables in command
        command = [self._substitute_templates(str(arg), params) for arg in command]

        # Resolve command path relative to project root if it's a relative path
        if command and not command[0].startswith('/'):
            command_path = project_root / command[0]
            if command_path.exists():
                command[0] = str(command_path)

        self.logger.info(f"Executing custom command: {' '.join(command)}")

        # Set working directory to project root
        working_dir = config.get('working_dir', str(project_root))
        working_dir = self._substitute_templates(working_dir, params)

        # Set environment variables if specified
        env = os.environ.copy()
        if config.get('env'):
            for key, value in config['env'].items():
                env[key] = self._substitute_templates(str(value), params)

        # Execute command
        start_time = datetime.utcnow()
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                cwd=working_dir,
                env=env
            )
        except Exception as e:
            self.logger.error(f"Failed to execute command: {e}")
            raise RuntimeError(f"Custom task execution failed: {e}")

        duration = (datetime.utcnow() - start_time).total_seconds()

        if result.returncode != 0:
            self.logger.error(f"Command failed with exit code {result.returncode}")
            self.logger.error(f"stderr: {result.stderr}")
            raise RuntimeError(f"Custom task failed with exit code {result.returncode}: {result.stderr}")

        self.logger.info(f"Command completed successfully in {duration:.2f}s")
        if result.stdout:
            self.logger.info(f"stdout: {result.stdout}")

        # Parse output if JSON
        output_data = {}
        if result.stdout.strip():
            try:
                output_data = json.loads(result.stdout)
                self.logger.info(f"Parsed JSON output: {output_data}")
            except json.JSONDecodeError:
                output_data = {"stdout": result.stdout.strip()}

        # Handle output files specified in config
        outputs = {}
        s3_urls = []
        if config.get('outputs'):
            for output_name, output_path in config['outputs'].items():
                # Substitute template variables in output path
                resolved_path = self._substitute_templates(str(output_path), params)
                output_file = Path(resolved_path)

                # Convert to absolute path if relative
                if not output_file.is_absolute():
                    output_file = project_root / output_file

                if output_file.exists():
                    outputs[output_name] = str(output_file)
                    self.logger.info(f"Output file created: {output_name} -> {output_file}")

                    # Sync to S3 if enabled
                    if self.s3_sync:
                        self.logger.info(f"Syncing custom task output to S3: {output_file}")
                        # Construct S3 key based on the local path structure
                        data_root = os.environ.get('FLOWBASE_DATA_ROOT')
                        if data_root and output_file.is_relative_to(Path(data_root)):
                            relative_path = output_file.relative_to(Path(data_root))
                        else:
                            relative_path = output_file.relative_to(project_root)
                        s3_key = str(relative_path)
                        if self.s3_sync.upload_file(output_file, s3_key):
                            s3_url = f"s3://{self.s3_sync.bucket}/{self.s3_sync.prefix}/{s3_key}".replace("//", "/")
                            s3_urls.append(s3_url)
                            self.logger.info(f"Custom task output synced to S3: {s3_url}")
                        else:
                            self.logger.warning(f"Failed to sync output to S3: {output_file}")
                else:
                    self.logger.warning(f"Expected output file not found: {output_file}")

        return {
            "type": "custom",
            "command": " ".join(command),
            "exit_code": result.returncode,
            "duration_seconds": duration,
            "outputs": outputs if outputs else None,
            "s3_urls": s3_urls if s3_urls else None,
            **output_data
        }

    def _resolve_task_config_path(self, config_path: str, workflow_dir: Path) -> str:
        """Resolve task config path relative to project root."""
        # Walk up to find project root
        search_dir = workflow_dir.resolve()
        while search_dir != search_dir.parent:
            candidate = search_dir / config_path
            if candidate.exists():
                return str(candidate)

            # Check if we're at project root (has data/ or models/)
            if (search_dir / "data").exists() or (search_dir / "models").exists():
                # Try from project root
                candidate = search_dir / config_path
                if candidate.exists():
                    return str(candidate)
                break

            search_dir = search_dir.parent

        # If not found, return as-is and let the task runner handle it
        return config_path

    def _task_to_dict(self, task: WorkflowTask) -> Dict[str, Any]:
        """Convert task to dictionary for output."""
        return {
            "name": task.name,
            "type": task.task_type.value,
            "config": task.config,
            "depends_on": task.depends_on,
            "params": task.params,
            "condition": task.condition,
        }

    def _result_to_dict(self, result: TaskResult) -> Dict[str, Any]:
        """Convert task result to dictionary for output."""
        return {
            "name": result.name,
            "status": result.status.value,
            "output": result.output,
            "error": result.error,
            "error_traceback": result.error_traceback,
            "duration_seconds": result.duration_seconds,
        }


__all__ = ["WorkflowRunner", "TaskType", "TaskStatus", "TaskResult", "WorkflowTask"]
