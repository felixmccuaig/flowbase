"""Workflow orchestration runner for scheduled ML pipelines."""

from __future__ import annotations

import re
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

    def __init__(self, base_dir: str = "workflows"):
        self.base_dir = Path(base_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
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
        runtime_params = params or {}
        config, config_file = self.load_config(workflow_name, config_path=config_path)
        config_dir = config_file.parent

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

        for task in execution_order:
            # Check dependencies
            if not self._check_dependencies(task, results):
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

            if result.output:
                task_outputs[task.name] = result.output

        return {
            "workflow": config.get("name", workflow_name),
            "config_path": str(config_file),
            "template_vars": template_vars,
            "results": [self._result_to_dict(r) for r in results],
            "success": all(
                r.status in {TaskStatus.SUCCESS, TaskStatus.SKIPPED}
                for r in results
            ),
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
            return TaskResult(
                name=task.name,
                status=TaskStatus.FAILED,
                error=str(e),
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

        # Extract date parameter
        date = params.get("date")

        runner = ScraperRunner()
        result = runner.run(config_path, date=date)

        return {
            "type": "scraper",
            "config": config_path,
            "date": date,
            "row_count": result.get("rows", 0) if result else 0,
            "destination": result.get("destination") if result else None,
        }

    def _run_features(
        self,
        task: WorkflowTask,
        config_dir: Path,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a feature generation task."""
        import subprocess
        import yaml

        # Resolve config path relative to project root
        config_path = self._resolve_task_config_path(task.config, config_dir)

        # Load feature config to get name
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        feature_config = config.get('features', config)
        feature_name = feature_config.get('name', 'features')

        # Find project root for output path
        search_dir = Path(config_path).parent.resolve()
        project_root = None
        while search_dir != search_dir.parent:
            if (search_dir / "data").exists() or (search_dir / "models").exists():
                project_root = search_dir
                break
            search_dir = search_dir.parent

        if not project_root:
            project_root = Path(config_path).parent.parent

        output_path = project_root / "data" / "features" / f"{feature_name}.parquet"

        # Use the CLI command to compile features (it handles all the complexity)
        result = subprocess.run(
            ["flowbase", "features", "compile", config_path, "--output", str(output_path)],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"Feature compilation failed: {result.stderr}")

        # Parse row count from output
        row_count = 0
        for line in result.stdout.split('\n'):
            if 'Generated' in line and 'rows' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'rows' in part and i > 0:
                        row_count = int(parts[i-1].replace(',', ''))
                        break

        return {
            "type": "features",
            "config": config_path,
            "feature_name": feature_name,
            "row_count": row_count,
            "output_path": str(output_path),
        }

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

        # Extract model name from config path or use task name
        model_name = Path(config_path).parent.name

        runner = InferenceRunner()
        result = runner.run(
            model_name=model_name,
            params=params,
            config_path=config_path,
            skip_outputs=False,
        )

        return {
            "type": "inference",
            "model": result.get("model"),
            "row_count": len(result.get("results", [])),
            "outputs": result.get("outputs"),
        }

    def _run_custom(
        self,
        task: WorkflowTask,
        config_dir: Path,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a custom task (user-defined script)."""
        # TODO: Implement custom task runner
        return {
            "type": "custom",
            "config": task.config,
            "status": "not_implemented",
            "message": "Custom task execution not yet implemented"
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
            "duration_seconds": result.duration_seconds,
        }


__all__ = ["WorkflowRunner", "TaskType", "TaskStatus", "TaskResult", "WorkflowTask"]
