"""Helpers for loading project-defined registry provider functions."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable


def load_provider_function(spec: str, *, project_root: Path | None = None) -> Callable[..., Any]:
    """Load a provider function from ``module:function`` or ``file.py:function``."""
    if ":" not in spec:
        raise ValueError(f"Provider spec must be in format 'module:function' or 'file.py:function', got: {spec}")

    module_ref, function_name = spec.rsplit(":", 1)

    if module_ref.endswith(".py"):
        file_path = Path(module_ref)
        if not file_path.is_absolute() and project_root is not None:
            file_path = project_root / file_path
        file_path = file_path.resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Provider file not found: {file_path}")

        module_name = f"flowbase_registry_provider_{file_path.stem}"
        spec_obj = importlib.util.spec_from_file_location(module_name, file_path)
        if spec_obj is None or spec_obj.loader is None:
            raise ImportError(f"Could not load provider module from {file_path}")

        module = importlib.util.module_from_spec(spec_obj)
        sys.modules[module_name] = module
        spec_obj.loader.exec_module(module)
    else:
        if project_root is not None and str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        module = importlib.import_module(module_ref)

    if not hasattr(module, function_name):
        raise AttributeError(f"Function '{function_name}' not found in provider module '{module_ref}'")

    function = getattr(module, function_name)
    if not callable(function):
        raise TypeError(f"Provider target '{spec}' is not callable")
    return function
