import os
import sys
from typing import Any, Dict


def load_params(params_path: str = None) -> Dict[str, Any]:
    """
    Load params.yaml (YAML) and return as dict.
    Requires PyYAML; prompts installation if missing.
    """
    if params_path is None:
        params_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "params.yaml")
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load params.yaml. Install with `pip install pyyaml` "
            "or `pip install --target ./.deps pyyaml`."
        ) from exc

    if not os.path.exists(params_path):
        raise FileNotFoundError(f"params.yaml not found at {params_path}")

    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    return params


def ensure_local_deps() -> None:
    """Ensure ./ .deps is on sys.path for local offline installs."""
    deps = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".deps")
    if os.path.isdir(deps) and deps not in sys.path:
        sys.path.insert(0, deps)
