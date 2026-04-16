from __future__ import annotations

import os
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def _load_config() -> dict:
    root = Path(__file__).resolve().parent
    explicit_path = os.environ.get("QEVALS_CONFIG_PATH", "").strip()

    if explicit_path:
        config_path = Path(explicit_path)
    else:
        config_path = root / "config" / "config.toml"

    if not config_path.exists():
        template_path = root / "config" / "config.toml.template"
        if not template_path.exists():
            return {}
        config_path = template_path

    data = tomllib.loads(config_path.read_text(encoding="utf-8"))

    # Backward compatibility for old naming in qevals.
    if "EVAL" not in data and "DATAEVAL" in data:
        data["EVAL"] = data["DATAEVAL"]

    return data


config = _load_config()
