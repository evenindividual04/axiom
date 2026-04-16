from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def legacy_root() -> Path:
    return project_root() / "legacy"


def qevals_root() -> Path:
    return legacy_root() / "qevals"


def genai_root() -> Path:
    return legacy_root() / "genai_evals"
