from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .pathing import genai_root, qevals_root


def _run(module: str, cwd: Path, pythonpath_roots: list[Path]) -> int:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    root_str = os.pathsep.join(str(root) for root in pythonpath_roots)
    env["PYTHONPATH"] = f"{root_str}{os.pathsep}{existing}" if existing else root_str
    cmd = [sys.executable, "-m", module]
    result = subprocess.run(cmd, cwd=str(cwd), env=env, check=False)
    return result.returncode


def run_qevals_datagen() -> int:
    root = qevals_root()
    return _run(module="datagen.client", cwd=root, pythonpath_roots=[root])


def run_qevals_eval() -> int:
    root = qevals_root()
    return _run(module="eval.client", cwd=root, pythonpath_roots=[root])


def run_genai_prompt_eval() -> int:
    root = genai_root()
    return _run(module="evals_intro.simple_prompt_eval_playground", cwd=root, pythonpath_roots=[root])


def run_genai_agent_eval() -> int:
    root = genai_root()
    return _run(module="evals_intro.simple_agent_eval_playground", cwd=root, pythonpath_roots=[root])
