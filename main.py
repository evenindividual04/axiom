from __future__ import annotations

import argparse
import json
import sys
import subprocess
from datetime import datetime, UTC
from pathlib import Path

from dotenv import load_dotenv

SRC_PATH = Path(__file__).resolve().parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

load_dotenv(Path(__file__).resolve().parent / ".env")

from llm_reliability_engine.runner import (
    run_genai_agent_eval,
    run_genai_prompt_eval,
    run_qevals_datagen,
    run_qevals_eval,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified runner for llm-reliability-engine")
    parser.add_argument("command", help="Command to execute")
    parser.add_argument("--live", action="store_true", help="Force live model execution (mock mode off)")
    parser.add_argument("--config", default="config.yaml", help="Path to runtime config")
    return parser


def _validate_command(command: str) -> None:
    allowed = {
        "run-pipeline",
        "dashboard",
        "qevals-datagen",
        "qevals-eval",
        "genai-prompt-eval",
        "genai-agent-eval",
        "status",
    }
    if command not in allowed:
        raise SystemExit(f"Unsupported command: {command}")


def _run_dashboard() -> int:
    cmd = [sys.executable, "-m", "streamlit", "run", "dashboard/app.py"]
    return subprocess.run(cmd, check=False).returncode


def _run_pipeline_command(config_path: str, force_live: bool) -> int:
    from llm_reliability_engine.orchestrator import run_pipeline

    payload = run_pipeline(config_path=Path(config_path), force_live=force_live)
    print(json.dumps(payload, indent=2))
    return 0


def _status_payload() -> dict[str, str]:
    return {
        "project": "llm-reliability-engine",
        "status": "ready",
        "timestamp": datetime.now(UTC).isoformat(),
    }


def main() -> int:
    args = _build_parser().parse_args()
    _validate_command(args.command)

    if args.command == "run-pipeline":
        return _run_pipeline_command(config_path=args.config, force_live=args.live)
    if args.command == "dashboard":
        return _run_dashboard()
    if args.command == "qevals-datagen":
        return run_qevals_datagen()
    if args.command == "qevals-eval":
        return run_qevals_eval()
    if args.command == "genai-prompt-eval":
        return run_genai_prompt_eval()
    if args.command == "genai-agent-eval":
        return run_genai_agent_eval()

    print(json.dumps(_status_payload(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
