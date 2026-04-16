from __future__ import annotations

import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
SOURCES_PATH = ROOT / "scripts" / "seed_doc_sources.json"
RAW_DIR = ROOT / "data" / "raw"


def _extract_text(url: str) -> str:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "text" in content_type or "html" in content_type:
        return response.text
    return response.content.decode("utf-8", errors="ignore")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    sources = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))

    for source in sources:
        target = RAW_DIR / source["name"]
        text = _extract_text(source["url"])
        target.write_text(text[:20000], encoding="utf-8")
        print(f"wrote {target}")


if __name__ == "__main__":
    main()
