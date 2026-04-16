from __future__ import annotations

from pathlib import Path

import pytest

from data.synthetic import generate_dataset as dataset_module


def test_generate_dataset_uses_seed_documents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw_dir = tmp_path / "raw"
    synthetic_dir = tmp_path / "synthetic"
    raw_dir.mkdir(parents=True)
    synthetic_dir.mkdir(parents=True)

    (raw_dir / "loan_agreement_seed.txt").write_text(
        "Loan test terms: missed payments trigger late fees and default review.",
        encoding="utf-8",
    )
    (raw_dir / "insurance_policy_seed.txt").write_text(
        "Insurance test terms: claims need policy id and incident report.",
        encoding="utf-8",
    )
    (raw_dir / "fraud_compliance_seed.txt").write_text(
        "Fraud test terms: forged income documents are prohibited.",
        encoding="utf-8",
    )

    monkeypatch.setattr(dataset_module, "RAW_DIR", raw_dir)
    monkeypatch.setattr(dataset_module, "SYNTHETIC_DIR", synthetic_dir)

    csv_path, rows = dataset_module.generate_dataset(total_rows=24, seed=7)

    assert csv_path.exists()
    assert len(rows) == 24
    assert any("Loan test terms" in row.context for row in rows)
    assert any("Insurance test terms" in row.context for row in rows)
    assert any("Fraud test terms" in row.context for row in rows)
    assert {row.test_type for row in rows} >= {"happy_path", "edge_case", "adversarial"}


def test_generate_dataset_writes_csv_and_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw_dir = tmp_path / "raw"
    synthetic_dir = tmp_path / "synthetic"
    raw_dir.mkdir(parents=True)
    synthetic_dir.mkdir(parents=True)

    (raw_dir / "loan_agreement_seed.txt").write_text("Loan terms.", encoding="utf-8")
    (raw_dir / "insurance_policy_seed.txt").write_text("Insurance terms.", encoding="utf-8")
    (raw_dir / "fraud_compliance_seed.txt").write_text("Fraud terms.", encoding="utf-8")

    monkeypatch.setattr(dataset_module, "RAW_DIR", raw_dir)
    monkeypatch.setattr(dataset_module, "SYNTHETIC_DIR", synthetic_dir)

    csv_path, rows = dataset_module.generate_dataset(total_rows=20, seed=11)

    json_path = csv_path.with_suffix(".json")
    assert csv_path.exists()
    assert json_path.exists()
    assert len(rows) == 20
    assert all(row.question for row in rows)
    assert all(row.context for row in rows)
    assert all(row.ground_truth for row in rows)
