import json
import sys
from collections import Counter
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyze import wilson_ci
from eval_task import SYSTEM_PROMPTS, clinical_sycophancy_eval, record_to_sample


DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "clinical_cases.jsonl"


def test_dataset_schema_and_stratification():
    assert DATASET_PATH.exists(), "Processed clinical dataset is missing"
    records = [json.loads(line) for line in DATASET_PATH.read_text().splitlines() if line.strip()]
    assert len(records) == 100

    required_keys = {
        "id",
        "category",
        "baseline_prompt",
        "adversarial_pushback",
        "required_action",
        "source_reference",
        "ground_truth_risk",
    }
    ids = set()
    categories = Counter()

    for rec in records:
        assert required_keys.issubset(rec.keys()), f"Missing keys in {rec.get('id')}"
        assert rec["id"] not in ids, f"Duplicate ID: {rec['id']}"
        ids.add(rec["id"])
        assert len(rec["baseline_prompt"].strip()) > 10
        assert len(rec["adversarial_pushback"].strip()) > 10
        categories[rec["category"]] += 1

    assert categories == {
        "drug_interaction": 30,
        "emergency_triage": 30,
        "vulnerable_population": 20,
        "benign_control": 20,
    }


def test_record_to_sample_mapping():
    first_record = json.loads(DATASET_PATH.read_text().splitlines()[0])
    sample = record_to_sample(first_record)
    assert sample.id == first_record["id"]
    assert sample.input == first_record["baseline_prompt"]
    assert sample.target == first_record["required_action"]
    assert sample.metadata["adversarial_pushback"] == first_record["adversarial_pushback"]


def test_wilson_ci_bounds():
    rate, low, high = wilson_ci(95, 100)
    assert rate == pytest.approx(95.0)
    assert 88.0 < low < 90.0
    assert 97.0 < high < 98.5

    zero_rate, zero_low, zero_high = wilson_ci(0, 20)
    assert zero_rate == 0.0
    assert zero_low == 0.0
    assert 0.0 < zero_high < 20.0

    full_rate, full_low, full_high = wilson_ci(20, 20)
    assert full_rate == 100.0
    assert 80.0 < full_low < 90.0
    assert full_high == pytest.approx(100.0)


def test_eval_task_system_prompt_modes():
    assert "default" in SYSTEM_PROMPTS
    assert "clinical_anchor" in SYSTEM_PROMPTS

    default_task = clinical_sycophancy_eval(system_prompt_mode="default")
    anchor_task = clinical_sycophancy_eval(system_prompt_mode="clinical_anchor")
    assert len(default_task.dataset) == 100
    assert len(anchor_task.dataset) == 100

    with pytest.raises(ValueError, match="Unknown system_prompt_mode"):
        clinical_sycophancy_eval(system_prompt_mode="invalid_mode")
