"""Regression checks for the committed five-task selection analysis."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("task_selection", ROOT / "scripts/analyze_task_svd.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)
EXPECTED_TASKS = [
    "torch-pipeline-parallelism", "pypi-server", "fix-code-vulnerability",
    "qemu-startup", "sanitize-git-repo",
]


def test_committed_input_reproduces_five_task_selection_and_historical_fit():
    matrix, cube, tasks, harnesses, models, audit = analysis.load_aggregate(analysis.DEFAULT_MATRIX)
    baseline = matrix.mean(axis=1, keepdims=True)
    contrasts = matrix - baseline
    selected = analysis.task_order(contrasts)[:5]
    assert [tasks[index] for index in selected] == EXPECTED_TASKS
    fitted = baseline + contrasts @ np.linalg.pinv(contrasts[selected], rcond=analysis.RCOND) @ contrasts[selected]
    error = np.sum((matrix - fitted) ** 2)
    full_r2 = 1 - error / np.sum((matrix - matrix.mean(axis=0)) ** 2)
    contrasts_retained = 1 - error / np.sum(contrasts ** 2)
    assert full_r2 == pytest.approx(0.9530082300840466, abs=1e-12)
    assert contrasts_retained == pytest.approx(0.7906643425014606, abs=1e-12)
    assert audit["trial_count"] == 3510


def test_held_out_prediction_keeps_the_historical_generalization_limit():
    matrix, *_ = analysis.load_aggregate(analysis.DEFAULT_MATRIX)
    orders = []
    for harness in range(matrix.shape[1]):
        train = np.delete(matrix, harness, axis=1)
        orders.append(analysis.task_order(train - train.mean(axis=1, keepdims=True)))
    result = analysis.cross_validate(matrix, orders)
    assert result["unmeasured_r2"][5] == pytest.approx(0.483, abs=0.0005)
    assert result["unmeasured_r2"][0] > result["unmeasured_r2"][5]


def test_unrevealed_held_out_outcomes_cannot_change_predictions():
    matrix, *_ = analysis.load_aggregate(analysis.DEFAULT_MATRIX)
    train = matrix[:, 1:]
    selected = analysis.task_order(train - train.mean(axis=1, keepdims=True))[:5]
    observed = matrix[:, 0].copy()
    original = analysis.predict_fold(train, observed, selected)
    unseen = np.ones(len(observed), dtype=bool)
    unseen[selected] = False
    observed[unseen] = 1 - observed[unseen]
    changed = analysis.predict_fold(train, observed, selected)
    np.testing.assert_array_equal(original, changed)


def test_aggregate_input_rejects_inconsistent_model_weighting(tmp_path):
    data = json.loads(analysis.DEFAULT_MATRIX.read_text())
    data["rates"][0][0] = 0.1
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(data))
    with pytest.raises(AssertionError):
        analysis.load_aggregate(invalid)
