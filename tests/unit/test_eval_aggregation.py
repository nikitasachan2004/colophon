"""
Unit tests for RAGAS score aggregation logic — rules.md §4 requirement.

This guards against the exact bug that broke the last Phase 4 run:
  float(result["faithfulness"])  →  TypeError because result["metric"]
  returns a List[float] in RAGAS 0.3.9, not a scalar.

The correct path is:
  df = result.to_pandas()   # one row per question, one column per metric
  np.nanmean(df[col].values)  # aggregate DOWN rows, per column

These tests verify that aggregation logic in isolation, using a hand-constructed
DataFrame that simulates what result.to_pandas() returns.  No LLM API calls needed.

Verified against RAGAS 0.3.9 source:
  - EvaluationResult._scores_dict maps metric_name → List[float] (per-row)
  - result["metric"] returns that list — NOT a scalar
  - result.to_pandas() concatenates dataset + scores DataFrames along axis=1
  - Each row = one evaluated question; each metric column = that question's score
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# Import the aggregation helper we extract from run_eval.py
from src.evaluation.run_eval import _aggregate_ragas_df

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


# ── Helper ─────────────────────────────────────────────────────────────────


def _make_mock_result(scores_by_metric: dict[str, list[float]]) -> MagicMock:
    """
    Build a mock RAGAS EvaluationResult whose to_pandas() returns a DataFrame
    shaped exactly like the real object (one row per question, one col per metric).
    Also simulates result[metric] returning a list — the path that must NOT be
    used for aggregation.
    """
    n = len(next(iter(scores_by_metric.values())))
    rows = [{m: scores_by_metric[m][i] for m in scores_by_metric} for i in range(n)]
    df = pd.DataFrame(rows)

    mock = MagicMock()
    mock.to_pandas.return_value = df
    # result["metric"] returns a list — simulating actual RAGAS 0.3.9 behaviour
    mock.__getitem__ = lambda self, key: scores_by_metric[key]
    return mock


# ── Core aggregation correctness ───────────────────────────────────────────


def test_simple_mean_no_nans():
    """Basic case: all scores present, nanmean should equal plain mean."""
    result = _make_mock_result({
        "faithfulness":      [0.8, 1.0, 0.6],
        "answer_relevancy":  [0.7, 0.9, 0.5],
        "context_precision": [0.6, 0.8, 0.4],
        "context_recall":    [0.5, 0.3, 0.7],
    })
    scores = _aggregate_ragas_df(result)
    assert scores["faithfulness"]      == pytest.approx(0.8000, abs=1e-6)
    assert scores["answer_relevancy"]  == pytest.approx(0.7000, abs=1e-6)
    assert scores["context_precision"] == pytest.approx(0.6000, abs=1e-6)
    assert scores["context_recall"]    == pytest.approx(0.5000, abs=1e-6)


def test_nanmean_skips_nan_rows():
    """
    NaN in a cell (timeout/failure) must be skipped per-column, not per-row.
    Dropping the whole row would silently under-count; averaging across the wrong
    axis would produce a single mean instead of per-metric means.
    """
    result = _make_mock_result({
        "faithfulness":      [0.8,        1.0,        float("nan")],
        "answer_relevancy":  [0.7,        0.9,        0.5         ],
        "context_precision": [0.6,        float("nan"), 0.4       ],
        "context_recall":    [0.5,        0.3,        0.7         ],
    })
    scores = _aggregate_ragas_df(result)

    # faithfulness: nanmean([0.8, 1.0]) = 0.9 (ignores nan in row 3)
    assert abs(scores["faithfulness"] - 0.9) < 1e-6

    # answer_relevancy: mean([0.7, 0.9, 0.5]) = 0.7  (no nans)
    assert abs(scores["answer_relevancy"] - 0.7) < 1e-6

    # context_precision: nanmean([0.6, 0.4]) = 0.5 (ignores nan in row 2)
    assert abs(scores["context_precision"] - 0.5) < 1e-6

    # context_recall: mean([0.5, 0.3, 0.7]) = 0.5  (no nans)
    assert abs(scores["context_recall"] - 0.5) < 1e-6


def test_all_nan_column_returns_nan():
    """
    If every row timed out for a metric, the column is all-NaN.
    The result should be NaN (not 0.0, not crash) — a visible NaN is honest.
    """
    result = _make_mock_result({
        "faithfulness":      [float("nan"), float("nan")],
        "answer_relevancy":  [0.5,          0.5         ],
        "context_precision": [float("nan"), float("nan")],
        "context_recall":    [0.3,          0.7         ],
    })
    scores = _aggregate_ragas_df(result)

    assert math.isnan(scores["faithfulness"])
    assert math.isnan(scores["context_precision"])
    assert abs(scores["answer_relevancy"] - 0.5) < 1e-6
    assert abs(scores["context_recall"]   - 0.5) < 1e-6


def test_single_question():
    """Edge case: exactly one scoreable question."""
    result = _make_mock_result({
        "faithfulness":      [0.75],
        "answer_relevancy":  [0.85],
        "context_precision": [0.65],
        "context_recall":    [0.55],
    })
    scores = _aggregate_ragas_df(result)
    assert abs(scores["faithfulness"]      - 0.75) < 1e-6
    assert abs(scores["answer_relevancy"]  - 0.85) < 1e-6
    assert abs(scores["context_precision"] - 0.65) < 1e-6
    assert abs(scores["context_recall"]    - 0.55) < 1e-6


def test_extra_columns_in_dataframe_are_ignored():
    """
    to_pandas() also includes non-metric columns (question, answer, contexts, …).
    The aggregation must only extract the four RAGAS metric columns, not crash
    on trying to nanmean a string column.
    """
    n = 3
    scores_by_metric = {
        "faithfulness":      [0.9, 0.8, 0.7],
        "answer_relevancy":  [0.6, 0.5, 0.4],
        "context_precision": [0.5, 0.6, 0.7],
        "context_recall":    [0.4, 0.3, 0.2],
    }
    rows = [
        {
            "question":   f"q{i}",
            "answer":     f"a{i}",
            "contexts":   [f"ctx{i}"],
            **{m: scores_by_metric[m][i] for m in scores_by_metric},
        }
        for i in range(n)
    ]
    df = pd.DataFrame(rows)
    mock = MagicMock()
    mock.to_pandas.return_value = df

    scores = _aggregate_ragas_df(mock)

    assert set(scores.keys()) == set(METRICS)
    assert scores["faithfulness"] == pytest.approx(0.8, abs=1e-6)


def test_missing_metric_column_is_absent_from_result():
    """
    If RAGAS silently omits a metric column (e.g., context_recall not computable),
    the aggregator must not crash and must not invent a value for the missing metric.
    """
    result = _make_mock_result({
        "faithfulness":      [0.9, 0.8],
        "answer_relevancy":  [0.7, 0.6],
        # context_precision and context_recall are absent
    })
    scores = _aggregate_ragas_df(result)

    assert "faithfulness" in scores
    assert "answer_relevancy" in scores
    assert "context_precision" not in scores
    assert "context_recall" not in scores


def test_result_getitem_returns_list_not_scalar():
    """
    Document the RAGAS 0.3.9 behaviour that caused the original bug:
    result["faithfulness"] returns a List[float], not a float.
    float(a_list) raises TypeError — the fallback path is broken by design.
    This test exists so any future reader understands WHY we use to_pandas().
    """
    scores_list = [0.8, 0.9, 1.0]
    mock = MagicMock()
    mock.__getitem__ = MagicMock(return_value=scores_list)
    returned = mock["faithfulness"]
    assert isinstance(returned, list), "RAGAS result['metric'] must return a list"
    with pytest.raises(TypeError):
        float(returned)  # this is the bug — must NOT be used for aggregation
