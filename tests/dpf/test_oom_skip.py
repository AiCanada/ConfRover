# Copyright 2025 Bytedance Ltd. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""A batch that does not fit must cost one example, not the run.

One oversized 9-frame window ended PDBcluster_from_base 25 steps in, before its
first checkpoint, while every other batch had been fitting.
"""

from __future__ import annotations

import pytest
import torch

from confrover.model import train as model_train

from .test_train_step import _make_batch, train_model  # noqa: F401


@pytest.fixture
def fresh(train_model):
    train_model.oom_skipped = 0
    train_model._oom_consecutive = 0
    train_model.max_oom_skips = 20
    yield train_model
    train_model.oom_skipped = 0
    train_model._oom_consecutive = 0


def _raise(exc):
    def _step(batch, batch_idx=None):
        raise exc
    return _step


def test_an_oom_batch_is_skipped_and_logged(fresh, monkeypatch, caplog):
    monkeypatch.setattr(fresh, "_step", _raise(torch.OutOfMemoryError("CUDA out of memory")))
    with caplog.at_level("WARNING"):
        out = fresh.training_step(_make_batch("forward", [6]), batch_idx=3)
    assert out is None
    assert fresh.oom_skipped == 1
    assert "skipping it" in caplog.text and "1/20 skips" in caplog.text


def test_the_kernel_form_of_oom_is_recognised():
    exc = RuntimeError("CUDA error: out of memory\nSearch for `cudaErrorMemoryAllocation'")
    assert model_train._is_out_of_memory(exc)
    assert not model_train._is_out_of_memory(RuntimeError("shape mismatch"))


def test_other_errors_still_propagate(fresh, monkeypatch):
    monkeypatch.setattr(fresh, "_step", _raise(RuntimeError("shape mismatch")))
    with pytest.raises(RuntimeError, match="shape mismatch"):
        fresh.training_step(_make_batch("iid", [6]), batch_idx=0)
    assert fresh.oom_skipped == 0


def test_a_budget_of_zero_fails_on_the_first_oom(fresh, monkeypatch):
    fresh.max_oom_skips = 0
    monkeypatch.setattr(fresh, "_step", _raise(torch.OutOfMemoryError("CUDA out of memory")))
    with pytest.raises(torch.OutOfMemoryError):
        fresh.training_step(_make_batch("iid", [6]), batch_idx=0)


def test_too_many_in_a_row_aborts(fresh, monkeypatch):
    monkeypatch.setattr(fresh, "_step", _raise(torch.OutOfMemoryError("CUDA out of memory")))
    batch = _make_batch("iid", [6])
    for _ in range(model_train.MAX_CONSECUTIVE_OOM):
        assert fresh.training_step(batch, batch_idx=0) is None
    with pytest.raises(RuntimeError, match="consecutive"):
        fresh.training_step(batch, batch_idx=0)


def test_a_successful_step_resets_the_streak_but_not_the_total(fresh, monkeypatch):
    real_step = fresh._step
    monkeypatch.setattr(fresh, "_step", _raise(torch.OutOfMemoryError("CUDA out of memory")))
    assert fresh.training_step(_make_batch("iid", [6]), batch_idx=0) is None
    monkeypatch.setattr(fresh, "_step", real_step)
    out = fresh.training_step(_make_batch("iid", [6]), batch_idx=1)
    assert out is not None and torch.isfinite(out["loss"])
    assert fresh._oom_consecutive == 0 and fresh.oom_skipped == 1
    fresh.zero_grad(set_to_none=True)


def test_the_total_budget_aborts(fresh, monkeypatch):
    fresh.max_oom_skips = 2
    real_step = fresh._step
    oom = _raise(torch.OutOfMemoryError("CUDA out of memory"))
    batch = _make_batch("iid", [6])
    for _ in range(2):
        monkeypatch.setattr(fresh, "_step", oom)
        assert fresh.training_step(batch, batch_idx=0) is None
        monkeypatch.setattr(fresh, "_step", real_step)
        fresh.training_step(batch, batch_idx=1)
        fresh.zero_grad(set_to_none=True)
    monkeypatch.setattr(fresh, "_step", oom)
    with pytest.raises(RuntimeError, match="max_oom_skips"):
        fresh.training_step(batch, batch_idx=0)
