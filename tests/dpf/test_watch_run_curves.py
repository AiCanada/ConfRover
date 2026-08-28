# Copyright 2025 Bytedance Ltd. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""The live curve watcher: what it reads out of a console log, and what it does
with a path a POSIX-emulating shell has rewritten. Plotting is not tested; the
parsing is what a wrong reading of the run would come from."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
watch = pytest.importorskip("watch_run_curves")

# Two heartbeats and two validations as the trainer actually writes them: the
# progress bar is rewritten with carriage returns, one heartbeat carries no iid
# batch at all, and the val line is the current val_*_loss spelling.
CONSOLE = (
    "Epoch 000/029  [>---]  10/304  3.3%  0:01:54 | 7:16:06  0.35it/s\r"
    "[step 10] epoch=0 train_loss(mean over 39)=0.47441 train_fwd_loss=0.42231 "
    "train_iid_loss=0.50697 trans=0.01250 rot=0.16281 t=0.496 iid=24 fwd=15 L=245 "
    "mem=68.6/74.3G val_loss=0.73450 lr=6.060e-06\n"
    "[val] epoch=0 step=125 val_loss=0.51800 val_fwd_loss=0.44072 val_iid_loss=0.59527 "
    "trans=0.00913 rot=0.17147 t=0.505\n"
    "Epoch 000/029  [=>--]  20/304  6.6%\r"
    "[step 20] epoch=0 train_loss(mean over 40)=0.44000 train_fwd_loss=0.40000 "
    "iid=0 fwd=40 L=200 mem=70.0/80.0G\n"
    "[val] epoch=1 step=250 val_loss=0.51234 val_fwd_loss=0.43224 val_iid_loss=0.59244\n"
)


def test_parses_heartbeats_and_validations_through_the_carriage_returns():
    train, vals = watch.parse_console(CONSOLE)
    assert [r["step"] for r in train] == [10, 20]
    assert train[0]["train_loss"] == pytest.approx(0.47441)
    assert train[0]["train_fwd"] == pytest.approx(0.42231)
    assert train[0]["train_iid"] == pytest.approx(0.50697)
    # a window with no iid batch reports no iid loss: a hole, not a zero
    assert train[1]["train_iid"] is None
    assert train[1]["train_fwd"] == pytest.approx(0.40000)

    assert [r["step"] for r in vals] == [125, 250]
    assert [r["epoch"] for r in vals] == [0, 1]
    assert vals[0]["val_loss"] == pytest.approx(0.51800)
    assert vals[0]["val_fwd"] == pytest.approx(0.44072)
    assert vals[0]["val_iid"] == pytest.approx(0.59527)


def test_a_heartbeats_val_loss_echo_is_not_read_as_a_validation_point():
    """Every heartbeat repeats the last val_loss; only [val] lines are points."""
    train, vals = watch.parse_console(CONSOLE)
    assert len(vals) == 2, "the echo on the step-10 line must not add a third"
    assert all("val_loss" not in r for r in train)


def test_the_older_bare_fwd_iid_spelling_still_parses():
    older = "[val] epoch=2 step=2832 val_loss=0.33479 fwd_loss=0.28956 iid_loss=0.38001\n"
    _, vals = watch.parse_console(older)
    assert vals[0]["val_fwd"] == pytest.approx(0.28956)
    assert vals[0]["val_iid"] == pytest.approx(0.38001)


def test_rolling_mean_steps_over_the_holes():
    assert watch.rolling([1.0, None, 3.0], 2) == [1.0, 1.0, 3.0]
    assert watch.rolling([None, None], 3) == [None, None]


@pytest.mark.parametrize(
    "raw,want",
    [
        # what Git Bash hands the script when the user types the POSIX path
        ("C:/Program Files/Git/workspace/runs/r/logs/console.log",
         "/workspace/runs/r/logs/console.log"),
        ("C:\\Program Files\\Git\\workspace/runs/r/logs/console.log",
         "/workspace/runs/r/logs/console.log"),
        ("//workspace/runs/r/logs/console.log", "/workspace/runs/r/logs/console.log"),
        # already correct: left alone
        ("/workspace/runs/r/logs/console.log", "/workspace/runs/r/logs/console.log"),
        ("/root/other.log", "/root/other.log"),
    ],
)
def test_a_shell_rewritten_remote_path_is_recovered(raw, want):
    assert watch.unmangle_remote_path(raw) == want


def test_csv_round_trips_both_series(tmp_path):
    import csv as csv_mod

    train, vals = watch.parse_console(CONSOLE)
    out = tmp_path / "curves.csv"
    watch.write_csv(out, train, vals)
    rows = list(csv_mod.DictReader(out.open(encoding="utf-8")))
    assert [r["kind"] for r in rows] == ["train", "train", "val", "val"]
    assert rows[0]["train_fwd"] == "0.42231"
    assert rows[2]["val_iid"] == "0.59527"
