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


def test_csv_round_trips_every_run(tmp_path):
    import csv as csv_mod

    train, vals = watch.parse_console(CONSOLE)
    out = tmp_path / "curves.csv"
    watch.write_csv(out, [{"label": "v2", "train": train, "vals": vals, "accum": 4}])
    rows = list(csv_mod.DictReader(out.open(encoding="utf-8")))
    assert [r["kind"] for r in rows] == ["train", "train", "val", "val"]
    assert {r["run"] for r in rows} == {"v2"}
    assert rows[0]["train_fwd"] == "0.42231"
    assert rows[2]["val_iid"] == "0.59527"
    # batches = step x accumulation, the axis runs are compared on
    assert rows[0]["batches"] == "40" and rows[2]["batches"] == "500"


# --- overlaying several runs -------------------------------------------------


def test_accumulation_is_inferred_from_the_runs_own_counters():
    """samples= counts this process's samples and restarts on a resume, while
    step does not (v888 ends at samples=2,805, step=5,265 over three restarts).
    Deltas survive that; their ratio is --accumulate_grad_batches."""
    accum4 = [{"step": 10, "samples": 40}, {"step": 20, "samples": 80}, {"step": 30, "samples": 120}]
    assert watch.infer_accumulation(accum4) == 4
    plain = [{"step": 10, "samples": 10}, {"step": 20, "samples": 20}]
    assert watch.infer_accumulation(plain) == 1
    resumed = [  # the reset drops out: its delta is negative
        {"step": 10, "samples": 10}, {"step": 20, "samples": 20},
        {"step": 30, "samples": 5}, {"step": 40, "samples": 15},
    ]
    assert watch.infer_accumulation(resumed) == 1
    assert watch.infer_accumulation([{"step": 10}, {"step": 20}]) == 1, "logs predating samples="


def test_run_specs_accept_a_local_path_and_an_ssh_source(tmp_path):
    label, source = watch.parse_run_spec(f"stage2={tmp_path / 'console.log'}")
    assert label == "stage2" and source.host is None and source.log == tmp_path / "console.log"

    label, source = watch.parse_run_spec(
        "v2=root@1.2.3.4:27032:/workspace/runs/r/logs/console.log"
    )
    assert label == "v2" and source.host == "1.2.3.4" and source.port == 27032
    assert source.user == "root" and source.remote_log == "/workspace/runs/r/logs/console.log"

    # a shell-rewritten remote path is recovered here too
    _, source = watch.parse_run_spec(
        "v2=root@1.2.3.4:27032:C:/Program Files/Git/workspace/runs/r/logs/console.log"
    )
    assert source.remote_log == "/workspace/runs/r/logs/console.log"


def test_every_run_is_distinguishable_by_colour_and_again_without_it():
    """Each panel holds one series, so colour identifies the run -- and
    thickness plus marker say it again, so the figure survives greyscale."""
    styles = watch.RUN_STYLES
    assert len({s["color"] for s in styles}) == len(styles)
    assert len({(s["linewidth"], s["marker"]) for s in styles}) == len(styles)
    # the series carry no colour of their own any more: the panel title names them
    assert all(len(entry) == 3 for entry in watch.SERIES)
    assert [name for _, _, name in watch.SERIES] == ["total", "forward", "iid"]


# --- what the reader can change from the window ------------------------------


def test_the_cli_offers_both_axes_and_both_scales():
    parser = watch.build_parser()
    args = parser.parse_args([])
    assert args.x_axis == "batches" and args.yscale == "linear"
    args = parser.parse_args(["--x", "step", "--yscale", "log"])
    assert args.x_axis == "step" and args.yscale == "log"
    with pytest.raises(SystemExit):
        parser.parse_args(["--yscale", "sqrt"])


def test_the_x_axis_control_repaints_without_reading_anything():
    """Tk callbacks run on the thread that draws the window, so a control that
    read the logs froze it for the whole ssh round trip -- which is what
    happened. The handler may only write into the view and repaint."""
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError as exc:  # no display (CI)
        pytest.skip(f"no Tk display: {exc}")
    try:
        root.withdraw()
        view = {"x_axis": "batches", "yscale": "linear", "smooth": None}
        repaints, fetches = [], []
        controls = watch.build_controls(
            tk.Frame(root), watch.CARBON, view,
            lambda: repaints.append(1), lambda: fetches.append(1),
        )
        controls["x"].set("step")
        controls["on_x"]()
        assert view["x_axis"] == "step"
        assert repaints == [1] and fetches == [], "the toggle must not fetch"

        controls["x"].set("batches")
        controls["on_x"]()
        assert view["x_axis"] == "batches" and len(repaints) == 2
    finally:
        root.destroy()


def test_focus_live_limits_the_x_range_to_the_newest_run():
    """A 900-step run beside a 37,000-step one is a sliver at the origin. With
    focus on, the newest run fills the width and the older ones are read where
    they overlap it."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    live = {"label": "live", "accum": 4, "smooth": 10,
            "train": [{"step": s, "train_loss": 0.5, "train_fwd": 0.4, "train_iid": 0.6}
                      for s in (10, 200)],
            "vals": [{"step": 125, "val_loss": 0.5, "val_fwd": 0.44, "val_iid": 0.6}]}
    old = {"label": "old", "accum": 1, "smooth": 50,
           "train": [{"step": s, "train_loss": 0.5, "train_fwd": 0.4, "train_iid": 0.6}
                     for s in (10, 36000)],
           "vals": [{"step": 36000, "val_loss": 0.5, "val_fwd": 0.42, "val_iid": 0.6}]}

    fig, axes = plt.subplots(len(watch.PANELS), 1)
    axes = list(axes)
    try:
        view = {"x_axis": "batches", "yscale": "linear", "smooth": 10, "focus_live": False}
        watch.draw(axes, [live, old], view, "light")
        assert axes[0].get_xlim()[1] > 30000, "unfocused, the long run sets the range"

        view["focus_live"] = True
        watch.draw(axes, [live, old], view, "light")
        # the live run ends at step 200 x accumulation 4 = 800 samples
        assert 800 <= axes[0].get_xlim()[1] <= 900
        assert axes[-1].get_xlim() == axes[0].get_xlim(), "every panel shares it"
    finally:
        plt.close(fig)


def test_the_window_offers_zoom_and_fit_but_only_when_wired():
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"no Tk display: {exc}")
    try:
        root.withdraw()
        view = {"x_axis": "batches", "yscale": "linear", "smooth": None, "focus_live": False}
        repaints, zooms, fits = [], [], []
        controls = watch.build_controls(
            tk.Frame(root), watch.CARBON, view,
            lambda: repaints.append(1), lambda: None,
            on_zoom=lambda factor: zooms.append(factor), on_fit=lambda: fits.append(1),
        )
        controls["focus"].set(True)
        controls["on_focus"]()
        assert view["focus_live"] is True and repaints == [1]
    finally:
        root.destroy()
