# Copyright 2025 Bytedance Ltd. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Live loss curves for a running (or finished) `confrover train` job.

Reads the run's ``logs/console.log`` -- locally, or straight off the rented box
over ssh -- and redraws every ``--interval`` seconds: the 10-step train
heartbeats (total / forward / iid) and the validation points (val / val_forward
/ val_iid) on one shared step axis.

    # the cloud run, over ssh, refreshing every minute
    py -3.13 scripts/watch_run_curves.py --host 170.64.254.80 --port 27032 ^
        --key %USERPROFILE%\\.ssh\\id_ed25519 ^
        --remote_log /workspace/runs/dpf_from_base_v2/logs/console.log

    # a log the puller has already brought down, or any finished run
    py -3.13 scripts/watch_run_curves.py --log "A:\\ATLAS DATA\\remote_payload\\run\\dpf_from_base_v2\\logs\\console.log"

    # one static render (no window), plus the parsed series as CSV
    py -3.13 scripts/watch_run_curves.py --log ... --once --out curves.png --csv curves.csv

    # overlay earlier runs: --run LABEL=SOURCE, repeatable, SOURCE is a local
    # path or user@host:port:/remote/path
    py -3.13 scripts/watch_run_curves.py ^
        --run v2=root@170.64.254.80:27032:/workspace/runs/dpf_from_base_v2/logs/console.log ^
        --run "stage2=A:\\ATLAS DATA\\remote_payload\\run\\dpf_from_PDBcluster\\logs\\console.log" ^
        --run "v888=runs\\dpf_base_train_v888\\logs\\console.log"

Colour is the series (total / forward / iid) and never the run, so the same
quantity is the same colour everywhere; runs are told apart by line thickness
and marker shape, listed in the second legend. The x axis is training samples
(optimizer steps x the accumulation factor, inferred per run from its own
``samples=`` counter), so a run with --accumulate_grad_batches 4 lines up with
one without it.

Train heartbeats are noisy by construction: one step is a single protein at a
single diffusion time, and the mix of iid and forward windows changes step to
step. The rolling mean (``--smooth``, default 10 points) is the line to read;
the raw points are drawn faintly behind it. Validation is a fixed bag on a
deterministic t grid, so those points are comparable step to step as they are.
"""

from __future__ import annotations

import argparse
import csv
import re
import shlex
import subprocess
import sys
from pathlib import Path, PurePosixPath

STEP_RE = re.compile(r"\[step (\d+)\]")
SAMPLES_RE = re.compile(r"samples=(\d+)")
TRAIN_FIELDS = {
    "train_loss": re.compile(r"train_loss\(mean over \d+\)=([0-9.]+)"),
    "train_fwd": re.compile(r"train_fwd_loss=([0-9.]+)"),
    "train_iid": re.compile(r"train_iid_loss=([0-9.]+)"),
}
VAL_RE = re.compile(r"\[val\] epoch=(\d+) step=(\d+) val_loss=([0-9.]+)")
VAL_FIELDS = {
    # the current format; the older runs wrote bare fwd_loss= / iid_loss=
    "val_fwd": re.compile(r"val_fwd_loss=([0-9.]+)|(?<![A-Za-z_])fwd_loss=([0-9.]+)"),
    "val_iid": re.compile(r"val_iid_loss=([0-9.]+)|(?<![A-Za-z_])iid_loss=([0-9.]+)"),
}


def unmangle_remote_path(raw: str) -> str:
    """Undo a POSIX-emulating shell's rewrite of ``--remote_log``.

    Git Bash / MSYS2 rewrite a bare ``/workspace/x`` argument into
    ``C:/Program Files/Git/workspace/x`` before python sees it, and the ssh call
    then cats a path that does not exist on the instance. ``stage_remote_payload
    .validate_remote_root`` refuses such a value because it gets baked into a
    payload; here the intent is unambiguous, so recover it instead. ``//x`` is
    MSYS's own escape and collapses to ``/x``.
    """
    raw = raw.replace("\\", "/")
    if raw.startswith("//"):
        return raw[1:]
    match = re.search(r"(?:^|/)(?:[A-Za-z]:)?(?:/Program Files/Git)?(/workspace/.*)$", raw)
    if match and not raw.startswith("/workspace/"):
        return match.group(1)
    if re.match(r"^[A-Za-z]:/", raw):
        # some other drive-letter rewrite: keep everything from the first
        # directory that looks like an instance root
        for anchor in ("/workspace/", "/root/", "/opt/"):
            if anchor in raw:
                return raw[raw.index(anchor):]
    return raw


def _first_float(match: re.Match[str] | None) -> float | None:
    if match is None:
        return None
    for group in match.groups():
        if group is not None:
            return float(group)
    return None


def parse_console(text: str) -> tuple[list[dict], list[dict]]:
    """``(heartbeats, validations)`` from a console log's text.

    The heartbeat rewrites the progress bar with carriage returns, so the file
    is not line-oriented until they are turned into newlines.
    """
    train: list[dict] = []
    vals: list[dict] = []
    for raw in text.replace("\r", "\n").splitlines():
        if raw.startswith("[val]"):
            head = VAL_RE.search(raw)
            if head is None:
                continue
            row = {
                "epoch": int(head.group(1)),
                "step": int(head.group(2)),
                "val_loss": float(head.group(3)),
            }
            for name, pattern in VAL_FIELDS.items():
                row[name] = _first_float(pattern.search(raw))
            vals.append(row)
            continue
        step = STEP_RE.search(raw)
        if step is None or "train_loss(mean" not in raw:
            continue
        row = {"step": int(step.group(1))}
        samples = SAMPLES_RE.search(raw)
        if samples is not None:
            row["samples"] = int(samples.group(1))
        for name, pattern in TRAIN_FIELDS.items():
            row[name] = _first_float(pattern.search(raw))
        train.append(row)
    return train, vals


def infer_accumulation(train: list[dict]) -> int:
    """Batches per optimizer step, from the run's own counters.

    ``samples=`` counts training samples this *process* has seen, so it restarts
    on a resume while ``step`` (Lightning's global_step) does not -- v888 ends at
    samples=2,805 and step=5,265 across three restarts. Deltas between
    consecutive heartbeats are immune to that as long as the reset is dropped,
    and their ratio is --accumulate_grad_batches: 1 for every run before
    dpf_from_base_v2, 4 for that one. Unknown (no samples= at all) answers 1,
    which is what every log written before the flag existed means.
    """
    ratios: list[float] = []
    for prev, cur in zip(train, train[1:]):
        if "samples" not in prev or "samples" not in cur:
            continue
        d_step = cur["step"] - prev["step"]
        d_samples = cur["samples"] - prev["samples"]
        if d_step > 0 and d_samples > 0:
            ratios.append(d_samples / d_step)
    if not ratios:
        return 1
    ratios.sort()
    return max(1, round(ratios[len(ratios) // 2]))


def smoothing_window(n_points: int, requested: int | None) -> int:
    """Heartbeats to average over. ``None`` scales with the run's length.

    One --smooth for every run reads badly when they differ by 50x: a mean of 10
    is right for a 74-heartbeat run and pure noise on a 3,694-heartbeat one.
    """
    if requested is not None:
        return max(1, requested)
    return int(min(200, max(10, n_points // 40)))


def rolling(values: list[float | None], window: int) -> list[float | None]:
    """Centred rolling mean that steps over the gaps (a window with no iid
    batch reports no iid loss, so the series has holes by design)."""
    out: list[float | None] = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        seen = [v for v in values[lo : i + 1] if v is not None]
        out.append(sum(seen) / len(seen) if seen else None)
    return out


#: How runs are told apart: colour is the series, never the run.
RUN_STYLES = [
    {"linewidth": 2.4, "marker": "o"},
    {"linewidth": 1.7, "marker": "s"},
    {"linewidth": 1.2, "marker": "^"},
    {"linewidth": 0.9, "marker": "D"},
    {"linewidth": 0.7, "marker": "v"},
    {"linewidth": 0.6, "marker": "P"},
]
SERIES = (
    ("train_loss", "val_loss", "total", "tab:blue"),
    ("train_fwd", "val_fwd", "forward", "tab:orange"),
    ("train_iid", "val_iid", "iid", "tab:green"),
)


def parse_run_spec(spec: str) -> tuple[str, "LogSource"]:
    """``LABEL=SOURCE``; SOURCE is a local path or ``user@host:port:/remote``."""
    label, sep, source = spec.partition("=")
    if not sep:
        label, source = Path(spec).stem, spec
    if "@" in source and source.count(":") >= 2:
        userhost, port, remote = source.split(":", 2)
        user, _, host = userhost.rpartition("@")
        return label, LogSource(None, host, int(port), user or "root", None,
                                unmangle_remote_path(remote))
    return label, LogSource(Path(source), None, 22, "root", None, None)


class LogSource:
    """Where the console log is read from: a local path or an ssh host."""

    def __init__(self, log: Path | None, host: str | None, port: int, user: str,
                 key: str | None, remote_log: str | None):
        self.log = log
        self.host = host
        self.port = port
        self.user = user
        self.key = key
        self.remote_log = remote_log

    def read(self) -> str:
        if self.log is not None:
            return self.log.read_text(encoding="utf-8", errors="replace")
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30"]
        if self.key:
            cmd += ["-i", self.key]
        cmd += ["-p", str(self.port), f"{self.user}@{self.host}",
                f"cat {shlex.quote(self.remote_log)}"]
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout

    def label(self) -> str:
        return str(self.log) if self.log is not None else f"{self.host}:{self.remote_log}"


def write_csv(path: Path, runs: list[dict]) -> None:
    """Every parsed point of every run, long-form."""
    fields = ["run", "kind", "epoch", "step", "batches", "samples", "train_loss",
              "train_fwd", "train_iid", "val_loss", "val_fwd", "val_iid"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for run in runs:
            accum = run["accum"]
            for row in run["train"]:
                writer.writerow({"run": run["label"], "kind": "train",
                                 "batches": row["step"] * accum, **row})
            for row in run["vals"]:
                writer.writerow({"run": run["label"], "kind": "val",
                                 "batches": row["step"] * accum, **row})


def draw(axes, runs: list[dict], smooth: int | None, x_axis: str, title: str) -> None:
    """Overlay every run. Colour is the series; runs differ in thickness/marker."""
    ax_train, ax_val = axes
    for ax in axes:
        ax.clear()
    solo = len(runs) == 1

    for index, run in enumerate(runs):
        style = RUN_STYLES[index % len(RUN_STYLES)]
        scale = run["accum"] if x_axis == "batches" else 1

        train = run["train"]
        steps = [r["step"] * scale for r in train]
        every = max(1, len(train) // 12)
        window = smoothing_window(len(train), smooth)
        run["smooth"] = window
        for train_key, _val_key, name, colour in SERIES:
            pts = [(s, r.get(train_key)) for s, r in zip(steps, train)
                   if r.get(train_key) is not None]
            if not pts:
                continue
            xs, ys = zip(*pts)
            if solo:  # the raw heartbeats only when they are not four deep
                ax_train.plot(xs, ys, color=colour, alpha=0.18, linewidth=0.8)
            ax_train.plot(
                xs, rolling(list(ys), window), color=colour,
                linewidth=style["linewidth"], marker=style["marker"],
                markevery=every, markersize=4,
                label=name if index == 0 else None,
            )

        vals = run["vals"]
        vsteps = [r["step"] * scale for r in vals]
        for _train_key, val_key, name, colour in SERIES:
            pts = [(s, r.get(val_key)) for s, r in zip(vsteps, vals)
                   if r.get(val_key) is not None]
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax_val.plot(xs, ys, color=colour, linewidth=style["linewidth"],
                        marker=style["marker"], markersize=4,
                        markevery=max(1, len(xs) // 25),
                        label=f"val {name}" if index == 0 else None)

    ax_train.set_ylabel("train loss")
    ax_train.set_title(title, fontsize=9)
    ax_train.grid(alpha=0.25)
    ax_val.set_ylabel("val loss (fixed bag, t grid)")
    ax_val.set_xlabel("training samples (steps x accumulation)"
                      if x_axis == "batches" else "optimizer step")
    ax_val.grid(alpha=0.25)

    series_legend = ax_train.legend(loc="upper right", fontsize=8, title="series",
                                    framealpha=0.9)
    ax_train.add_artist(series_legend)
    if not solo:
        from matplotlib.lines import Line2D

        handles = []
        for i, run in enumerate(runs):
            style = RUN_STYLES[i % len(RUN_STYLES)]
            best = min((r["val_fwd"] for r in run["vals"] if r.get("val_fwd") is not None),
                       default=None)
            label = run["label"]
            if run["accum"] > 1:
                label += f" (accum x{run['accum']})"
            if best is not None:
                label += f"  best val_fwd {best:.4f}"
            label += f"  [mean {run.get('smooth', '?')}]"
            handles.append(Line2D([], [], color="0.35", linewidth=style["linewidth"],
                                  marker=style["marker"], markersize=4, label=label))
        ax_train.legend(handles=handles, loc="upper left", fontsize=7.5, title="run",
                        framealpha=0.9)
    ax_val.legend(loc="upper right", fontsize=8, framealpha=0.9, ncol=3)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = parser.add_argument_group("log sources")
    src.add_argument("--log", type=Path, help="Local console.log (the primary run)")
    src.add_argument("--host")
    src.add_argument("--port", type=int, default=22)
    src.add_argument("--user", default="root")
    src.add_argument("--key", help="ssh private key")
    src.add_argument("--remote_log", help="console.log path on the instance")
    src.add_argument(
        "--run", action="append", default=[], metavar="LABEL=SOURCE",
        help="Overlay another run; SOURCE is a local path or "
        "user@host:port:/remote/path. Repeatable.",
    )
    parser.add_argument("--interval", type=float, default=60.0, help="Seconds between refreshes.")
    parser.add_argument("--smooth", type=int, default=None,
                        help="Rolling-mean window in heartbeats; default scales "
                             "with each run's length (10-200).")
    parser.add_argument("--x", dest="x_axis", choices=("batches", "step"), default="batches",
                        help="x axis: training samples (default, comparable across "
                             "accumulation settings) or raw optimizer steps.")
    parser.add_argument("--once", action="store_true", help="Render once and exit (no window).")
    parser.add_argument("--out", type=Path, help="Also save the figure here (PNG).")
    parser.add_argument("--csv", type=Path, help="Write the parsed series as CSV.")
    args = parser.parse_args()

    sources: list[tuple[str, LogSource]] = []
    if args.log is not None:
        sources.append((args.log.parent.parent.name or args.log.stem,
                        LogSource(args.log, None, 22, "root", None, None)))
    elif args.host and args.remote_log:
        remote = unmangle_remote_path(args.remote_log)
        if remote != args.remote_log:
            print(f"note: --remote_log {args.remote_log!r} looks shell-rewritten; "
                  f"using {remote!r}", file=sys.stderr)
        label = PurePosixPath(remote).parent.parent.name or args.host
        sources.append((label, LogSource(None, args.host, args.port, args.user,
                                         args.key, remote)))
    for spec in args.run:
        label, source = parse_run_spec(spec)
        if source.host is not None and source.key is None:
            source.key = args.key  # one --key serves every ssh source
        sources.append((label, source))
    if not sources:
        parser.error("give --log, --host with --remote_log, or --run LABEL=SOURCE")

    import matplotlib

    if args.once:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, height_ratios=[2, 1])
    if not args.once:
        try:
            fig.canvas.manager.set_window_title("ConfRover training")
        except AttributeError:
            pass

    def refresh() -> list[dict]:
        runs: list[dict] = []
        for label, source in sources:
            try:
                train, vals = parse_console(source.read())
            except Exception as exc:  # a dead source must not hide the live one
                print(f"{label}: {exc}", file=sys.stderr)
                continue
            runs.append({"label": label, "train": train, "vals": vals,
                         "accum": infer_accumulation(train)})
        if not runs:
            raise RuntimeError("no run could be read")
        live = runs[0]
        last = live["train"][-1]["step"] if live["train"] else 0
        title = (f"{live['label']} @ step {last}"
                 + (f"  +{len(runs) - 1} earlier run(s) overlaid" if len(runs) > 1 else "")
                 + "\nloss is comparable only between runs with the same "
                   "--window_frames and task mix")
        draw(axes, runs, args.smooth, args.x_axis, title)
        fig.tight_layout()
        if args.csv:
            write_csv(args.csv, runs)
        if args.out:
            fig.savefig(args.out, dpi=150)
        return runs

    if args.once:
        runs = refresh()
        for run in runs:
            print(f"{run['label']}: {len(run['train'])} heartbeats, "
                  f"{len(run['vals'])} validations, accumulation x{run['accum']}")
        if args.out:
            print(f"-> {args.out}")
        return 0

    plt.ion()
    plt.show(block=False)
    try:
        while True:
            try:
                refresh()
            except Exception as exc:  # a transient ssh failure must not end the watch
                print(f"refresh failed, retrying in {args.interval:.0f}s: {exc}", file=sys.stderr)
            fig.canvas.draw_idle()
            plt.pause(max(args.interval, 1.0))
            if not plt.fignum_exists(fig.number):
                return 0
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
