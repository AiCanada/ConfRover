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
from pathlib import Path

STEP_RE = re.compile(r"\[step (\d+)\]")
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
        for name, pattern in TRAIN_FIELDS.items():
            row[name] = _first_float(pattern.search(raw))
        train.append(row)
    return train, vals


def rolling(values: list[float | None], window: int) -> list[float | None]:
    """Centred rolling mean that steps over the gaps (a window with no iid
    batch reports no iid loss, so the series has holes by design)."""
    out: list[float | None] = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        seen = [v for v in values[lo : i + 1] if v is not None]
        out.append(sum(seen) / len(seen) if seen else None)
    return out


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


def write_csv(path: Path, train: list[dict], vals: list[dict]) -> None:
    rows = [{"kind": "train", **r} for r in train] + [{"kind": "val", **r} for r in vals]
    fields = ["kind", "epoch", "step", "train_loss", "train_fwd", "train_iid",
              "val_loss", "val_fwd", "val_iid"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def draw(axes, train: list[dict], vals: list[dict], smooth: int, title: str) -> None:
    ax_train, ax_val = axes
    for ax in axes:
        ax.clear()

    steps = [r["step"] for r in train]
    series = [
        ("train_loss", "total", "tab:blue"),
        ("train_fwd", "forward", "tab:orange"),
        ("train_iid", "iid", "tab:green"),
    ]
    for key, label, colour in series:
        raw = [r.get(key) for r in train]
        pts = [(s, v) for s, v in zip(steps, raw) if v is not None]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax_train.plot(xs, ys, color=colour, alpha=0.18, linewidth=0.8)
        smoothed = rolling(list(ys), smooth)
        ax_train.plot(xs, smoothed, color=colour, linewidth=1.8, label=f"{label} (mean of {smooth})")
    ax_train.set_ylabel("train loss")
    ax_train.set_title(title)
    ax_train.legend(loc="upper right", fontsize=8)
    ax_train.grid(alpha=0.25)

    vsteps = [r["step"] for r in vals]
    for key, label, colour in (
        ("val_loss", "val total", "tab:blue"),
        ("val_fwd", "val forward", "tab:orange"),
        ("val_iid", "val iid", "tab:green"),
    ):
        pts = [(s, r.get(key)) for s, r in zip(vsteps, vals) if r.get(key) is not None]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax_val.plot(xs, ys, "o-", color=colour, markersize=3.5, linewidth=1.4, label=label)
        best = min(range(len(ys)), key=lambda i: ys[i])
        if key == "val_fwd":
            ax_val.annotate(
                f"best {ys[best]:.4f} @ {xs[best]}",
                (xs[best], ys[best]), textcoords="offset points", xytext=(6, -12),
                fontsize=8, color=colour,
            )
    ax_val.set_xlabel("optimizer step")
    ax_val.set_ylabel("val loss (fixed bag, t grid)")
    ax_val.legend(loc="upper right", fontsize=8)
    ax_val.grid(alpha=0.25)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = parser.add_argument_group("log source (local path, or ssh)")
    src.add_argument("--log", type=Path, help="Local console.log")
    src.add_argument("--host")
    src.add_argument("--port", type=int, default=22)
    src.add_argument("--user", default="root")
    src.add_argument("--key", help="ssh private key")
    src.add_argument("--remote_log", help="console.log path on the instance")
    parser.add_argument("--interval", type=float, default=60.0, help="Seconds between refreshes.")
    parser.add_argument("--smooth", type=int, default=10, help="Rolling-mean window, in heartbeats.")
    parser.add_argument("--once", action="store_true", help="Render once and exit (no window).")
    parser.add_argument("--out", type=Path, help="Also save the figure here (PNG).")
    parser.add_argument("--csv", type=Path, help="Write the parsed series as CSV.")
    args = parser.parse_args()

    if args.log is None and not (args.host and args.remote_log):
        parser.error("give --log, or --host and --remote_log")
    remote_log = unmangle_remote_path(args.remote_log) if args.remote_log else None
    if remote_log and remote_log != args.remote_log:
        print(f"note: --remote_log {args.remote_log!r} looks shell-rewritten; using {remote_log!r}", file=sys.stderr)
    source = LogSource(args.log, args.host, args.port, args.user, args.key, remote_log)

    import matplotlib

    if args.once:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, height_ratios=[2, 1])
    fig.canvas.manager.set_window_title("ConfRover training") if not args.once else None

    def refresh() -> tuple[int, int]:
        train, vals = parse_console(source.read())
        last = f"step {train[-1]['step']}" if train else "no steps yet"
        best = min((r["val_fwd"] for r in vals if r.get("val_fwd") is not None), default=None)
        title = (
            f"{source.label()}\n{len(train)} heartbeats, {len(vals)} validations, {last}"
            + (f", best val_forward {best:.4f}" if best is not None else "")
        )
        draw(axes, train, vals, max(1, args.smooth), title)
        fig.tight_layout()
        if args.csv:
            write_csv(args.csv, train, vals)
        if args.out:
            fig.savefig(args.out, dpi=150)
        return len(train), len(vals)

    if args.once:
        n_train, n_val = refresh()
        print(f"{n_train} heartbeats, {n_val} validations"
              + (f" -> {args.out}" if args.out else "")
              + (f", csv -> {args.csv}" if args.csv else ""))
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
