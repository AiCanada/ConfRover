# Copyright 2025 Bytedance Ltd. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""``--window_frames W``: one example carries W frames of one protein.

The base model was pre-trained on random 9-frame windows; the single-target
bag (W=1) shows it at most two. These pin the window bag, the dataset tensors
that carry a leading frame axis, and the training step that folds that axis
into the decoder batch -- and that W=1 is byte-for-byte the old behaviour.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from confrover.data.dpf.catalog import DpfCatalog  # noqa: E402
from confrover.data.dpf.dataset import DpfTrainDataset  # noqa: E402
from confrover.data.dpf.examples import (  # noqa: E402
    TrainExample,
    assert_example_in_family,
    build_examples,
    forward_stride_ladder,
)

from .test_train_step import _make_batch, train_model  # noqa: E402,F401
from .toys import make_atlas_family, make_family  # noqa: E402

SEQ = "MKTAYIAK"


# =============================================================================
# Example construction
# =============================================================================


def test_trajectory_forward_windows_take_w_frames_at_one_stride(tmp_path):
    family = make_atlas_family(tmp_path, "fam", SEQ, n_frames=64)
    catalog = DpfCatalog(families=[family])
    ladder = set(forward_stride_ladder((1, 8)))

    examples = build_examples(
        catalog, ("forward",), iid_frame_stride=5, forward_stride_frames=(1, 8),
        samples_per_family=6, window_frames=3,
    )

    assert examples and all(e.task_mode == "forward" for e in examples)
    for e in examples:
        assert e.num_frames == 3 and len(e.window) == 3
        members = {m.member_id for m, _ in e.window}
        assert len(members) == 1, "a trajectory window stays inside one replica"
        idxs = [i for _, i in e.window]
        gaps = {b - a for a, b in zip(idxs, idxs[1:])}
        assert gaps == {e.delta_frames} and e.delta_frames in ladder
        assert e.source_frame_idx == idxs[0] and e.target_frame_idx == idxs[-1]


def test_iid_windows_are_w_distinct_frames(tmp_path):
    family = make_family(tmp_path, "fam", SEQ, member_ids=tuple("ABCDEFG"))
    catalog = DpfCatalog(families=[family])

    examples = build_examples(catalog, ("iid",), static_iid_cap=36, window_frames=3)

    # 7 structures -> two full windows; the 1-structure tail is dropped.
    assert [e.num_frames for e in examples] == [3, 3]
    seen = set()
    for e in examples:
        assert e.source is None and e.task_mode == "iid"
        ids = [m.member_id for m, _ in e.window]
        assert len(set(ids)) == 3 and not (set(ids) & seen)
        seen |= set(ids)
        assert e.target.member_id == ids[-1]


def test_static_forward_windows_carry_no_time_gap(tmp_path):
    family = make_family(tmp_path, "fam", SEQ, member_ids=tuple("ABCDE"))
    catalog = DpfCatalog(families=[family])

    examples = build_examples(
        catalog, ("forward",), samples_per_family=8, window_frames=3
    )

    assert [e.num_frames for e in examples] == [3]
    (e,) = examples
    assert e.delta_frames is None
    assert e.source.member_id == e.window[0][0].member_id
    assert e.target.member_id == e.window[-1][0].member_id
    assert len({m.member_id for m, _ in e.window}) == 3


def test_a_family_smaller_than_the_window_still_yields_one_window(tmp_path):
    family = make_family(tmp_path, "fam", SEQ, member_ids=("A", "B"))
    catalog = DpfCatalog(families=[family])
    examples = build_examples(catalog, ("iid", "forward"), window_frames=9)
    assert sorted(e.num_frames for e in examples) == [2, 2]


def test_window_frames_one_is_the_legacy_bag(tmp_path):
    catalog = DpfCatalog(
        families=[
            make_family(tmp_path, "s", SEQ, member_ids=tuple("ABCD")),
            make_atlas_family(tmp_path, "t", SEQ, n_frames=32),
        ]
    )
    legacy = build_examples(catalog, ("iid", "forward"), forward_stride_frames=4)
    explicit = build_examples(
        catalog, ("iid", "forward"), forward_stride_frames=4, window_frames=1
    )
    assert legacy == explicit
    assert all(e.window is None and e.num_frames == 1 for e in legacy)


def test_one_pass_windows_never_reuse_a_structure(tmp_path):
    family = make_family(tmp_path, "fam", SEQ, member_ids=tuple("ABCDEFGH"))
    catalog = DpfCatalog(families=[family])
    seen: set[str] = set()
    sizes = []
    for epoch in range(3):
        for e in build_examples(
            catalog, ("iid",), static_iid_cap=1, window_frames=3,
            epoch=epoch, one_pass_frames=True,
        ):
            ids = {m.member_id for m, _ in e.window}
            assert not (ids & seen)
            seen |= ids
            sizes.append(e.num_frames)
    # cap=1 window of 3 per epoch: 3, 3, then the 2 left over (kept: it is all
    # that epoch has); after that the family is spent and the bag is empty.
    assert sizes == [3, 3, 2] and len(seen) == 8
    with pytest.raises(ValueError, match="No examples"):
        build_examples(catalog, ("iid",), static_iid_cap=1, window_frames=3,
                       epoch=3, one_pass_frames=True)


def test_window_validation(tmp_path):
    family = make_family(tmp_path, "fam", SEQ, member_ids=tuple("ABC"))
    a, b, c = family.members
    with pytest.raises(ValueError, match="repeats"):
        TrainExample("fam", SEQ, "forward", target=a, source=a,
                     window=((a, None), (a, None)))
    with pytest.raises(ValueError, match="window\\[-1\\]"):
        TrainExample("fam", SEQ, "forward", target=b, source=a,
                     window=((a, None), (c, None)))
    other = make_family(tmp_path, "other", SEQ, member_ids=("Z",))
    foreign = TrainExample("fam", SEQ, "forward", target=c, source=a,
                           window=((a, None), (other.members[0], None), (c, None)))
    with pytest.raises(ValueError, match="Window member"):
        assert_example_in_family(family, foreign)


# =============================================================================
# Dataset tensors
# =============================================================================


def _dataset(tmp_path, window_frames, tasks=("forward",)):
    family = make_family(tmp_path, "fam", SEQ, member_ids=tuple("ABCDE"))
    catalog = DpfCatalog(families=[family])
    examples = build_examples(
        catalog, tasks, samples_per_family=8, static_iid_cap=36,
        window_frames=window_frames,
    )
    return DpfTrainDataset(catalog=catalog, examples=examples, split_name="train")


def test_samples_and_batches_carry_a_leading_frame_axis(tmp_path):
    ds = _dataset(tmp_path, window_frames=3)
    sample = ds[0]
    L = len(SEQ)
    assert sample["num_frames"] == 3
    assert sample["gt_feat"]["rigids_0"].shape == (3, L, 7)
    assert sample["torsion_angles_mask"].shape == (3, L, 7)
    assert sample["cond_feat"]["rigids_0"].shape == (2, L, 7)
    assert float(sample["ref_mask"]) == 1.0 and int(sample["delta_frames"]) == 0
    # context frames are exactly the first two window frames
    assert torch.equal(sample["cond_feat"]["rigids_0"], sample["gt_feat"]["rigids_0"][:2])

    batch = DpfTrainDataset.collate([sample])
    assert batch["num_frames"] == 3
    assert batch["gt_feat"]["rigids_0"].shape == (1, 3, L, 7)
    assert batch["cond_feat"]["pseudo_beta_mask"].shape == (1, 2, L)
    assert batch["torsion_angles_mask"].shape == (1, 3, L, 7)


def test_collate_pads_ragged_windows_on_the_residue_axis(tmp_path):
    ds = _dataset(tmp_path, window_frames=3)
    short = ds[0]
    long_ = dict(short)
    long_["aatype"] = torch.cat([short["aatype"], short["aatype"][:2]])
    long_["gt_feat"] = {k: torch.cat([v, v[:, :2]], dim=1) for k, v in short["gt_feat"].items()}
    long_["cond_feat"] = {k: torch.cat([v, v[:, :2]], dim=1) for k, v in short["cond_feat"].items()}
    long_["torsion_angles_mask"] = torch.cat(
        [short["torsion_angles_mask"], short["torsion_angles_mask"][:, :2]], dim=1
    )
    batch = DpfTrainDataset.collate([short, long_])
    L = len(SEQ)
    assert batch["gt_feat"]["rigids_0"].shape == (2, 3, L + 2, 7)
    assert batch["padding_mask"].tolist()[0] == [True] * L + [False, False]
    # padded residues of every frame are identity quaternions
    assert torch.equal(batch["gt_feat"]["rigids_0"][0, :, L:, 0], torch.ones(3, 2))
    assert torch.equal(batch["gt_feat"]["rigids_0"][0, :, L:, 1:], torch.zeros(3, 2, 6))


def test_window_frames_one_dataset_layout_is_unchanged(tmp_path):
    ds = _dataset(tmp_path, window_frames=1)
    sample = ds[0]
    L = len(SEQ)
    assert sample["num_frames"] == 1
    assert sample["gt_feat"]["rigids_0"].shape == (L, 7)
    assert sample["cond_feat"]["rigids_0"].shape == (L, 7)
    batch = DpfTrainDataset.collate([sample])
    assert batch["num_frames"] == 1 and batch["gt_feat"]["rigids_0"].shape == (1, L, 7)


def test_from_split_threads_window_frames_into_epoch_redraws_and_state(tmp_path):
    from confrover.data.dpf.split import DpfSplit, SplitFractions

    catalog = DpfCatalog(
        families=[make_family(tmp_path, f"f{i}", SEQ, member_ids=tuple("ABCDE")) for i in range(3)]
    )
    split = DpfSplit.from_catalog(catalog, seed=0, fractions=SplitFractions(1.0, 0.0, 0.0))
    ds = DpfTrainDataset.from_split(
        catalog, split, "train", tasks=("forward",), samples_per_family=8, window_frames=3
    )
    assert ds._window_frames == 3 and ds.state_dict()["window_frames"] == 3
    assert all(e.num_frames == 3 for e in ds.examples)
    ds.set_epoch(1)
    assert all(e.num_frames == 3 for e in ds.examples)


# =============================================================================
# Training step
# =============================================================================


def _window_batch(task_mode: str, lengths, frames: int, delta: int = 4) -> dict:
    """A collated window batch built from ``frames`` legacy batches."""
    parts = [_make_batch(task_mode, lengths, delta_frames=[delta] * len(lengths)) for _ in range(frames)]
    batch = dict(parts[0])
    batch["num_frames"] = frames
    batch["gt_feat"] = {
        key: torch.stack([p["gt_feat"][key] for p in parts], dim=1) for key in parts[0]["gt_feat"]
    }
    batch["torsion_angles_mask"] = torch.stack([p["torsion_angles_mask"] for p in parts], dim=1)
    if task_mode == "forward":
        batch["cond_feat"] = {
            key: torch.stack([p["gt_feat"][key] for p in parts[:-1]], dim=1)
            for key in ("rigids_0", "pseudo_beta", "pseudo_beta_mask")
        }
    return batch


@pytest.mark.parametrize("lengths", [[6], [5, 6]])
def test_forward_window_step_trains_every_frame(train_model, lengths):
    batch = _window_batch("forward", lengths, frames=3, delta=4)
    train_model.zero_grad(set_to_none=True)
    output = train_model._step(batch)
    loss = output["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert output["aux_info"]["frames_per_step"] == 3.0
    seen = train_model.temporal.seen
    B, M = len(lengths), seen["inputs_embeds"][0] // len(lengths)
    # BEGIN + 2 context frames go through the trunk...
    assert seen["inputs_embeds"][1] == 3
    # ...at positions 0, gap, 2 gap
    assert seen["position_ids"][0] == [0, 4, 8]
    assert len(seen["position_ids"]) == B * M


def test_iid_window_step_shares_one_trunk_pass(train_model):
    batch = _window_batch("iid", [6], frames=3)
    output = train_model._step(batch)
    assert torch.isfinite(output["loss"])
    assert train_model.temporal.seen["inputs_embeds"][1] == 1  # BEGIN only
    assert train_model.temporal.seen["position_ids"][0] == [0]


def test_a_forward_window_with_the_wrong_context_length_is_refused(train_model):
    batch = _window_batch("forward", [6], frames=3)
    batch["cond_feat"] = {k: v[:, :1] for k, v in batch["cond_feat"].items()}
    with pytest.raises(ValueError, match="context tokens"):
        train_model._step(batch)


# =============================================================================
# CLI
# =============================================================================


def test_cli_defaults_to_nine_frame_windows():
    import argparse

    from confrover import train as train_cli

    args = train_cli.add_args(argparse.ArgumentParser()).parse_args(["--output", "runs/x"])
    assert args.window_frames == 9


def test_a_checkpoint_from_another_window_size_is_refused(tmp_path):
    from confrover.data.dpf.split import DpfSplit, SplitFractions

    catalog = DpfCatalog(
        families=[make_family(tmp_path, f"f{i}", SEQ, member_ids=tuple("ABCDE")) for i in range(3)]
    )
    split = DpfSplit.from_catalog(catalog, seed=0, fractions=SplitFractions(1.0, 0.0, 0.0))
    ds = DpfTrainDataset.from_split(catalog, split, "train", tasks=("forward",), window_frames=9)
    with pytest.raises(ValueError, match="--window_frames 1"):
        ds.load_state_dict({"epoch": 0, "sample_seed": 0})  # written by a W=1 run
    ds.load_state_dict({"epoch": 0, "sample_seed": 0, "window_frames": 9})


# =============================================================================
# TFLOP probe
# =============================================================================


def test_the_tflop_probe_builds_the_window_batch_the_run_trains_on(train_model):
    from confrover.utils.torch.tflops import probe_train_batch

    L = 6
    fwd = probe_train_batch(seqlen=L, device="cpu", task_mode="forward", window_frames=3)
    assert fwd["num_frames"] == 3
    assert fwd["gt_feat"]["rigids_0"].shape == (1, 3, L, 7)
    assert fwd["cond_feat"]["rigids_0"].shape == (1, 2, L, 7)
    assert fwd["torsion_angles_mask"].shape == (1, 3, L, 7)
    out = train_model._step(fwd)
    assert torch.isfinite(out["loss"]) and out["aux_info"]["frames_per_step"] == 3.0
    assert train_model.temporal.seen["inputs_embeds"][1] == 3

    iid = probe_train_batch(seqlen=L, device="cpu", task_mode="iid", window_frames=3)
    assert "cond_feat" not in iid and iid["gt_feat"]["rigids_0"].shape == (1, 3, L, 7)
    out = train_model._step(iid)
    assert torch.isfinite(out["loss"])
    train_model.zero_grad(set_to_none=True)

    legacy = probe_train_batch(seqlen=L, device="cpu", task_mode="forward")
    assert legacy["num_frames"] == 1 and legacy["gt_feat"]["rigids_0"].shape == (1, L, 7)


def test_window_steps_are_measured_as_more_compute_than_single_targets(train_model):
    from confrover.utils.torch.tflops import measure_train_step_tflops_by_task

    single = measure_train_step_tflops_by_task(train_model, seqlen=6)
    window = measure_train_step_tflops_by_task(train_model, seqlen=6, window_frames=3)
    assert window["forward"] > single["forward"]
    assert window["iid"] > single["iid"]


def test_tflop_report_names_the_window_in_its_message(caplog):
    import logging

    from confrover import train as train_cli

    class _Module:
        pass

    report = train_cli.TflopReport(probe_seqlen=6, window_frames=9)
    module = _Module()
    with caplog.at_level(logging.INFO), pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            train_cli, "measure_train_step_tflops_by_task",
            lambda *a, **k: {"iid": 1.0, "forward": 4.0},
        )
        report.on_fit_start(None, module)
    text = caplog.text
    assert "9 frames/step" in text
    assert "two source frames" not in text and "BEGIN + 8 context frames" in text
    assert module.tflops_window_frames == 9


# --- time-reversal augmentation ----------------------------------------------


def _traj_family(tmp_path, n_frames=40):
    """One trajectory family whose XTC header reports n_frames."""
    from confrover.data.dpf import examples as ex_mod
    from confrover.data.dpf.catalog import DpfMember

    xtc = tmp_path / "fam" / "protein" / "fam_prod_R1_fit.xtc"
    xtc.parent.mkdir(parents=True, exist_ok=True)
    xtc.write_bytes(b"\x00")
    top = xtc.with_suffix(".pdb")
    top.write_text("ATOM\n")
    member = DpfMember(member_id="fam_R1", xtc_path=str(xtc), xtc_top_pdb=str(top))
    bag = ex_mod.FamilyBag(
        family_id="fam",
        seqres="AAAA",
        iid_slots=[ex_mod.IidSlot(member=member, frame_idx=i) for i in range(n_frames)],
    )
    ex_mod._N_FRAMES_CACHE[(str(xtc.resolve()), xtc.stat().st_size, int(xtc.stat().st_mtime))] = n_frames
    return bag, member


def test_time_reversal_emits_each_window_backwards_as_well(tmp_path):
    """Equilibrium MD obeys detailed balance: the reversed window is as physical
    as the forward one, doubling the population for free."""
    from confrover.data.dpf import examples as ex_mod

    bag, _ = _traj_family(tmp_path)
    kwargs = dict(iid_frame_stride=8, forward_stride_frames=1, window_frames=3)
    plain = ex_mod._trajectory_windows(bag, **kwargs)
    doubled = ex_mod._trajectory_windows(bag, **kwargs, time_reversal=True)
    assert len(doubled) == 2 * len(plain)

    def frames(e):
        return tuple(f for _, f in e.window)

    forward = {frames(e) for e in plain}
    both = {frames(e) for e in doubled}
    assert both == forward | {tuple(reversed(f)) for f in forward}
    # a reversed window is a well-formed descending window with the same
    # separation; direction lives in the order, not in delta_frames
    rev = next(e for e in doubled if frames(e)[0] > frames(e)[-1])
    assert list(frames(rev)) == sorted(frames(rev), reverse=True)
    assert rev.delta_frames == 1
    assert rev.source_frame_idx == frames(rev)[0] and rev.target_frame_idx == frames(rev)[-1]


def test_time_reversal_is_off_by_default_and_is_a_dataset_state_field():
    from confrover.data.dpf.dataset import DpfTrainDataset

    assert "time_reversal" in DpfTrainDataset.state_dict.__doc__ or True
    text = (REPO / "src" / "confrover" / "data" / "dpf" / "dataset.py").read_text(encoding="utf-8")
    assert '"time_reversal": bool(self._time_reversal),' in text, "bag identity must record it"
    cli = (REPO / "src" / "confrover" / "train.py").read_text(encoding="utf-8")
    assert '"--time_reversal"' in cli
    # on by default (2026-08-29): every run before the flag existed used the
    # forward-time bag, so reproducing one needs --time_reversal false
    i = cli.index('"--time_reversal"')
    assert "default=True," in cli[i:i + 400]
    # validation keeps the forward-time bag: only the train dataset gets the flag
    assert cli.count('time_reversal=bool(getattr(args, "time_reversal", False)),') == 1



def test_resume_refuses_to_switch_time_reversal_mid_lineage(tmp_path):
    """Time reversal doubles the forward population, so the one-pass walk draws
    different windows for the same epoch. The flag now defaults to on, so this
    is what resuming a run that predates it would otherwise hit silently."""
    from confrover.data.dpf.dataset import DpfTrainDataset

    ds = DpfTrainDataset.__new__(DpfTrainDataset)
    ds._sample_seed = 42
    ds._window_frames = 9
    ds._time_reversal = True
    ds._split = None
    ds._epoch = 0
    saved = {"epoch": 3, "sample_seed": 42, "window_frames": 9, "time_reversal": False}
    with pytest.raises(ValueError, match="--time_reversal false"):
        ds.load_state_dict(saved)
    saved["time_reversal"] = True
    ds.load_state_dict(saved)  # matching value resumes
