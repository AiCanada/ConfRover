# Copyright 2025 Bytedance Ltd. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Execute ConfRoverTrain._step and ConfDiffLoss.

The DPF suite previously covered the catalog/split/manifest layer thoroughly and
never touched the training computation, so three independent faults that stopped
every single training step shipped with a fully green suite:

  1. the bool padding_mask was passed as rigids_mask, and the IPA computes
     ``inf * (square_mask - 1)``, which PyTorch refuses on bool;
  2. _encode_context called the storage-checked rearrange() on ``.expand()``ed
     tensors, which trips its assertion for task_mode="forward" at batch_size>1;
  3. the diffusion embedder stayed activation-checkpointed, so it received no
     gradient at all.

Each test below fails against the corresponding unfixed code.
"""

from __future__ import annotations

import logging
import warnings

import pytest

torch = pytest.importorskip("torch")

from confrover.model.decoder.confdiff.loss import ConfDiffLoss  # noqa: E402

CONFIG = "src/confrover/configs/model/confrover.yaml"
SEQLEN = 6
SINGLE_DIM = 384
PAIR_DIM = 128


class _Output:
    def __init__(self, hidden):
        self.last_hidden_state = hidden


class _StubTemporal(torch.nn.Module):
    """Shape-faithful stand-in for the Llama/pairformer trunk.

    Keeps the test independent of the installed transformers version while still
    exercising the real encoder, decoder, diffuser and loss.
    """

    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.proj = torch.nn.Linear(hidden_size, hidden_size)
        self.seen: dict = {}

    def forward(self, inputs_embeds=None, position_ids=None, **kwargs):
        self.seen = {
            "inputs_embeds": tuple(inputs_embeds.shape),
            "position_ids": None if position_ids is None else position_ids.tolist(),
        }
        return _Output(self.proj(inputs_embeds))


@pytest.fixture(scope="module")
def train_model():
    """A real ConfRoverTrain with the temporal trunk stubbed out.

    The stub isolates decoder/loss blockers. Native Llama/pairformer construction
    and a real ``_step`` are covered in ``test_from_config_native.py``.
    """
    from pathlib import Path

    from confrover.model.confrover import ConfRover
    from confrover.model.train import ConfRoverTrain

    cfg = Path(__file__).resolve().parents[2] / CONFIG
    if not cfg.is_file():
        pytest.skip(f"model config not found: {cfg}")
    backbone = ConfRover.from_config(str(cfg), seed=0)
    model = ConfRoverTrain(
        encoder=backbone.encoder,
        temporal=_StubTemporal(),
        decoder=backbone.decoder,
        seed=0,
        forward_stride_frames=256,
    )
    model.decoder.loss = ConfDiffLoss()
    model.enable_decoder_training()
    model.train()
    return model


def _make_batch(task_mode: str, lengths, delta_frames=None) -> dict:
    """A batch shaped exactly like DpfTrainDataset.collate produces."""
    batch_size, max_l = len(lengths), max(lengths)
    padding_mask = torch.zeros(batch_size, max_l, dtype=torch.bool)
    for i, length in enumerate(lengths):
        padding_mask[i, :length] = True
    keep = padding_mask.float()[..., None]

    rigids = torch.zeros(batch_size, max_l, 7)
    for i, length in enumerate(lengths):
        quat = torch.randn(length, 4)
        rigids[i, :length, :4] = quat / quat.norm(dim=-1, keepdim=True)
        rigids[i, :length, 4:] = torch.randn(length, 3) * 5.0
    rigids[:, :, 0] = rigids[:, :, 0] + (1.0 - keep[..., 0])  # identity on padding

    batch = {
        "task_mode": task_mode,
        "num_frames": 1,
        "forward_stride_frames": 256,
        "padding_mask": padding_mask,
        "aatype": torch.randint(0, 20, (batch_size, max_l)),
        "torsion_angles_mask": torch.ones(batch_size, max_l, 7) * keep,
        "gt_feat": {
            "rigids_0": rigids,
            "rigid_mask": torch.ones(batch_size, max_l) * keep[..., 0],
            "atom14_gt_positions": torch.randn(batch_size, max_l, 14, 3) * keep[..., None],
            "atom14_gt_exists": torch.ones(batch_size, max_l, 14) * keep,
            "atom14_atom_exists": torch.ones(batch_size, max_l, 14) * keep,
            "pseudo_beta": torch.randn(batch_size, max_l, 3) * keep,
            "pseudo_beta_mask": torch.ones(batch_size, max_l) * keep[..., 0],
            "torsion_angles_sin_cos": torch.randn(batch_size, max_l, 7, 2) * keep[..., None],
        },
        "ref_mask": torch.full((batch_size,), 1.0 if task_mode == "forward" else 0.0),
        "is_inference_batch": False,
        "pretrained_single": torch.randn(batch_size, max_l, SINGLE_DIM) * keep,
        "pretrained_pair": torch.randn(batch_size, max_l, max_l, PAIR_DIM),
        "job_info": [{} for _ in range(batch_size)],
    }
    if delta_frames is not None:
        batch["delta_frames"] = torch.tensor(delta_frames, dtype=torch.long)
    if task_mode == "forward":
        cond = torch.zeros(batch_size, max_l, 7)
        for i, length in enumerate(lengths):
            quat = torch.randn(length, 4)
            cond[i, :length, :4] = quat / quat.norm(dim=-1, keepdim=True)
            cond[i, :length, 4:] = torch.randn(length, 3) * 5.0
        cond[:, :, 0] = cond[:, :, 0] + (1.0 - keep[..., 0])
        batch["cond_feat"] = {
            "rigids_0": cond,
            "pseudo_beta": torch.randn(batch_size, max_l, 3) * keep,
            "pseudo_beta_mask": torch.ones(batch_size, max_l) * keep[..., 0],
        }
    return batch


@pytest.mark.parametrize("task_mode", ["iid", "forward"])
@pytest.mark.parametrize("lengths", [[SEQLEN], [SEQLEN, SEQLEN], [SEQLEN, SEQLEN - 2]])
def test_step_runs_and_produces_gradients(train_model, task_mode, lengths):
    """Blockers 1 and 2: every task mode and batch shape must complete a step."""
    torch.manual_seed(0)
    output = train_model._step(_make_batch(task_mode, lengths))

    loss = output["loss"]
    assert loss.ndim == 0
    assert torch.isfinite(loss), f"non-finite loss for {task_mode} {lengths}"

    train_model.zero_grad(set_to_none=True)
    loss.backward()
    grads = [
        p.grad for p in train_model.parameters() if p.grad is not None and torch.any(p.grad != 0)
    ]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite gradient"
    train_model.zero_grad(set_to_none=True)


def test_padding_mask_stays_bool_and_rigids_mask_is_float(train_model):
    """Blocker 1, pinned precisely: the two masks have different dtype contracts."""
    seen = {}
    real_forward = train_model.decoder.forward

    def spy(*args, **kwargs):
        seen["rigids_mask"] = kwargs["rigids_mask"]
        seen["padding_mask"] = kwargs["padding_mask"]
        return real_forward(*args, **kwargs)

    train_model.decoder.forward = spy
    try:
        train_model._step(_make_batch("iid", [SEQLEN]))
    finally:
        train_model.decoder.forward = real_forward
    train_model.zero_grad(set_to_none=True)

    assert seen["padding_mask"].dtype == torch.bool, "embedder needs a bool padding mask"
    assert seen["rigids_mask"].dtype != torch.bool, "IPA cannot subtract from a bool mask"


def test_embedder_receives_gradient(train_model):
    """Blocker 3: the checkpointed embedder was silently frozen (0/20 params)."""
    embedder = train_model.decoder.model_nn.embedder
    assert not hasattr(embedder, "precheckpoint_forward"), "embedder still checkpointed"

    params = [(n, p) for n, p in train_model.named_parameters() if "embedder" in n]
    assert params, "no embedder parameters found"

    torch.manual_seed(0)
    train_model.zero_grad(set_to_none=True)
    # The residual branches feeding the embedder are zero-initialised, so a
    # couple of optimizer steps are needed before every parameter is reached.
    optimizer = torch.optim.AdamW(
        [p for p in train_model.parameters() if p.requires_grad], lr=1e-3
    )
    reached: set[str] = set()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        train_model._step(_make_batch("iid", [SEQLEN, SEQLEN]))["loss"].backward()
        reached.update(
            name
            for name, p in params
            if p.grad is not None and torch.any(p.grad != 0)
        )
        optimizer.step()
    train_model.zero_grad(set_to_none=True)

    missing = [name for name, _ in params if name not in reached]
    assert not missing, f"embedder parameters never trained: {missing}"


def test_forward_position_ids_use_the_examples_own_gap(train_model):
    """A static personality pair has no time separation and must not get the stride."""
    train_model._step(_make_batch("forward", [SEQLEN, SEQLEN], delta_frames=[256, 0]))
    train_model.zero_grad(set_to_none=True)

    position_ids = train_model.temporal.seen["position_ids"]
    rows = {tuple(row) for row in position_ids}
    assert rows == {(0, 256), (0, 0)}, f"unexpected RoPE positions: {rows}"


def test_iid_position_ids_are_all_zero(train_model):
    train_model._step(_make_batch("iid", [SEQLEN]))
    train_model.zero_grad(set_to_none=True)
    rows = {tuple(row) for row in train_model.temporal.seen["position_ids"]}
    assert rows == {(0,)}


def test_validation_t_is_deterministic(train_model):
    """val/loss must be comparable across epochs, so validation walks a fixed grid."""
    batch = _make_batch("iid", [SEQLEN, SEQLEN])
    train_model.eval()
    try:
        draws = []
        for _ in range(3):
            with torch.no_grad():
                draws.append(
                    train_model._sample_t(
                        2, torch.device("cpu"), torch.float32, batch_idx=2
                    ).tolist()
                )
    finally:
        train_model.train()
    assert draws[0] == draws[1] == draws[2], f"validation t is not deterministic: {draws}"
    assert all(train_model.tmin <= t <= train_model.tmax for t in draws[0])
    del batch


def test_training_t_is_per_example(train_model):
    torch.manual_seed(0)
    drawn = train_model._sample_t(8, torch.device("cpu"), torch.float32)
    assert len(set(round(float(x), 8) for x in drawn)) > 1, "t is shared across the batch"


# --------------------------------------------------------------------------
# ConfDiffLoss
# --------------------------------------------------------------------------


def _score_batch(batch_size=2, seqlen=5):
    return {
        "t": torch.full((batch_size,), 0.5),
        "rigids_mask": torch.ones(batch_size, seqlen),
        "torsion_angles_mask": torch.ones(batch_size, seqlen, 7),
        "pred_rigids_0": None,
        "pred_torsion_sin_cos": torch.randn(batch_size, seqlen, 7, 2),
        "pred_atom14": torch.randn(batch_size, seqlen, 14, 3),
        "pred_rot_score": torch.randn(batch_size, seqlen, 3),
        "pred_trans_score": torch.randn(batch_size, seqlen, 3),
        "pred_sidechain_frame": None,
    }


def test_score_terms_are_divided_by_score_scaling():
    """Without this the objective swings ~500x in magnitude with the timestep."""
    loss_fn = ConfDiffLoss(rot_weight=0.0, torsion_weight=0.0, atom14_weight=0.0)
    kwargs = _score_batch()
    gt = {
        "trans_score": torch.ones(2, 5, 3),
        "trans_score_scaling": torch.tensor([1.0, 1.0]),
    }
    kwargs["pred_trans_score"] = torch.zeros(2, 5, 3)

    unscaled, _ = loss_fn(gt_feat=dict(gt), **kwargs)
    gt_scaled = dict(gt, trans_score_scaling=torch.tensor([10.0, 10.0]))
    scaled, _ = loss_fn(gt_feat=gt_scaled, **kwargs)

    assert float(unscaled) == pytest.approx(1.0, rel=1e-5)
    # residual/10 => squared error /100
    assert float(scaled) == pytest.approx(0.01, rel=1e-5)


def test_atom14_term_is_gated_on_t():
    loss_fn = ConfDiffLoss(aux_loss_t_lim=0.25)
    gt = {
        "atom14_gt_positions": torch.randn(2, 5, 14, 3),
        "atom14_gt_exists": torch.ones(2, 5, 14),
    }
    pred = torch.randn(2, 5, 14, 3)
    low = float(loss_fn._atom14_loss(pred, gt, torch.tensor([0.1, 0.1])))
    high = float(loss_fn._atom14_loss(pred, gt, torch.tensor([0.9, 0.9])))
    assert low > 0.0
    assert high == 0.0, "atom14 must not be supervised at high t"


def test_symmetric_side_chain_alternative_is_not_penalised():
    """A prediction matching the alternate naming exactly must score zero."""
    loss_fn = ConfDiffLoss(aux_loss_t_lim=1.0)
    gt_pos = torch.randn(1, 4, 14, 3)
    alt_pos = torch.randn(1, 4, 14, 3)
    gt = {
        "atom14_gt_positions": gt_pos,
        "atom14_gt_exists": torch.ones(1, 4, 14),
        "atom14_alt_gt_positions": alt_pos,
        "atom14_alt_gt_exists": torch.ones(1, 4, 14),
    }
    matched_alt = float(loss_fn._atom14_loss(alt_pos.clone(), gt, torch.tensor([0.1])))
    assert matched_alt == pytest.approx(0.0, abs=1e-6)

    torsion_gt = torch.randn(1, 4, 7, 2)
    torsion_alt = torch.randn(1, 4, 7, 2)
    feat = {
        "torsion_angles_sin_cos": torsion_gt,
        "alt_torsion_angles_sin_cos": torsion_alt,
    }
    matched = float(
        loss_fn._torsion_loss(torsion_alt.clone(), feat, torch.ones(1, 4, 7))
    )
    assert matched == pytest.approx(0.0, abs=1e-6)


def test_loss_rejects_an_empty_objective():
    loss_fn = ConfDiffLoss()
    with pytest.raises(ValueError, match="no supervised terms"):
        loss_fn(gt_feat={}, **_score_batch())


def test_atom14_frac_reports_how_much_of_the_batch_the_gate_let_through():
    """Without it, a consumer cannot tell "not supervised" from "no error"."""
    loss_fn = ConfDiffLoss(aux_loss_t_lim=0.25)
    kwargs = _score_batch(batch_size=2)
    gt = {
        "atom14_gt_positions": torch.randn(2, 5, 14, 3),
        "atom14_gt_exists": torch.ones(2, 5, 14),
    }

    kwargs["t"] = torch.tensor([0.1, 0.1])
    _, aux_open = loss_fn(gt_feat=dict(gt), **kwargs)
    kwargs["t"] = torch.tensor([0.9, 0.9])
    _, aux_shut = loss_fn(gt_feat=dict(gt), **kwargs)
    kwargs["t"] = torch.tensor([0.1, 0.9])
    _, aux_half = loss_fn(gt_feat=dict(gt), **kwargs)

    assert float(aux_open["atom14_frac"]) == pytest.approx(1.0)
    assert float(aux_shut["atom14_frac"]) == pytest.approx(0.0)
    assert float(aux_half["atom14_frac"]) == pytest.approx(0.5)

    # The 0.0 at high t is the gate, not a perfect side-chain prediction.
    assert float(aux_shut["atom14_loss"]) == 0.0
    assert float(aux_open["atom14_loss"]) > 0.0


# =============================================================================
# Gradient coverage: measure the property, do not sniff for one known wrapper
# =============================================================================


def _run_coverage(model, steps, task="iid"):
    """Drive the check the way Lightning does: backward, hook, optimizer step."""
    model._begin_gradient_coverage()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        model._step(_make_batch(task, [SEQLEN, SEQLEN]))["loss"].backward()
        model.on_after_backward()
        optimizer.step()
    model.zero_grad(set_to_none=True)


def test_gradient_coverage_passes_on_a_healthy_model(train_model, caplog):
    from confrover.model.train import GRAD_COVERAGE_STEPS

    with caplog.at_level(logging.INFO):
        _run_coverage(train_model, GRAD_COVERAGE_STEPS)
    assert "received gradient" in caplog.text
    assert train_model._grad_coverage_done is True


def test_being_checkpoint_wrapped_is_not_by_itself_an_error(train_model):
    """The trunk, the IPA linears and the Llama layers are wrapped by design.

    A guard that scanned for precheckpoint_forward would condemn all of them.
    Their inputs require grad, so their checkpoints take part in autograd.
    """
    wrapped = [
        name
        for name, module in train_model.named_modules()
        if hasattr(module, "precheckpoint_forward")
    ]
    assert len(wrapped) > 50, f"expected the model to be widely wrapped, got {wrapped}"
    train_model._assert_trainable()  # must not raise


def test_gradient_coverage_catches_the_embedder_being_recheckpointed(train_model):
    """The original defect, reproduced: 20 tensors silently frozen."""
    from confrover.model.train import GRAD_COVERAGE_STEPS
    from confrover.model.utils.checkpoint_activations import (
        checkpoint_wrapper,
        unwrap_checkpoint,
    )

    embedder = train_model.decoder.model_nn.embedder
    checkpoint_wrapper(embedder, offload_to_cpu=True)
    try:
        with pytest.raises(RuntimeError, match="received no gradient"):
            _run_coverage(train_model, GRAD_COVERAGE_STEPS)
    finally:
        unwrap_checkpoint(embedder)
    # and the model still trains once it is put back
    _run_coverage(train_model, 2)


def test_a_severed_module_is_named_by_its_outermost_dead_parent(train_model):
    """Report `decoder...embedder`, not its 20 leaves."""
    trainable = {n for n, p in train_model.named_parameters() if p.requires_grad}
    seen = {n for n in trainable if "embedder" not in n}
    severed = train_model._severed_modules(seen)
    names = [name for name, _ in severed]
    assert names, "the embedder should read as severed when nothing reached it"
    assert any(name.endswith("embedder") for name in names), names
    # no child of a reported module is also reported
    for name in names:
        assert not any(name.startswith(f"{other}.") for other in names if other != name)


def test_coverage_waits_before_judging(train_model):
    """One step is one sample of one task; a branch can be absent from it."""
    from confrover.model.train import GRAD_COVERAGE_STEPS

    assert GRAD_COVERAGE_STEPS > 1
    _run_coverage(train_model, GRAD_COVERAGE_STEPS - 1)
    assert train_model._grad_coverage_done is False


def test_the_check_is_inert_until_fit_arms_it(train_model):
    """Inference and ad-hoc use must not pay for it or trip over it."""
    train_model.__dict__.pop("_grad_coverage_done", None)
    train_model.on_after_backward()  # must not raise, must not measure
    assert "_grad_coverage_done" not in train_model.__dict__


def test_assert_trainable_rejects_a_missing_embedder(train_model, monkeypatch):
    """A rename used to disable the fix and its guard together, silently."""
    monkeypatch.setattr(type(train_model), "_embedder", lambda self: None)
    with pytest.raises(RuntimeError, match="embedder is missing"):
        train_model._assert_trainable()


def test_assert_trainable_rejects_a_still_wrapped_embedder(train_model):
    from confrover.model.utils.checkpoint_activations import (
        checkpoint_wrapper,
        unwrap_checkpoint,
    )

    embedder = train_model.decoder.model_nn.embedder
    checkpoint_wrapper(embedder, offload_to_cpu=True)
    try:
        with pytest.raises(RuntimeError, match="still activation-checkpointed"):
            train_model._assert_trainable()
    finally:
        unwrap_checkpoint(embedder)


def test_enable_decoder_training_reports_what_it_unwrapped(train_model):
    """Silence used to be indistinguishable from 'found nothing to unwrap'."""
    from confrover.model.utils.checkpoint_activations import checkpoint_wrapper

    assert train_model.enable_decoder_training() == []  # already unwrapped
    checkpoint_wrapper(train_model.decoder.model_nn.embedder, offload_to_cpu=True)
    unwrapped = train_model.enable_decoder_training()
    assert unwrapped and any("embedder" in name for name in unwrapped)
    assert train_model.enable_decoder_training() == []


def test_coverage_catches_a_severance_the_wrapper_scan_cannot_see(train_model):
    """The point of measuring instead of sniffing.

    The old guard tested one module for one attribute. Any other way of cutting
    the graph -- a detach, a no_grad region, an unused head, requires_grad --
    passed it silently. Here the embedder is fine and nothing is re-wrapped, so
    the structural check is happy; only reading the gradient finds this.
    """
    from confrover.model.train import GRAD_COVERAGE_STEPS

    target = train_model.encoder.aatype_embedding
    handle = target.register_forward_hook(lambda mod, inp, out: out.detach())
    try:
        train_model._assert_trainable()  # structurally spotless
        with pytest.raises(RuntimeError, match="aatype_embedding"):
            _run_coverage(train_model, GRAD_COVERAGE_STEPS)
    finally:
        handle.remove()
    _run_coverage(train_model, 2)  # healthy again once the cut is removed


def test_a_forward_probe_batch_is_accepted_by_a_real_step(train_model):
    """_step raises unless a forward batch carries exactly 2 source frames."""
    from confrover.utils.torch.tflops import probe_train_batch

    out = train_model._step(
        probe_train_batch(seqlen=SEQLEN, device="cpu", task_mode="forward")
    )
    assert torch.isfinite(out["loss"])
    train_model.zero_grad(set_to_none=True)


def test_a_forward_step_costs_more_than_an_iid_one(train_model):
    """The probe measured only iid, which is the cheaper task, for every number
    this repo derives from it: tflops/step, the dead-unit census, the
    checkpointing audit."""
    from confrover.utils.torch.tflops import measure_train_step_tflops_by_task

    by_task = measure_train_step_tflops_by_task(train_model, seqlen=SEQLEN)
    assert by_task["forward"] > by_task["iid"], by_task
    train_model.zero_grad(set_to_none=True)


# =============================================================================
# The upstream silent-freeze warning was unreachable
# =============================================================================


def test_the_silent_freeze_warning_actually_fires():
    """`if torch.is_grad_enabled()` inside Function.forward is always False.

    That made torch's own "None of the inputs have requires_grad=True" warning
    dead code for every one of the 72 wrappers, which is how the embedder lost
    gradient on 20 tensors for a whole fine-tune without a word in the log.
    """
    from confrover.model.utils.checkpoint_activations import checkpoint_wrapper

    module = checkpoint_wrapper(torch.nn.Linear(4, 4), offload_to_cpu=False)
    frozen = torch.randn(2, 4)  # requires_grad=False: nothing to back-propagate to
    with pytest.warns(UserWarning, match="requires_grad"):
        module(frozen)


def test_the_warning_stays_quiet_when_there_is_a_gradient_to_have():
    from confrover.model.utils.checkpoint_activations import checkpoint_wrapper

    module = checkpoint_wrapper(torch.nn.Linear(4, 4), offload_to_cpu=False)
    live = torch.randn(2, 4, requires_grad=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        module(live)


def test_the_warning_stays_quiet_under_no_grad():
    """Validation and sampling run under no_grad; nothing is wrong there."""
    from confrover.model.utils.checkpoint_activations import checkpoint_wrapper

    module = checkpoint_wrapper(torch.nn.Linear(4, 4), offload_to_cpu=False)
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("error")
        module(torch.randn(2, 4))


def test_the_parameterless_ipa_ops_are_no_longer_wrapped(train_model):
    """Measured to save exactly 0 bytes, at both iid and forward shapes."""
    ipa = train_model.decoder.model_nn.structure_module.trunk["ipa_0"]
    assert not hasattr(ipa.softmax, "precheckpoint_forward")
    assert not hasattr(ipa.softplus, "precheckpoint_forward")
    # the coarse blocks are load-bearing and stay wrapped
    assert hasattr(
        train_model.decoder.model_nn.structure_module.trunk["seq_tfmr_0"],
        "precheckpoint_forward",
    )


# =============================================================================
# Checkpoint selection on the forward task
# =============================================================================


class _ProbeVal(torch.utils.data.Dataset):
    """Alternating iid/forward batches, exactly as the 40/40 val split does."""

    def __init__(self, n: int = 4) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        from confrover.utils.torch.tflops import probe_train_batch

        return probe_train_batch(
            seqlen=SEQLEN, device="cpu", task_mode="iid" if i % 2 == 0 else "forward"
        )


def test_val_loss_is_logged_per_task_into_callback_metrics(train_model, tmp_path):
    """The monitor names val/loss_forward; if nothing logs it, selection is dead.

    ModelCheckpoint warns once about a missing monitor and then never saves for
    the rest of the run, so this contract has to be checked against a real
    Trainer rather than by reading the logging call.
    """
    import lightning as L

    from confrover.train import BEST_CHECKPOINT_MONITOR, _build_best_val_checkpoint

    cb = _build_best_val_checkpoint(tmp_path)
    trainer = L.Trainer(
        accelerator="cpu", devices=1, logger=False, max_epochs=1,
        enable_progress_bar=False, enable_model_summary=False,
        num_sanity_val_steps=0, default_root_dir=str(tmp_path), callbacks=[cb],
    )
    loader = torch.utils.data.DataLoader(
        _ProbeVal(4), batch_size=None, collate_fn=lambda x: x
    )
    trainer.validate(train_model, dataloaders=loader, verbose=False)

    metrics = trainer.callback_metrics
    assert "val/loss_iid" in metrics, sorted(metrics)
    assert "val/loss_forward" in metrics, sorted(metrics)
    assert BEST_CHECKPOINT_MONITOR in metrics
    # the two tasks are scored separately, not collapsed into one number
    assert metrics["val/loss_iid"] != metrics["val/loss_forward"]
    train_model.zero_grad(set_to_none=True)
