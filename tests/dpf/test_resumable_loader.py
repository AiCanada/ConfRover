# Copyright 2025 Bytedance Ltd. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Mid-epoch resume must not reshuffle.

Resuming a plain DataLoader restarts it on a fresh permutation, so the samples
already consumed in that epoch are drawn a second time and the ones that were
still pending are never drawn. Lightning warns and continues. These tests pin
the property that matters: across an interruption, the epoch still covers every
sample exactly once.
"""

from __future__ import annotations

import torch
from lightning.fabric.utilities.types import _Stateful
from torch.utils.data import DataLoader, Dataset

from confrover.data.datamodule import ConfRoverDataModule
from confrover.data.loader_config import LoaderConfig
from confrover.data.resumable_loader import ResumableDataLoader

N, BATCH = 20, 2
FULL_BATCHES = N // BATCH


class _Indices(Dataset):
    def __len__(self) -> int:
        return N

    def __getitem__(self, index: int) -> int:
        return index


def _loader(seed: int = 7, shuffle: bool = True) -> ResumableDataLoader:
    return ResumableDataLoader(
        _Indices(), seed=seed, shuffle=shuffle, batch_size=BATCH, num_workers=0
    )


def _drain(loader, limit: int | None = None) -> list[int]:
    seen: list[int] = []
    for i, batch in enumerate(loader):
        seen.extend(batch.tolist())
        if limit is not None and i + 1 >= limit:
            break
    return seen


def test_lightning_recognises_it_as_resumable():
    assert isinstance(_loader(), _Stateful)


def test_closing_an_incomplete_iterator_does_not_advance_the_epoch():
    """STOP/PAUSE close the iterator; the cursor must stay on this epoch."""
    loader = _loader()
    iterator = iter(loader)
    for _ in range(3):
        next(iterator)
    iterator.close()
    state = loader.state_dict()
    assert state["epoch"] == 0
    assert state["batches_consumed"] == 3


def test_an_interrupted_epoch_still_covers_every_sample_exactly_once():
    """The whole point: no replay, no gap."""
    first = _loader()
    before = _drain(first, limit=3)
    state = first.state_dict()
    assert state["batches_consumed"] == 3

    second = _loader()
    second.load_state_dict(state)
    after = _drain(second)

    assert len(before) == 3 * BATCH
    assert len(after) == (FULL_BATCHES - 3) * BATCH
    assert sorted(before + after) == list(range(N))
    assert set(before).isdisjoint(after)


def test_the_resumed_half_is_the_tail_of_the_same_permutation():
    uninterrupted = _drain(_loader())

    first = _loader()
    before = _drain(first, limit=3)
    second = _loader()
    second.load_state_dict(first.state_dict())
    after = _drain(second)

    assert before + after == uninterrupted


def test_a_plain_dataloader_is_the_behaviour_this_replaces():
    """Guard the premise: a plain shuffled loader really does replay and skip."""
    torch.manual_seed(0)
    plain = DataLoader(_Indices(), batch_size=BATCH, shuffle=True)
    before = _drain(plain, limit=3)
    after = _drain(plain)  # a fresh iterator == a fresh permutation
    assert sorted(before + after) != list(range(N))
    assert not set(before).isdisjoint(after)


def test_len_reports_the_full_epoch_even_while_resuming():
    """num_training_batches and the cosine LR horizon are read from __len__."""
    loader = _loader()
    assert len(loader) == FULL_BATCHES
    loader.load_state_dict({"epoch": 0, "batches_consumed": 3})
    assert len(loader) == FULL_BATCHES


def test_the_permutation_is_reproducible_and_independent_of_global_rng():
    torch.manual_seed(1234)
    a = _drain(_loader(seed=7))
    torch.manual_seed(999)
    for _ in range(50):
        torch.rand(10)
    b = _drain(_loader(seed=7))
    assert a == b


def test_each_epoch_draws_a_different_order():
    loader = _loader()
    first = _drain(loader)
    second = _drain(loader)
    assert sorted(first) == sorted(second) == list(range(N))
    assert first != second


def test_resuming_targets_the_epoch_recorded_in_the_state():
    loader = _loader()
    _drain(loader)              # finish epoch 0 -> sampler advances to epoch 1
    assert loader.state_dict()["epoch"] == 1

    resumed = _loader()
    resumed.load_state_dict({"epoch": 1, "batches_consumed": 0})
    reference = _loader()
    _drain(reference)
    assert _drain(resumed) == _drain(reference)


class _EpochProbe(Dataset):
    """Reports the last set_epoch so we can see when the loader applied it."""

    def __init__(self) -> None:
        self._epoch = 0
        self.epochs: list[int] = []

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        self.epochs.append(self._epoch)

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> int:
        return self._epoch


def test_iter_pushes_each_sampler_epoch_into_the_dataset():
    """Persistent workers prefetch on iter(); set_epoch must run first."""
    dataset = _EpochProbe()
    loader = ResumableDataLoader(
        dataset, seed=0, shuffle=False, batch_size=1, num_workers=0
    )
    for expected in range(4):
        assert _drain(loader) == [expected] * 4
    assert dataset.epochs == [0, 1, 2, 3]


def test_a_fresh_loader_does_not_rewind_a_later_dataset_epoch():
    """Replacement DataLoaders start their sampler at 0; that must not replay."""
    dataset = _EpochProbe()
    dataset.set_epoch(3)
    dataset.epochs.clear()
    loader = ResumableDataLoader(
        dataset, seed=0, shuffle=False, batch_size=1, num_workers=0
    )
    assert _drain(loader) == [3, 3, 3, 3]
    assert dataset.epochs == [3]


class _RestartDS(_Indices):
    """Index dataset with the hooks ConfRoverDataModule persists."""

    def __init__(self) -> None:
        self._epoch = 0
        self._sample_seed = 7
        self.loader_cfg = LoaderConfig(
            batch_size=BATCH, num_workers=0, shuffle=True
        )
        self.collate = None

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def state_dict(self) -> dict:
        return {"epoch": int(self._epoch), "sample_seed": int(self._sample_seed)}

    def load_state_dict(self, state: dict) -> None:
        self.set_epoch(int(state["epoch"]))


def test_datamodule_checkpoint_continues_the_same_epoch_without_replay():
    """Every .ckpt carries bag epoch + loader cursor; resume must not reshuffle."""
    live = ConfRoverDataModule(train_dataset=_RestartDS())
    before = _drain(live.train_dataloader(), limit=3)
    saved = live.state_dict()
    assert saved["train_loader"]["batches_consumed"] == 3
    assert saved["train_dataset"]["epoch"] == 0

    resumed = ConfRoverDataModule(train_dataset=_RestartDS())
    resumed.load_state_dict(saved)
    after = _drain(resumed.train_dataloader())
    assert set(before).isdisjoint(after)
    assert sorted(before + after) == list(range(N))
    assert before + after == _drain(_loader(seed=7))


def test_without_shuffle_the_order_is_sequential_and_still_resumable():
    loader = _loader(shuffle=False)
    before = _drain(loader, limit=3)
    assert before == list(range(3 * BATCH))
    resumed = _loader(shuffle=False)
    resumed.load_state_dict(loader.state_dict())
    assert _drain(resumed) == list(range(3 * BATCH, N))


# =============================================================================
# A finished epoch that never raises StopIteration
# =============================================================================


def _lightning_style_epoch(loader) -> list[int]:
    """Drive exactly len(loader) batches and stop, as Lightning's loop does.

    ``_TrainingEpochLoop.done`` is true once ``batch_progress.ready`` reaches
    ``num_training_batches``, so the fetcher is never advanced past the last
    batch: the generator stays suspended at its final ``yield`` and any
    epilogue after the for-loop never runs.
    """
    iterator = iter(loader)
    seen: list[int] = []
    for _ in range(len(loader)):
        seen.extend(next(iterator).tolist())
    return seen


def test_a_finished_epoch_reads_as_the_start_of_the_next_one():
    loader = _loader()
    assert len(_lightning_style_epoch(loader)) == N
    state = loader.state_dict()
    assert state["epoch"] == 1
    assert state["batches_consumed"] == 0


def test_resuming_a_finished_epoch_yields_a_full_next_epoch_not_zero_batches():
    """The defect: the cursor was applied to the next epoch, emptying it."""
    loader = _loader()
    _lightning_style_epoch(loader)

    resumed = _loader()
    resumed.load_state_dict(loader.state_dict())
    drawn = _drain(resumed)
    assert len(drawn) == N
    assert sorted(drawn) == list(range(N))

    reference = _loader()
    _drain(reference)  # advance it to epoch 1 the exhausting way
    assert drawn == _drain(reference)


def test_a_suspended_epoch_does_not_replay_on_the_next_iter():
    """Lightning reuses one loader object for every epoch."""
    loader = _loader()
    first = _lightning_style_epoch(loader)
    second = _lightning_style_epoch(loader)
    assert sorted(first) == sorted(second) == list(range(N))
    assert first != second


def test_a_checkpoint_written_before_this_fix_still_resumes():
    """Cursors already on disk record a finished epoch as fully consumed."""
    stale = {"epoch": 0, "batches_consumed": FULL_BATCHES, "seed": 7}
    resumed = _loader()
    resumed.load_state_dict(stale)
    assert len(_drain(resumed)) == N


def test_a_partial_cursor_is_left_exactly_where_it_stopped():
    """The roll must not swallow a genuine mid-epoch stop."""
    loader = _loader()
    _drain(loader, limit=FULL_BATCHES - 1)
    state = loader.state_dict()
    assert state["epoch"] == 0
    assert state["batches_consumed"] == FULL_BATCHES - 1
    resumed = _loader()
    resumed.load_state_dict(state)
    assert len(_drain(resumed)) == BATCH


def test_a_prefetched_batch_is_not_counted_as_trained():
    """Lightning's fetcher pulls ahead; the cursor must follow the trainer.

    The loader can only see what it handed to the fetcher. Counting that as
    progress makes the next resume skip batches that were pulled but never
    trained -- and because the cursor is saved every checkpoint, the loss
    compounds across restarts.
    """
    loader = _loader()
    iterator = iter(loader)
    next(iterator)
    next(iterator)
    assert loader._batches_consumed == 2, "the fetcher pulled two batches"

    # what ConfRoverDataModule.state_dict passes on: no training step finished
    loader.note_trained_batches(0)
    assert loader.state_dict()["batches_consumed"] == 0

    resumed = _loader()
    resumed.load_state_dict(loader.state_dict())
    assert len(_drain(resumed)) == N, "a prefetched batch must still be trained"


def test_the_trainer_count_is_offset_by_a_mid_epoch_resume():
    """The trainer's per-epoch counter restarts at 0 on the shortened iterator."""
    loader = _loader()
    loader.load_state_dict({"epoch": 0, "batches_consumed": 5, "seed": 7})
    iterator = iter(loader)
    next(iterator)
    next(iterator)
    loader.note_trained_batches(1)  # one batch trained *in this run*
    assert loader.state_dict()["batches_consumed"] == 6


def test_a_stale_trainer_count_cannot_exceed_what_was_pulled():
    """Over-reporting skips untrained batches, so the loader's count is the ceiling."""
    loader = _loader()
    iterator = iter(loader)
    next(iterator)
    loader.note_trained_batches(99)
    assert loader.state_dict()["batches_consumed"] == 1


def test_an_injected_sampler_is_refused_with_an_actionable_message():
    """Lightning re-instantiates with sampler= when it injects a DistributedSampler.

    Left alone that collides with the sampler this class owns and surfaces as
    "got multiple values for keyword argument 'sampler'", which says nothing
    about what to do. Honouring it silently would be worse: mid-epoch resume is
    the entire point of the class.
    """
    import pytest

    with pytest.raises(ValueError, match="use_distributed_sampler"):
        ResumableDataLoader(
            _Indices(),
            seed=7,
            shuffle=True,
            batch_size=BATCH,
            sampler=torch.utils.data.SequentialSampler(_Indices()),
        )


def test_seed_and_shuffle_survive_as_attributes():
    """Lightning rebuilds __init__ arguments from same-named attributes."""
    loader = _loader(seed=11, shuffle=True)
    assert loader.seed == 11
    assert loader.shuffle is True


class _ShrinkingDS(_RestartDS):
    """A one-pass bag: epoch 0 has N items, epoch 1 has 3, epoch 2 none."""

    def set_epoch(self, epoch: int) -> None:
        # DpfTrainDataset.set_epoch is monotonic; the datamodule relies on it.
        if int(epoch) >= self._epoch:
            self._epoch = int(epoch)

    def __len__(self) -> int:
        return {0: N, 1: 3}.get(self._epoch, 0)

    def __getitem__(self, idx):
        return idx


def test_reloaded_train_loader_is_measured_on_the_current_epochs_bag():
    """Lightning reads len(loader) when it rebuilds the loader at an epoch
    boundary and only then runs on_train_epoch_start, where the bag used to
    switch. On the PDB-cluster run the cached count stayed at epoch 0's 7,864
    while epoch 1 held 539: wrong progress bar, wrong ETA, and every later
    epoch logged as 'stopped early' with no epoch-end checkpoint."""

    class _Trainer:
        current_epoch = 1

    dm = ConfRoverDataModule(train_dataset=_ShrinkingDS())
    dm.trainer = _Trainer()
    loader = dm.train_dataloader()
    assert dm.train_dataset._epoch == 1
    assert len(loader) == -(-3 // BATCH)

    # monotonic: a stale/lower trainer epoch never rewinds the bag
    _Trainer.current_epoch = 0
    dm.train_dataloader()
    assert dm.train_dataset._epoch == 1

    # no trainer attached (plain consumers): unchanged behaviour
    dm2 = ConfRoverDataModule(train_dataset=_ShrinkingDS())
    assert len(dm2.train_dataloader()) == -(-N // BATCH)
