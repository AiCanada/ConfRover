# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased]

Local DPF fine-tune work on top of official ConfRover v1.0 (inference-only).

### Added

- Cloud run for the PDB-cluster corpus: `stage_remote_payload.py --catalog` ships a
  catalog JSON by member path (required for the unique catalog, whose merged
  families span several cluster directories), `--weights` ships one chosen
  weights file, `--checkpoint none` stages a fresh run;
  `scripts/vast_bootstrap_pdbcluster.sh` installs, downloads, verifies, smoke-tests
  and launches `PDBcluster_from_base` (original base weights, `--window_frames 9`,
  `--ckpt_prefix PDBcluster`, the edited 1,442/168/68 split) with the checkpoint
  sync watcher; `runsPDB/stage_PDBcluster_payload.ps1` stages it locally.
- `--window_frames W` (default 9, the paper's pre-training window): each
  training example carries W frames of one protein. forward: W frames of one
  trajectory at one stride from the 1-1024 ladder (PDB clusters: W distinct
  structures, gap 0); the temporal module gets W tokens and every token's state
  predicts its own frame, so a step trains W predictions with 0..W-1 frames of
  context instead of one prediction with at most one. iid: W context-free
  targets sharing one trunk pass. `TrainExample.window`, window-aware
  `_build_sample`/`collate` (leading frame axis) and `_step` (frame axis folded
  into the decoder batch). `--window_frames 1` reproduces the earlier runs
  exactly; `frames_per_step` is logged. Checkpoints record `window_frames` in
  the bag state.
- `--ckpt_prefix NAME` (default `dpf`): first token of every checkpoint file
  name -- `NAME-epoch{E}-step{S}`, `NAME-epoch{E}-end`, `NAME-bestfwd-step{S}`,
  `NAME-stopped-step{S}`. `--resume epoch` looks for the same prefix. Lets a
  PDB-cluster run (`--ckpt_prefix PDBcluster`) be told apart from an ATLAS run
  by its files alone.
- `confrover train`: fine-tune **ConfRover-base-20M-v1.0** on Dual Personality Fragments (`iid` + `forward` only).
- Family catalog, group split, and family-bag sampler (`src/confrover/data/dpf/`). Split is by family / seqres / PDB identity — never by conformation.
- `ConfRoverTrain` Lightning module, ConfDiff score+torsion+atom14 loss, cosine LR schedule.
- Vendored OpenFold at `src/confrover/_ext/openfold` (no `import openfold`).
- `scripts/newpdbidlistrain.py`: 303 novel ATLAS PDB IDs with L ≤ 384.
- `tests/dpf/`: catalog, leakage, sampler, loss, native `from_config`.
- Transformers 5.x compatibility in `model/temporal/llama.py` (optional SinkCache).
- `--accumulate_grad_batches`: the only way to raise the effective batch, since the fused token axis `L + L²` pins `--batch_size` to 1 on an 8 GB card.
- `--repr_cache_size` (default 8 → 32): OpenFold representations held per dataloader worker.
- `--resume epoch` plus retained `dpf-epoch{N}-end.ckpt` from a dedicated epoch-boundary `ModelCheckpoint`.
- Best-val checkpoint (`dpf-best-step{N}.ckpt`, monitors `val/loss`) kept separately from the recovery rollover.
- `ResumableDataLoader` (`data/resumable_loader.py`): mid-epoch resume continues the same permutation instead of reshuffling; implements Lightning's `_Stateful` protocol.
- `TimedModelCheckpoint`: logs size and duration of every checkpoint write, warning past 30 s.
- `utils/torch/tflops.py`: train-step TFLOP accounting, probed at the run's median sequence length with backward included.
- `StepHeartbeat` per-term breakdown (`trans/rot/torsion/atom14`), task mix, mean `L`, and `atom14_on` gate coverage.
- `model/utils/dead_units.py` + `--rescale_attention N` (default 8): decoder capacity repair. Rescales structure-module attention layers whose output swamps their residual, then Net2Net-splits any feed-forward units still dead.
- `ReleaseValidationMemory`: `empty_cache()` after each validation, so the val loop's reserved blocks do not tip the process into the Windows CUDA system-memory fallback.
- `scripts/export_finetuned_weights.py`: turn any Lightning `*.ckpt` into a `confrover generate` weights file. `train` only writes one at the end of `fit`, so a stopped run previously left its weights unusable.
- Heartbeat reports `mem=allocated/reserved` VRAM.
- `tests/dpf/`: `test_resumable_loader.py`, `test_generate_cli.py`, `test_metrics_handler.py`, `test_nested_tensor_fast_path.py`, `test_tflops.py`, and others — suite grew to 355 tests.

### Changed

- Log header reads `Load PDB clusters catalog` when the source is the cluster
  store, `Load DPF catalog` otherwise.
- PDB-cluster family stores renamed to say what they are: `dpf_families_95_over10`
  -> `pdb_clusters_95_over10_cap100` (95% identity clusters with >10 unique PDB
  ids, admitted up to `MAX_ADMIT = 100` members each) and `dpf_families` ->
  `pdb_clusters_95_max10` (the 54 clusters the 10-id cap kept). They are not
  Dual Personality Fragments; `--dpf_root` remains the generic family-store
  flag. Script defaults, tests, launch scripts and the three catalog JSONs
  (absolute `pdb_path`s) were rewritten.
- Val loader under `--one_pass_frames`: `pin_memory=False`, `persistent_workers=True`
  (was the reverse). One-pass sets `reload_dataloaders_every_n_epochs=1`, and
  pin_memory + persistent workers + reload is PyTorch #91252; Lightning's fix is to
  drop the pin thread, which val can spare at batch_size 1. Respawning the pool
  instead cost 4 spawn-mode workers per validation, ~168 times per 33k-step epoch.
- CLI: `query_msa`, `openfold_repr`, `generate`, `train`.
- `--max_epochs` default is 3. `--family_excludelist auto` drops the 1,080 base-train ATLAS IDs.
- Windows-safe logging (no emoji on redirected cp1252 streams).
- Dependency ranges allow Python 3.10 + torch 2.1 / transformers 4.41 and Python 3.13 + torch 2.13 / transformers 5.14.
- `--resume last` selects the checkpoint with the highest `global_step`, not the file named `last.ckpt`. Lightning uniquifies a clashing `save_last` target, so a stale `last.ckpt` from an earlier run silently rewound training by 616 steps.
- `save_last="link"` instead of `True`: `save_last=True` serialised a second full 236 MB copy per checkpoint event.
- Train and val loaders both set `persistent_workers=True`. `DpfTrainDataset.set_epoch` is monotonic and writes a shared-memory epoch that persistent workers re-read in `__getitem__`, so later epochs do not replay any earlier bag (including when a new DataLoader restarts its sampler at 0).
- Every Lightning checkpoint now stores the DPF bag epoch and loader cursor (`dpf_restart` + `restart.json` sidecar). `STOP` / `PAUSE` / Ctrl+C finish the current step and write that state. `--resume auto` is the default, so the same command continues a long run with no change to the remaining epoch.
- Run logs: UTF-8 `console.log` captures stdout/stderr (print, Rich, `\r` bars become new lines, no NULs). DataLoader workers re-attach `debug.log`. Do not use PowerShell `> train.log` — that file is UTF-16.
- Interval and best-val checkpoints use `save_top_k=-1`. Older `.ckpt` files are no longer deleted when a newer one is written.
- `nn.TransformerEncoder(enable_nested_tensor=False)` in the structure module, so validation uses the same kernels as training (torch's nested fast path is skipped whenever grad is enabled).
- `torch.cuda.amp.autocast` → `torch.amp.autocast("cuda", ...)` at 10 call sites.
- `confrover generate` no longer requires DeepSpeed: the Trainer strategy and `--use_kernel` are resolved against what is installed.
- OpenMM/pdbfixer/openmmtools moved from hard dependencies to a `relax` extra — nothing under `confrover` imports them and two are git URLs.

### Fixed

- The TFLOP probe measured the single-target batch whatever `--window_frames`
  was, so a W=9 run reported about a ninth of a real step and explained the
  forward/iid ratio with "two source frames". It now builds the window batch
  the run trains on and the report names the frames per step.
- **OpenFold repr store: index scan and writes are now crash-safe.** A run that
  died between the npy and meta writes left a directory the scan turned into an
  empty `,` row in `seqres_to_index.csv` (read back as NaN -> NaN); one that died
  mid-`json.dump` left a meta that made every later `OpenFoldReprLoader()` raise
  `JSONDecodeError` until the file was removed by hand. Both are now skipped with
  a warning. `save_index_file` and `build_index_file(save=True)` write via
  temp + `os.replace`, matching `MSALoader`, so a torn index can no longer
  regenerate the whole store.
- `OpenFoldReprLoader.check_cache` requires the `recycle{N}` npy files, not just
  the directory: a store built at a different `--num_recycles` read as cached
  and failed in `load` mid-training.
- `confrover openfold_repr --overwrite` no longer deletes and re-downloads the
  MSAs for the sequences it regenerates; `overwrite` was forwarded into
  `MSALoader.query_msa`.
- Index names go through `Path.name`, not `Path.stem`, in the MSA index rebuild
  and in both `unique_dir(...)` collision renames (a3m and repr). A dotted
  index (`1abc.A`) was truncated to `1abc` -- dropped from the rebuild, and on a
  collision the returned index named a directory that was never written.
- **The capacity repair was decided by a coin flip.** The saturation ratio is an
  activation statistic, so which layers crossed `SATURATION_RATIO = 10.0` depended
  on which batches the probe happened to draw: `seq_tfmr_1.layers.0` measured
  4.49-20.67 across draws and fell below the threshold in 2 of 20. Identical
  weights, different permanent model. The probe is now a pinned synthetic draw
  spanning both tasks, and the diffusion timestep is pinned per batch index --
  without the second pin the ratio still moved 55.4-57.6 between identical runs,
  because `_step` samples `t` on every call. Verified bit-identical across three
  runs with the global RNG deliberately disturbed between them. Both pins save and
  restore the global RNG, so the run's own schedule is untouched. Real batches are
  still used when `--split_dead_units` is on, where the census drives a decision.
- **The dead-unit census ran three times to produce a number that decides nothing.**
  With the split off it only fed a log line, and that line moved 21 units between
  two measurements of *unchanged* weights (`1023 -> 899 -> 920`) because activation
  sampling can only prove a unit fired for the inputs it saw. `repair_decoder_capacity`
  now takes `census=`, and the third census is skipped entirely when nothing was split.
- **A clean stop followed by a crash reported itself as a crash.** Windows delivers
  Ctrl+C to the whole console process group, so the DataLoader workers die while the
  parent's handler finishes the step and saves; Lightning then resets the loader,
  finds them gone, and raises. Observed on v888: `saved dpf-stopped-step00000026.ckpt`
  at 19:51:04, `DataLoader worker ... exited unexpectedly` at 19:51:11, manifest
  `failed`. `run_train`'s `except BaseException` now preserves a completed stop's
  status instead of overwriting it, and returns rather than re-raising. A crash with
  no preceding stop still reports `failed` and re-raises.
- **The decoder capacity repair ran again on every resume, and compounded.**
  Lightning restores checkpoint weights at `trainer.py:1046` and only reaches
  `on_fit_start` at `:1057`, so on a resumed run `SaturatedAttentionRescale`
  rewrote *trained* weights rather than base ones -- with no guard, on every
  restart. Measured across one resume: 24 of 899 tensors rewritten, 445 of 2560
  FFN units overwritten (`||dW||/||W||` up to 1.33), and effective FFN width
  (distinct features, >0.99-cosine twins counted once) 1661 -> 1563, of which
  1535 subsequent training steps recovered **one**. Donors are the
  highest-outgoing-norm live units, reused round-robin, so each round halves the
  strongest columns again (2.7997 -> 1.3993 -> 0.6981) -- capacity decays
  geometrically in the number of restarts while the loss stays flat
  (+0.000087 +- 0.000135). The repair now runs once per lineage, on the fresh
  run only.
- **The Net2Net split is off by default** (`--split_dead_units`, was
  unconditional). One round on the base weights cost 899 of 2560 distinct FFN
  features. The clones never separate: twin cosine moved 0.999940 -> 0.999924
  over 1535 steps, roughly 1e6 steps from independence, because Adam gives
  identical units identical updates. The attention rescale it follows is kept --
  that one has a measured root cause (43-110x attention/residual on the base
  weights) and does not compound (0 layers rescaled on the resume, all ratios
  0.3-2.3 against a threshold of 10).
- **`repair_decoder_capacity`'s function-preservation claim was false.** On real
  DPF batches the split moves `pred_atom14` by a median 5.8e-4 relative, max
  3.6e-2, **max single-atom 7.7 A** -- ~6000x fp32 roundoff. The docstring said
  "the layer computes exactly what it did before". Corrected, with the numbers.
- **The `forward` task was never measured, and it is where the run is slow.**
  `probe_train_batch` hardcoded `task_mode="iid"`, so every figure derived from
  it described the cheaper of the two tasks a run trains on -- the heartbeat's
  `tflops/step`, the dead-unit census, the attention-rescale probe. Measured at
  L=249 on the 8 GiB card: **iid 4.335 TFLOP / 2.54 s / 4.70 GiB reserved;
  forward 8.089 TFLOP / 16.65 s / 9.04 GiB reserved.** A forward step is 1.87x
  the FLOPs but **6.5x the wall time**, because 9.04 GiB does not fit in the
  6.93 GiB the allocator gets and Windows spills to host instead of raising
  OOM. The live mix is roughly half forward, which is what the 14-50 s/step and
  the 8.4-17.4 GiB heartbeat readings were. The probe now builds either task,
  `measure_train_step_tflops_by_task` measures both, and the heartbeat prices
  each window by the mix it actually ran instead of quoting the iid number.
- **Torch's silent-freeze warning was unreachable in all 72 checkpoint
  wrappers.** `CheckpointFunction.forward` gated
  `checkpoint.check_backward_validity` behind `if torch.is_grad_enabled():`,
  which autograd guarantees is False inside `Function.forward`. The caller's
  grad mode is now sampled in `_checkpointed_forward` and passed through, so
  the warning fires when training and stays quiet under `no_grad`. This is the
  guard that should have caught the embedder losing gradient on 20 tensors.
- **Dropped the two parameterless `checkpoint_wrapper`s** on `nn.Softmax` and
  `nn.Softplus` (`structure_module.py`, 8 instances). Measured to save exactly
  0 bytes of peak allocated at both iid and forward shapes while each cost an
  `autograd.Function` boundary, an RNG snapshot, and one more silent-freeze
  trap. The ~20 coarse blocks stay wrapped: removing all 72 takes peak
  allocated 3.62 -> 6.84 GiB and spills 3.1 GiB to host.
- **The frozen-embedder guard could not see the thing it was guarding.**
  `_assert_trainable` tested one hardcoded module for one wrapper attribute,
  using the same `getattr` chain as the fix it protected -- so a rename would
  disable the fix and its guard together, silently. It also could not see any
  other way of cutting the graph (`detach`, `no_grad`, `requires_grad`, an
  unused head), and `checkpoint_wrapper` is applied to 50+ modules on purpose,
  so "is it wrapped" was never the right question. Replaced by
  `_check_gradient_coverage`, which reads the gradient itself over the first
  `GRAD_COVERAGE_STEPS` (10) training steps and raises naming any module whose
  parameters all went untouched. Measured on the real model: all 898 trainable
  tensors are reached on step 0, so the window is a 10x margin. `_assert_trainable`
  is kept as a cheap pre-flight and now also fails when the embedder is missing.
- **Resuming from an epoch boundary trained an empty epoch.** Lightning's
  `_TrainingEpochLoop` stops as soon as `batch_progress.ready` reaches
  `num_training_batches` and never advances the fetcher again, so
  `ResumableDataLoader.__iter__` was left suspended at its final `yield` and the
  epilogue after the loop never ran. The cursor was checkpointed as
  `(epoch N, every batch consumed)` and applied verbatim to epoch N+1, which
  then yielded **zero** batches. Measured on a real `Trainer`: the resumed epoch
  saw 0/20 samples before, 20/20 after. The cursor is now normalised wherever it
  is read, so checkpoints already on disk resume correctly too.
- **A mid-epoch stop wrote `dpf-epoch{N}-end.ckpt`.** Lightning still runs
  `on_train_epoch_end` for an epoch cut short by `should_stop`, so the file
  landed at the same `global_step` as `dpf-stopped-step*.ckpt` under a name
  claiming a boundary the run never reached -- and `--resume epoch` trusts that
  name. `EpochBoundaryCheckpoint` now saves only when the epoch ran every batch.
- **A stopped run reported itself as `completed`.** `trainer.fit()` returns
  normally after `should_stop`, so `run_train` fell through to export
  `confrover_base_dpf.pt` and stamp the manifest `completed`, erasing the
  `interrupted`/`paused` status `GracefulStop` had just written. Every abandoned
  run read as a successful one.
- **The load-failure budget was per worker, not per corpus.** Each persistent
  DataLoader worker holds its own copy of the dataset and so its own
  `_failed_samples` set, silently multiplying `max_load_failures` by
  `num_workers`; on a split small enough that no single worker reached the limit
  (validation: 80 samples over 4 workers) the guard could not fire at all --
  measured at 30 corrupt samples over 30 passes, per-worker tallies
  `[10, 6, 6, 9]`, no error, versus raising at 21 with `num_workers=0`. The
  tally now lives in shared memory.
- Decoder embedder reentrant checkpoint blocking gradients.
- Score terms divided by diffuser `score_scaling`; atom14 gated to low `t`.
- Shared topology / dual pdb+xtc catalog leakage holes.
- Epoch-end steps were unrecoverable: setting `every_n_train_steps` disables Lightning's epoch-end save, so a run ending at step 1216 had its newest checkpoint at 1200.
- `MetricsHandler` reported `nan` for train metrics during the pre-training sanity validation, and logged that `nan` into `callback_metrics` where a checkpoint monitor would compare against it.
- Gate-weighted `atom14` is averaged by its open fraction; a fully gated window reports `n/a` rather than a misleading `0.0`.
- TFLOP probe measured forward-only at L=48, understating a real step by ~80x; it now runs forward+backward at the run's median length inside a saved RNG state.
- `pytest` could not run: `addopts = "-n 4"` with `pytest-xdist` absent.

### Known issues

- **Throughput collapses when the reserved pool exceeds the card.** Root cause: the Windows CUDA system-memory fallback silently serves allocations from host RAM over PCIe rather than raising OOM. Measured 14.6 GiB "allocated" on an 8.2 GiB card, with the GPU at 100% utilisation, ~20 W and ~0% memory-controller traffic. Once reserved reaches that state it does not recover: a step needing 5.8 GiB still ran at 50 s/step. Pure model compute is 3.2 s/step. Ruled out by measurement: dataloader workers (16 worse than 4), representation cache size, checkpoint frequency, `expandable_segments:True` (identical to baseline), disk, throttling and host RAM.
- **Four of the eight structure-module transformer layers leave FFN units with exactly zero gradient in the base checkpoint.** Measured on CUDA at L=249 over 8 batches: `seq_tfmr_1.layers.1.linear1.weight`, `.linear1.bias` and `.linear2.weight` carry gradient `0.000e+00`, so those units cannot learn during any fine-tune. `--rescale_attention` restores them to `1.224e+00` / `3.224e-02` / `1.064e-01`, and `seq_tfmr_1.layers.0.linear1.weight` by 169x, while layers it does not touch move 0.97-1.00x. **The mechanism previously recorded here was wrong**: this entry used to say LayerNorm emits a token-invariant vector because attention dominates the residual. LayerNorm is positively scale-invariant, so a large ratio erases no per-token signal -- measured token-spread 0.9972 at ratio 110 against a residual-only baseline of 0.9970 -- and `seq_tfmr_1.layers.1`, cited as the proof case, is the second most token-varying of the eight. The units are dead because their pre-activations are all negative; the ratio is a detector that happens to find them. A paired A/B over 200 steps still showed **no measurable learning benefit** (val 0.34418 vs 0.34359, inside a 2.1e-3 sem), so the case for the rescale rests on restored gradient, not on a demonstrated loss improvement.
- **DPF fine-tuning shows no measurable effect** on held-out denoising loss: +0.00116 ± 0.00215 over 80 paired samples (t = 0.54) despite 2.86% relative weight displacement.
- **Two-state coverage is poor in the base model and unchanged by fine-tuning.** Against basins found in the reference MD, generated ensembles badly under-populate the minority state (4laf_A: reference 46%, generated 4%). Pooled minority occupancy base 41.7% vs fine-tuned 40.6%, Fisher p = 1.00.
- Generation OOMs at `--batch_size 8` for L >= 340.

## [v0.1.0] - 2025-11

Official ConfRover v1.0: generate (forward, IID, interp), OpenFold repr CLI, ATLAS-oriented inference.
