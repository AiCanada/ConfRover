# Fine-tune on the PDB-cluster corpus (pdbc95_over10), starting from the
# v888 ATLAS run's step-8000 weights instead of the published base checkpoint.
#
#   init weights : runsPDB\init_v888_step8000\confrover_base_dpf.pt
#                  (exported from runs\dpf_base_train_v888\checkpoints\dpf-epoch007-step00008000.ckpt
#                   by scripts\export_finetuned_weights.py; the file name is the one
#                   train_policy accepts, the directory says where it came from)
#   output       : runsPDB\PDBcluster_from_v888
#   checkpoints  : PDBcluster-epoch{E}-step{S}.ckpt / PDBcluster-epoch{E}-end.ckpt /
#                  PDBcluster-bestfwd-step{S}.ckpt / PDBcluster-stopped-step{S}.ckpt
#
# Data / schedule settings are those of runs\dpf_base_train_pdbc95 (run_manifest.json).
#
# Training set: the PDB-cluster corpus only (no ATLAS families). --catalog takes
# precedence over --dpf_root (train.py:1332); the *_unique catalog holds every
# structure from the 1,684 pdb_clusters_95_over10_cap100 directories, merged by
# sequence into 1,678 families (6 clusters shared a seqres with another and were
# unioned into it -- scripts/dedup_pdbc95_catalog.py). The 80/10/10 group split
# (seed 0) gave train=1,342  val=168  test=168; 100 whole identity components
# were then moved test -> train in splits\0.json (val untouched, original kept
# as 0.json.bak-80-10-10), so the run trains on 1,442 / val 168 / test 68. The
# --*_frac values below are the realised fractions (1442/168/68 of 1678); they
# must match the persisted split or DpfSplit.load refuses it.
# --rescale_attention 0: that repair is for the *published* base weights; v888 has
# 8,000 trained steps on top of them and the SaturatedAttentionRescale record
# itself warns against re-running surgery on trained weights.
#
# The same command resumes the run (--resume auto). Drop a STOP or PAUSE file in
# the output directory to end it cleanly.
#
# NOTE: an 8 GB card fits one run. Stop any other `confrover train` first.
#
# --cache_dir / --folding_repr are passed explicitly because the default cache
# root is ./confrover_cache *relative to the working directory*: launched from
# the home directory, the excludelist and every representation silently
# resolve to an empty home cache (no excludelist -> base-trained families not
# excluded; no reprs -> RuntimeError before Fit).

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repo

py -3.13 -m confrover.cli train `
  --catalog  "$repo\confrover_cache\pdbc95_over10_catalog_unique.json" `
  --dpf_root "A:\ATLAS DATA\PDB_Cluster_Shards\pdb_clusters_95_over10_cap100" `
  --cache_dir    "$repo\confrover_cache" `
  --folding_repr "$repo\confrover_cache\folding_repr" `
  --model    "$repo\runsPDB\init_v888_step8000\confrover_base_dpf.pt" `
  --output   "$repo\runsPDB\PDBcluster_from_v888" `
  --ckpt_prefix PDBcluster `
  --window_frames 9 `
  --one_pass_frames true `
  --frac_split true --train_frac 0.859356 --val_frac 0.100119 --test_frac 0.040525 `
  --n_holdout 10 --n_val 5 `
  --iid_frame_stride 41 --forward_stride_frames 1-1024 `
  --samples_per_family 8 --static_iid_cap 36 `
  --max_seqlen 384 --max_epochs 3 --batch_size 1 `
  --num_data_workers 4 --repr_cache_size 128 `
  --lr 1e-4 --lr_schedule cosine --lr_warmup_steps 50 --lr_min_ratio 0.1 `
  --seed 42 `
  --rescale_attention 0 `
  --ckpt_every_n_steps 500 --val_every_n_steps 200 --log_every_n_steps 10 `
  --resume auto
