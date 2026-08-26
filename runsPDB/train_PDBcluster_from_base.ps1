# Fine-tune on the PDB-cluster corpus (pdbc95_over10) from the ORIGINAL published
# ConfRover-base-20M-v1.0 weights, with 9-frame windows.
#
#   init weights : runsPDB\original_confrover_base_20m_v1_0.pt
#                  (byte-identical to confrover_cache\confrover_ckpts\confrover_base_20m_v1_0.pt;
#                   provenance tag confrover_20m_base)
#   output       : runsPDB\PDBcluster_from_base
#   checkpoints  : PDBcluster-epoch{E}-step{S}.ckpt / PDBcluster-epoch{E}-end.ckpt /
#                  PDBcluster-bestfwd-step{S}.ckpt / PDBcluster-stopped-step{S}.ckpt
#
# --rescale_attention 8: the decoder capacity repair is meant for the *published*
# base weights (two IPA blocks at 13-89x residual, one dead FFN) and runs once
# on a fresh lineage. This IS a fresh lineage from those weights, so it is on.
# (The v888 continuation script sets 0 because v888 already had it applied.)
#
# Training set: the PDB-cluster corpus only. --catalog takes precedence over
# --dpf_root; the *_unique catalog holds every structure from the 1,684
# pdb_clusters_95_over10_cap100 directories merged by sequence into 1,678
# families. splits\0.json is a copy of the hand-edited split used by the v888
# continuation (train=1,442 / val=168 / test=68); the --*_frac values are its
# realised fractions and must match it or DpfSplit.load refuses the file.
#
# --window_frames 9: each example is 9 distinct structures of one cluster; the
# temporal module gets 9 tokens and every token predicts its own frame. With
# --one_pass_frames the corpus is consumed in ~1 epoch (about 7,900 steps);
# drop --one_pass_frames to re-draw windows in later epochs.
#
# The same command resumes the run (--resume auto). Drop a STOP or PAUSE file in
# the output directory to end it cleanly.
#
# --cache_dir / --folding_repr are passed explicitly because the default cache
# root is ./confrover_cache *relative to the working directory*.

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repo

py -3.13 -m confrover.cli train `
  --catalog  "$repo\confrover_cache\pdbc95_over10_catalog_unique.json" `
  --dpf_root "A:\ATLAS DATA\PDB_Cluster_Shards\pdb_clusters_95_over10_cap100" `
  --cache_dir    "$repo\confrover_cache" `
  --folding_repr "$repo\confrover_cache\folding_repr" `
  --model    "$repo\runsPDB\original_confrover_base_20m_v1_0.pt" `
  --output   "$repo\runsPDB\PDBcluster_from_base" `
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
  --rescale_attention 8 `
  --ckpt_every_n_steps 500 --val_every_n_steps 500 --log_every_n_steps 10 `
  --resume auto
