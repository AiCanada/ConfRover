# Stage the cloud payload for runsPDB\PDBcluster_from_base (see scripts/vast_bootstrap_pdbcluster.sh).
#
# What ships (hardlinked on the same volume, so no extra disk):
#   pdbc/            every structure the unique catalog names (~1.7 GB, test families included:
#                    the split fingerprint is over all 1,678 families)
#   folding_repr/    OpenFold representations for the 1,610 train+val sequences (~31 GB)
#   confrover_ckpts/ original_confrover_base_20m_v1_0.pt (the published base, 79 MB)
#   run/splits/      the hand-edited 1,442/168/68 split
#   catalog.json     paths rebased onto /workspace/confrover_data
#   MANIFEST.sha256  verified on the instance before any GPU time is spent
#
# Then (resumable, many small files):
#   hf upload-large-folder AICanada/ConfRover-PDBcluster A:\payloads\PDBcluster_from_base --repo-type dataset --num-workers 8
#
# --remote_root must be a POSIX path; run this from PowerShell (Git Bash rewrites
# "/workspace/..." into a C:\Program Files\Git\... path before python sees it).

# Not "Stop": Windows PowerShell 5.1 turns any native stderr line (python logging
# writes its notices there) into a terminating NativeCommandError. The exit code
# below is the real verdict.
$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$out = if ($env:PAYLOAD_OUT) { $env:PAYLOAD_OUT } else { "A:\payloads\PDBcluster_from_base" }
Set-Location $repo

py -3.13 scripts\stage_remote_payload.py `
  --catalog    "$repo\confrover_cache\pdbc95_over10_catalog_unique.json" `
  --pdbc_root  "A:\ATLAS DATA\PDB_Cluster_Shards\pdb_clusters_95_over10_cap100" `
  --split_file "$repo\runsPDB\PDBcluster_from_base\splits\0.json" `
  --cache_dir  "$repo\confrover_cache" `
  --run_dir    "$repo\runsPDB\PDBcluster_from_base" `
  --checkpoint none `
  --weights    "$repo\runsPDB\original_confrover_base_20m_v1_0.pt" `
  --out        "$out" `
  --remote_root /workspace/confrover_data `
  --bundle pdbc

if ($LASTEXITCODE -ne 0) { Write-Error "staging failed with exit code $LASTEXITCODE"; exit $LASTEXITCODE }
Write-Host "staged OK -> $out"
