#!/usr/bin/env bash
# Container entrypoint for the PDB-cluster cloud run.
#
#   smoke   download the payload, verify it, run 6 real steps, stop      (default)
#   train   the same, then train; the same command resumes after a stop
#   verify  download + verify only (no GPU time)
#   shell   an interactive shell in the container
#   <cmd>   anything else is exec'd as-is (e.g. `confrover train --help`)
#
# Needs HF_TOKEN in the environment (read on $HF_REPO; write for the checkpoint
# sync). Everything else has a default (see docker/Dockerfile and
# scripts/vast_bootstrap_pdbcluster.sh): HF_REPO, WORKERS, ONE_PASS, MAX_EPOCHS,
# TF32, RUN_NAME.
set -euo pipefail

REPO="${REPO:-/opt/ConfRover}"
MODE="${1:-smoke}"
BOOT="$REPO/scripts/vast_bootstrap_pdbcluster.sh"

need_token() {
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is not set. Pass it with: docker run -e HF_TOKEN=hf_... (read on $HF_REPO; write for checkpoint sync)" >&2
    exit 2
  fi
}

case "$MODE" in
  smoke)
    need_token
    exec bash "$BOOT"
    ;;
  train)
    need_token
    exec bash "$BOOT" --train
    ;;
  verify)
    need_token
    shift || true
    export HF_REPO="${HF_REPO:?}"
    DL="${DL:-/workspace/hf_download}"
    DATA="${DATA:-/workspace/confrover_data}"
    hf download "$HF_REPO" --repo-type dataset --local-dir "$DL" --exclude "checkpoints/*"
    [[ -e "$DATA" ]] || ln -s "$DL" "$DATA"
    exec python "$REPO/scripts/verify_remote_payload.py" --root "$DATA" "$@"
    ;;
  shell|bash)
    exec bash
    ;;
  *)
    exec "$@"
    ;;
esac
