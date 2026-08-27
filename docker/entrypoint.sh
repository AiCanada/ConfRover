#!/usr/bin/env bash
# Container entrypoint for the PDB-cluster cloud run (RTX PRO 6000 / Blackwell).
#
#   smoke   download the payload, verify it, run 6 real steps, stop      (default)
#   train   the same, then train; the same command resumes after a stop
#   verify  download + verify only (no GPU time)
#   gpu     GPU preflight only: driver, device, capability vs torch's kernels
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

# Fail in seconds, not after a 33 GB download: the card's compute capability must
# be one torch has kernels for. A cu126 build stops at sm_90; Blackwell is sm_120.
gpu_preflight() {
  python - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("no CUDA device visible: run with --gpus all (and the NVIDIA container toolkit installed)")
name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
archs = torch.cuda.get_arch_list()
free, total = torch.cuda.mem_get_info()
print(f"gpu: {name}  sm_{major}{minor}  {total/2**30:.0f} GiB ({free/2**30:.0f} free)")
print(f"torch {torch.__version__}  cuda {torch.version.cuda}  kernels for: {', '.join(archs)}")
def _ok(arch):
    # SASS built for sm_XY runs on any device of the same major with minor >= Y;
    # PTX (compute_XY) JIT-compiles on anything with capability >= XY.
    kind, _, cap = arch.partition("_")
    if not cap.isdigit():
        return False
    a_major, a_minor = int(cap[:-1]), int(cap[-1])
    if kind == "sm":
        return a_major == major and a_minor <= minor
    return (a_major, a_minor) <= (major, minor)
if not any(_ok(a) for a in archs):
    sys.exit(f"torch in this image has no kernels for sm_{major}{minor}; rebuild with a CUDA "
             f"{'13.0' if major >= 10 else '12.x'} base (see docker/Dockerfile ARG BASE)")
torch.ones(1, device="cuda").mul_(2)
torch.cuda.synchronize()
print("gpu preflight ok")
PY
}

case "$MODE" in
  smoke)
    need_token
    gpu_preflight
    exec bash "$BOOT"
    ;;
  train)
    need_token
    gpu_preflight
    exec bash "$BOOT" --train
    ;;
  gpu)
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
    gpu_preflight
    ;;
  verify)
    need_token
    shift || true
    export HF_REPO="${HF_REPO:?}"
    DL="${DL:-/workspace/hf_download}"
    DATA="${DATA:-/workspace/confrover_data}"
    hf download "$HF_REPO" --repo-type dataset --local-dir "$DL" --max-workers "${HF_WORKERS:-32}" --exclude "checkpoints/*"
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
