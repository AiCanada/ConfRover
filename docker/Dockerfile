# ConfRover PDB-cluster fine-tune for a rented NVIDIA RTX PRO 6000 (Blackwell, 96 GB):
#
#   docker build -t aicanada/confrover-pdbcluster:latest -f docker/Dockerfile \
#       --build-arg COMMIT=$(git rev-parse HEAD) .
#   docker run --gpus all --ipc=host -e HF_TOKEN=... -v confrover_ws:/workspace \
#       aicanada/confrover-pdbcluster:latest smoke      # download, verify, 6-step smoke
#   docker run --gpus all --ipc=host -e HF_TOKEN=... -v confrover_ws:/workspace \
#       aicanada/confrover-pdbcluster:latest train      # ... then train (resumes if re-run)
#
# The image carries the code only (~15 MB on top of the base). The 33 GB payload
# (structures, OpenFold representations, base weights, split) is pulled from the
# Hugging Face dataset repo at first start into the /workspace volume, where the
# run directory and checkpoints also live -- so a container can be replaced without
# downloading again, and the checkpoint sync watcher frees billed disk as it goes.
#
# Base: torch 2.13 (the version the fine-tune was validated on) built against
# CUDA 13.0. The CUDA 12.6 build of the same torch has no Blackwell kernels -- its
# arch list stops at sm_90, the RTX PRO 6000 is sm_120 -- and fails with "no kernel
# image is available". CUDA 13.0 needs a host driver >= 580; for an older driver
# use the last 12.8 build:  --build-arg BASE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime
ARG BASE=pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime
FROM ${BASE}

ARG COMMIT=unknown

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    # The pytorch image installs torch into Ubuntu's system Python, which PEP 668
    # marks externally managed; inside a container that is the environment we want.
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONUNBUFFERED=1 \
    # Code lives outside /workspace so mounting the data volume there cannot
    # hide it. scripts/vast_bootstrap_pdbcluster.sh reads these.
    REPO=/opt/ConfRover \
    DL=/workspace/hf_download \
    DATA=/workspace/confrover_data \
    HF_REPO=AICanada/ConfRover-PDBcluster     # A 6000 Pro host has the cores; 4 was the laptop's spawn-bound value.
    WORKERS=8 \
    # One validation pass walks several sequence lengths and the fused token axis
    # is 9 x (L + L^2); without this the allocator accumulates every shape.
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HUB_ENABLE_HF_TRANSFER=0

RUN apt-get update \
 && apt-get install -y --no-install-recommends git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ConfRover

# Dependency layer first, against a stub package, so code edits do not
# reinstall torch's friends.
COPY pyproject.toml README.md LICENSE ./
RUN mkdir -p src/confrover && touch src/confrover/__init__.py \
 && pip install -e . "huggingface_hub>=0.23" \
 && rm -rf src

COPY src ./src
COPY scripts ./scripts
COPY docker/entrypoint.sh /usr/local/bin/confrover-cloud
RUN pip install -e . --no-deps \
 && sed -i 's/\r$//' /usr/local/bin/confrover-cloud scripts/*.sh \
 && chmod +x /usr/local/bin/confrover-cloud scripts/*.sh \
 && echo "${COMMIT}" > COMMIT \
 && python -c "import confrover, torch; print('confrover ok; torch', torch.__version__, 'cuda', torch.version.cuda, 'archs', torch.cuda.get_arch_list())"

VOLUME ["/workspace"]
WORKDIR /workspace
ENTRYPOINT ["confrover-cloud"]
CMD ["smoke"]
