# OpenFold representation generation

Run OpenFold recycle-3 (no templates) on **seqres + MSA**. Writes:

- `pretrained_single` `[L, 384]`
- `pretrained_pair` `[L, L, 128]`

This is **not** folding your PDB/XTC trajectories. Coordinates stay the train targets. One embedding per unique sequence; every conformation of that seqres reuses it.

OpenFold code is vendored at `confrover._ext.openfold`. Weights: `confrover_cache/openfold_params/finetuning_no_templ_ptm_1.pt`.

```bash
confrover openfold_repr \
    --input_csv confrover_cache/dpf_seqres_index.csv \
    --msa_root confrover_cache/msa \
    --folding_repr confrover_cache/folding_repr \
    --openfold_params confrover_cache/openfold_params \
    --num_workers 1
```

`confrover train` requires `folding_repr/seqres_to_index.csv` to cover every train/val family. Pair features grow as `L²`; chains longer than ~384 residues often OOM on 8 GB GPUs.

See `confrover openfold_repr --help`.
