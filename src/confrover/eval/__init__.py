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

"""Ensemble-quality evaluation: the metrics the ATLAS literature actually uses.

Pure array functions only -- no filesystem, no torch, no model -- so the same
code scores the model arms, the MD-vs-MD floor, and the published fixtures.
"""

from __future__ import annotations

from .ensemble_metrics import (
    ATLAS_SUBSAMPLE_SEED,
    JS_MAX,
    JS_N_BINS,
    N_CONFORMATIONS,
    N_RMWD_REFERENCE_FRAMES,
    AtlasDraws,
    Pca,
    Tica,
    atlas_subsample,
    ca_bond_violation_fraction,
    clash_fraction,
    collapse_guard,
    collapse_verdict,
    contact_masks,
    contact_probability,
    debias_self_pairwise_rmsd,
    effective_sample_dimension,
    empirical_w2,
    ensemble_metrics,
    exposed_residue_jaccard,
    exposure_mask,
    exposure_mi_rho,
    exposure_mutual_information,
    gaussian_w2_per_atom,
    jaccard,
    js_columns,
    mean_pairwise_rmsd,
    pairwise_distance_features,
    pc1_cosine,
    pca_fit,
    pca_project,
    radius_of_gyration,
    reference_control,
    rmsf,
    rmwd,
    stride_segments,
    superpose,
    tica_fit,
    tica_project,
)

__all__ = [
    "ATLAS_SUBSAMPLE_SEED",
    "AtlasDraws",
    "JS_MAX",
    "JS_N_BINS",
    "N_CONFORMATIONS",
    "N_RMWD_REFERENCE_FRAMES",
    "Pca",
    "Tica",
    "atlas_subsample",
    "ca_bond_violation_fraction",
    "clash_fraction",
    "collapse_guard",
    "collapse_verdict",
    "contact_masks",
    "contact_probability",
    "debias_self_pairwise_rmsd",
    "effective_sample_dimension",
    "empirical_w2",
    "ensemble_metrics",
    "exposed_residue_jaccard",
    "exposure_mask",
    "exposure_mi_rho",
    "exposure_mutual_information",
    "gaussian_w2_per_atom",
    "jaccard",
    "js_columns",
    "mean_pairwise_rmsd",
    "pairwise_distance_features",
    "pc1_cosine",
    "pca_fit",
    "pca_project",
    "radius_of_gyration",
    "reference_control",
    "rmsf",
    "rmwd",
    "stride_segments",
    "superpose",
    "tica_fit",
    "tica_project",
]
