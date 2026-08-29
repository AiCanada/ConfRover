# Copyright 2025 Bytedance Ltd. and/or its affiliates.
from __future__ import annotations

from confrover._ext.openfold.np import residue_constants as rc


def test_residue_constants_are_vendored_not_pypi():
    assert rc.__name__.startswith("confrover._ext.openfold")
    assert rc.atom_order["CA"] == 1
    bonds, _virtual, _angles = rc.load_stereo_chemical_props()
    assert "ALA" in bonds


def test_data_transforms_import_without_pypi_openfold():
    from confrover.data.coords import atom37_to_openfold_feat
    from confrover._ext.openfold.data import data_transforms
    from confrover._ext.openfold.utils import rigid_utils as ru

    assert data_transforms.__name__.startswith("confrover._ext.openfold")
    assert ru.__name__.startswith("confrover._ext.openfold")
    assert callable(atom37_to_openfold_feat)
    assert callable(data_transforms.atom37_to_frames)
    assert callable(data_transforms.pseudo_beta_fn)
