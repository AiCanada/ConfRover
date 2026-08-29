# Copyright 2025 Bytedance Ltd. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""`torch.cuda.amp.autocast` is deprecated and will be removed.

Ten call sites used it to *disable* autocast around numerically sensitive
blocks (softmax, outer product mean, the triangular update, the structure
module). Every one printed a FutureWarning per forward pass, which buried the
useful output of a generation run; and when torch removes the shim they stop
being a warning and start being an AttributeError in the middle of attention.

`torch.amp.autocast("cuda", enabled=False)` is exactly equivalent for these
uses -- same autocast state, same nesting behaviour -- which the runtime test
below pins rather than assumes.
"""

from __future__ import annotations

import warnings

import torch

from confrover import PACKAGE_ROOT

_DEPRECATED = "torch.cuda.amp"


def _package_sources():
    for path in PACKAGE_ROOT.rglob("*.py"):
        if "_deprecated" in path.parts:
            continue
        yield path


def test_no_source_file_uses_the_deprecated_amp_namespace():
    offenders = [
        f"{path.relative_to(PACKAGE_ROOT)}:{i}"
        for path in _package_sources()
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if _DEPRECATED in line
    ]
    assert offenders == [], (
        "use torch.amp.autocast('cuda', ...) instead of torch.cuda.amp.autocast: "
        + ", ".join(offenders)
    )


def test_the_replacement_disables_autocast_and_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with torch.amp.autocast("cuda", enabled=False):
            assert torch.is_autocast_enabled("cuda") is False


def test_the_replacement_still_disables_autocast_when_nested_inside_one():
    """The actual use: an fp32 island inside a mixed-precision forward pass."""
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        assert torch.is_autocast_enabled("cuda") is True
        with torch.amp.autocast("cuda", enabled=False):
            assert torch.is_autocast_enabled("cuda") is False
        assert torch.is_autocast_enabled("cuda") is True
