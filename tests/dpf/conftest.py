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

from __future__ import annotations

from pathlib import Path

import pytest

from confrover.data.dpf.catalog import DpfCatalog
from tests.dpf.toys import make_family


@pytest.fixture
def toy_catalog(tmp_path: Path) -> DpfCatalog:
    families = [
        make_family(tmp_path, "DPF-001", "AGSL"),
        make_family(tmp_path, "DPF-002", "AGVE"),
        make_family(tmp_path, "DPF-003", "LVAG"),
    ]
    return DpfCatalog(families=families)
