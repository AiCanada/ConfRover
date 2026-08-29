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

"""Shared torch.DataLoader kwargs."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class LoaderConfig:
    """Additional configuration for torch.DataLoader"""

    batch_size: int | None = None
    num_workers: int | None = None
    pin_memory: bool | None = None
    shuffle: bool | None = None
    #: Keep the worker processes alive between passes. Without it every
    #: epoch (and every validation) respawns the pool, and on Windows (spawn,
    #: not fork) each worker re-imports torch and confrover from scratch --
    #: tens of seconds of dead time, and a window in which a worker picks up
    #: source that was edited after the run started. Train workers stay
    #: correct across epochs because DpfTrainDataset.set_epoch writes a
    #: shared-memory epoch that __getitem__ re-reads.
    persistent_workers: bool | None = None

    def to_dict(self, drop_none: bool = True):
        obj_dict = asdict(self)
        # DataLoader rejects persistent_workers=True when num_workers == 0, so
        # the invariant is enforced here rather than at each call site.
        if not obj_dict.get("num_workers"):
            obj_dict["persistent_workers"] = None
        if drop_none:
            return {k: v for k, v in obj_dict.items() if v is not None}
        return obj_dict
