# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
# SNN-ized by MofNeuroSim Project
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

"""
SpikeQwen3 - 100% Pure SNN Gate Circuit Implementation
======================================================

This module provides a standalone SNN implementation of Qwen3 that can be used
independently or integrated with the existing MofNeuroSim framework.
"""

from .modeling_qwen3 import (
    SpikeQwen3Config,
    SpikeQwen3RMSNorm,
    SpikeQwen3MLP,
    SpikeQwen3Attention,
    SpikeQwen3DecoderLayer,
    SpikeQwen3Model,
    SpikeQwen3ForCausalLM,
)

__all__ = [
    "SpikeQwen3Config",
    "SpikeQwen3RMSNorm",
    "SpikeQwen3MLP",
    "SpikeQwen3Attention",
    "SpikeQwen3DecoderLayer",
    "SpikeQwen3Model",
    "SpikeQwen3ForCausalLM",
]
