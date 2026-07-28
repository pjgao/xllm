#!/usr/bin/env python3

# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import tilelang
import tilelang.language as T

from .utils import DEFAULT_ASCEND_PASS_CONFIGS
from ....common.spec import DispatchField, TilelangKernel, register_kernel

SUPPORTED_SPEC_WIDTHS = (4, 5, 6)
MAX_KV_SEQ_LEN_INDEX = 22
KV_SPLIT_LENGTH_INDEX = 23
KV_SPLIT_CORE_COUNT_INDEX = 24
TILING_BATCH_OFFSET = 44
TILING_BATCH_STRIDE = 17
TILING_BATCH_KV_LEN_OFFSET = 1


def build_spec_verify_attention_tiling_update_kernel(
    spec_width: int, block_size: int
):
    if spec_width not in SUPPORTED_SPEC_WIDTHS:
        raise ValueError(f"unsupported MTP tiling width: {spec_width}")

    @T.prim_func
    def spec_verify_attention_tiling_update(
        src_kv_seq_lens: T.Tensor((spec_width,), "int32"),
        tiling_data: T.Tensor((262144,), "int32"),
    ):
        with T.Kernel(1, is_npu=True):
            # Keep the reduction explicit. The Ascend scalar lowering drops
            # assignments nested under a T.serial conditional here.
            max_kv_4 = T.max(
                T.max(src_kv_seq_lens[0], src_kv_seq_lens[1]),
                T.max(src_kv_seq_lens[2], src_kv_seq_lens[3]),
            )
            # Write each specialized reduction inside its branch. TileLang's
            # parser scopes branch-local scalar assignments, so assigning
            # max_kv here and consuming it after the branch is not valid.
            if spec_width == 4:
                max_kv = max_kv_4
                kv_split_core_num = tiling_data[KV_SPLIT_CORE_COUNT_INDEX]
                tiling_data[MAX_KV_SEQ_LEN_INDEX] = max_kv
                tiling_data[KV_SPLIT_LENGTH_INDEX] = (
                    T.ceildiv(
                        T.ceildiv(max_kv, block_size), kv_split_core_num
                    )
                    * block_size
                )
            elif spec_width == 5:
                max_kv = T.max(max_kv_4, src_kv_seq_lens[4])
                kv_split_core_num = tiling_data[KV_SPLIT_CORE_COUNT_INDEX]
                tiling_data[MAX_KV_SEQ_LEN_INDEX] = max_kv
                tiling_data[KV_SPLIT_LENGTH_INDEX] = (
                    T.ceildiv(
                        T.ceildiv(max_kv, block_size), kv_split_core_num
                    )
                    * block_size
                )
            else:
                max_kv = T.max(
                    T.max(max_kv_4, src_kv_seq_lens[4]),
                    src_kv_seq_lens[5],
                )
                kv_split_core_num = tiling_data[KV_SPLIT_CORE_COUNT_INDEX]
                tiling_data[MAX_KV_SEQ_LEN_INDEX] = max_kv
                tiling_data[KV_SPLIT_LENGTH_INDEX] = (
                    T.ceildiv(
                        T.ceildiv(max_kv, block_size), kv_split_core_num
                    )
                    * block_size
                )
            for i in T.serial(spec_width):
                kv_len = src_kv_seq_lens[i]
                tiling_data[
                    TILING_BATCH_OFFSET
                    + i * TILING_BATCH_STRIDE
                    + TILING_BATCH_KV_LEN_OFFSET
                ] = kv_len

    return spec_verify_attention_tiling_update


@register_kernel
class SpecVerifyAttentionTilingUpdateKernel(TilelangKernel):
    DISPATCH_SCHEMA = [
        DispatchField("spec_width", "int32"),
        DispatchField("block_size", "int32"),
    ]
    SPECIALIZATIONS = [
        {
            "variant_key": f"w{spec_width}_bs128",
            "spec_width": spec_width,
            "block_size": 128,
        }
        for spec_width in SUPPORTED_SPEC_WIDTHS
    ]

    @staticmethod
    def generate_source(spec_width: int, block_size: int) -> str:
        tilelang.disable_cache()
        kernel = build_spec_verify_attention_tiling_update_kernel(
            spec_width, block_size
        )
        with tilelang.tvm.transform.PassContext(
            opt_level=3, config=DEFAULT_ASCEND_PASS_CONFIGS
        ):
            lowered = tilelang.engine.lower(kernel)
        return lowered.kernel_source
