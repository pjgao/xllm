/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <cstdint>

namespace xllm {

// Describes target-model requirements at the speculative verification seam.
// A model that enables in_graph_input_update guarantees that its validation
// input follows the expanded-attention and single linear-state layout consumed
// by the NPU graph updater. Unsupported models keep the conservative defaults.
struct SpeculativeVerifyCapabilities {
  bool requires_causal_chunked_prefill = false;
  bool supports_in_graph_input_update = false;
  int32_t min_graph_update_speculative_tokens = 3;
  int32_t max_graph_update_speculative_tokens = 5;
  int32_t graph_update_block_size = 128;
  int64_t max_graph_update_block_table_width = (1 << 15) - 1;
};

struct SpeculativeVerifyGraphLayout {
  int32_t num_speculative_tokens = 0;
  int32_t num_sequences = 0;
  int32_t block_size = 0;
  int64_t block_table_width = 0;
};

inline bool supports_speculative_verify_graph_layout(
    const SpeculativeVerifyCapabilities& capabilities,
    const SpeculativeVerifyGraphLayout& layout) {
  return capabilities.supports_in_graph_input_update &&
         layout.num_speculative_tokens >=
             capabilities.min_graph_update_speculative_tokens &&
         layout.num_speculative_tokens <=
             capabilities.max_graph_update_speculative_tokens &&
         layout.num_sequences == 1 &&
         layout.block_size == capabilities.graph_update_block_size &&
         layout.block_table_width > 0 &&
         layout.block_table_width <=
             capabilities.max_graph_update_block_table_width;
}

}  // namespace xllm
