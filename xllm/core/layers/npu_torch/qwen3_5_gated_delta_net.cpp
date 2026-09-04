/* Copyright 2025-2026 The xLLM Authors.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    https://github.com/xLLM-AI/xllm/blob/main/LICENSE
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "qwen3_5_gated_delta_net.h"

#include <glog/logging.h>

namespace xllm {
namespace layer {

Qwen3_5GatedDeltaNetImpl::Qwen3_5GatedDeltaNetImpl(
    const ModelArgs& args,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options)
    : Qwen3NextGatedDeltaNetImpl(args,
                                 quant_args,
                                 parallel_args,
                                 options,
                                 /*init_projections=*/false),
      use_fused_projections_(options.dtype().toScalarType() ==
                                 torch::kBFloat16 &&
                             quant_args.quant_method().empty() &&
                             quant_args.quant_descs().empty()) {
  if (use_fused_projections_) {
    fused_qkvzba_proj_ = register_module(
        "fused_in_proj_qkvzba",
        ColumnParallelLinear(args.hidden_size(),
                             k_size_ * 2 + v_size_ * 2 + num_v_heads_ * 2,
                             /*bias=*/false,
                             /*gather_output=*/false,
                             quant_args,
                             parallel_args.tp_group_,
                             options));
    LOG(INFO) << "Qwen3.5 BF16 projection fusion enabled: QKV/Z/B/A";
    return;
  }

  in_proj_qkv_ = register_module("in_proj_qkv",
                                 ColumnParallelLinear(args.hidden_size(),
                                                      k_size_ * 2 + v_size_,
                                                      /*bias=*/false,
                                                      /*gather_output=*/false,
                                                      quant_args,
                                                      parallel_args.tp_group_,
                                                      options));
  in_proj_z_ = register_module("in_proj_z",
                               ColumnParallelLinear(args.hidden_size(),
                                                    v_size_,
                                                    /*bias=*/false,
                                                    /*gather_output=*/false,
                                                    quant_args,
                                                    parallel_args.tp_group_,
                                                    options));
  in_proj_b_ = register_module("in_proj_b",
                               ColumnParallelLinear(args.hidden_size(),
                                                    num_v_heads_,
                                                    /*bias=*/false,
                                                    /*gather_output=*/false,
                                                    quant_args,
                                                    parallel_args.tp_group_,
                                                    options));
  in_proj_a_ = register_module("in_proj_a",
                               ColumnParallelLinear(args.hidden_size(),
                                                    num_v_heads_,
                                                    /*bias=*/false,
                                                    /*gather_output=*/false,
                                                    quant_args,
                                                    parallel_args.tp_group_,
                                                    options));
}

torch::Tensor Qwen3_5GatedDeltaNetImpl::merge_qkvz_from_split_activations(
    const torch::Tensor& qkv,
    const torch::Tensor& z) const {
  CHECK_EQ(qkv.dim(), 3) << "Expected qkv activation to be 3D, got "
                         << qkv.sizes();
  CHECK_EQ(z.dim(), 3) << "Expected z activation to be 3D, got " << z.sizes();
  CHECK_EQ(qkv.size(0), z.size(0)) << "qkv/z batch size mismatch.";
  CHECK_EQ(qkv.size(1), z.size(1)) << "qkv/z sequence size mismatch.";
  CHECK_EQ(qkv.size(2), (2 * k_size_ + v_size_) / tp_size_)
      << "Unexpected qkv hidden size for Qwen3.5.";
  CHECK_EQ(z.size(2), v_size_ / tp_size_)
      << "Unexpected z hidden size for Qwen3.5.";
  CHECK_GT(num_k_heads_, 0) << "linear_num_key_heads must be positive.";
  CHECK_EQ(num_v_heads_ % num_k_heads_, 0)
      << "linear_num_value_heads must be divisible by linear_num_key_heads.";

  const int64_t bs = qkv.size(0);
  const int64_t seqlen = qkv.size(1);
  const int64_t local_k_heads = num_k_heads_ / tp_size_;
  const int64_t local_v_heads = num_v_heads_ / tp_size_;
  const int64_t num_v_heads_per_k = num_v_heads_ / num_k_heads_;

  auto qkv_split = torch::split(
      qkv, {k_size_ / tp_size_, k_size_ / tp_size_, v_size_ / tp_size_}, 2);
  auto q = qkv_split[0].view({bs, seqlen, local_k_heads, head_k_dim_});
  auto k = qkv_split[1].view({bs, seqlen, local_k_heads, head_k_dim_});
  auto v = qkv_split[2].view({bs, seqlen, local_v_heads, head_v_dim_});
  auto z_view = z.view({bs, seqlen, local_v_heads, head_v_dim_});

  v = v.view({bs, seqlen, local_k_heads, num_v_heads_per_k * head_v_dim_});
  z_view =
      z_view.view({bs, seqlen, local_k_heads, num_v_heads_per_k * head_v_dim_});

  return torch::cat({q, k, v, z_view}, -1).view({bs, seqlen, -1}).contiguous();
}

torch::Tensor Qwen3_5GatedDeltaNetImpl::merge_ba_from_split_activations(
    const torch::Tensor& b,
    const torch::Tensor& a) const {
  CHECK_EQ(b.dim(), 3) << "Expected b activation to be 3D, got " << b.sizes();
  CHECK_EQ(a.dim(), 3) << "Expected a activation to be 3D, got " << a.sizes();
  CHECK_EQ(b.size(0), a.size(0)) << "b/a batch size mismatch.";
  CHECK_EQ(b.size(1), a.size(1)) << "b/a sequence size mismatch.";
  CHECK_EQ(b.size(2), num_v_heads_ / tp_size_)
      << "Unexpected b hidden size for Qwen3.5.";
  CHECK_EQ(a.size(2), num_v_heads_ / tp_size_)
      << "Unexpected a hidden size for Qwen3.5.";
  CHECK_GT(num_k_heads_, 0) << "linear_num_key_heads must be positive.";
  CHECK_EQ(num_v_heads_ % num_k_heads_, 0)
      << "linear_num_value_heads must be divisible by linear_num_key_heads.";

  const int64_t bs = b.size(0);
  const int64_t seqlen = b.size(1);
  const int64_t local_k_heads = num_k_heads_ / tp_size_;
  const int64_t num_v_heads_per_k = num_v_heads_ / num_k_heads_;

  auto b_view = b.view({bs, seqlen, local_k_heads, num_v_heads_per_k});
  auto a_view = a.view({bs, seqlen, local_k_heads, num_v_heads_per_k});
  return torch::cat({b_view, a_view}, -1).view({bs, seqlen, -1}).contiguous();
}

std::pair<torch::Tensor, torch::Tensor>
Qwen3_5GatedDeltaNetImpl::project_decode_inputs(
    const torch::Tensor& hidden_states) {
  if (use_fused_projections_) {
    torch::Tensor qkvzba = fused_qkvzba_proj_->forward(hidden_states);
    const int64_t qkvz_width = (k_size_ * 2 + v_size_ * 2) / tp_size_;
    const int64_t ba_width = (num_v_heads_ * 2) / tp_size_;
    torch::Tensor qkvz = qkvzba.narrow(-1, 0, qkvz_width);
    torch::Tensor ba = qkvzba.narrow(-1, qkvz_width, ba_width);
    return {qkvz.reshape({qkvz.size(0), -1, qkvz.size(-1)}),
            ba.reshape({ba.size(0), -1, ba.size(-1)})};
  }

  const auto reshape_projection = [](const torch::Tensor& projection) {
    return projection.view({projection.size(0), -1, projection.size(-1)});
  };
  auto qkv = reshape_projection(in_proj_qkv_->forward(hidden_states));
  auto z_proj = reshape_projection(in_proj_z_->forward(hidden_states));
  auto b_proj = reshape_projection(in_proj_b_->forward(hidden_states));
  auto a_proj = reshape_projection(in_proj_a_->forward(hidden_states));
  return {merge_qkvz_from_split_activations(qkv, z_proj),
          merge_ba_from_split_activations(b_proj, a_proj)};
}

std::pair<torch::Tensor, torch::Tensor>
Qwen3_5GatedDeltaNetImpl::project_flat_inputs(
    const torch::Tensor& hidden_states) {
  if (use_fused_projections_) {
    torch::Tensor qkvzba = fused_qkvzba_proj_->forward(hidden_states);
    const int64_t qkvz_width = (k_size_ * 2 + v_size_ * 2) / tp_size_;
    const int64_t ba_width = (num_v_heads_ * 2) / tp_size_;
    return {qkvzba.narrow(-1, 0, qkvz_width),
            qkvzba.narrow(-1, qkvz_width, ba_width)};
  }

  auto qkv = in_proj_qkv_->forward(hidden_states).unsqueeze(0);
  auto z_proj = in_proj_z_->forward(hidden_states).unsqueeze(0);
  auto b_proj = in_proj_b_->forward(hidden_states).unsqueeze(0);
  auto a_proj = in_proj_a_->forward(hidden_states).unsqueeze(0);
  auto qkvz = merge_qkvz_from_split_activations(qkv, z_proj);
  auto ba = merge_ba_from_split_activations(b_proj, a_proj);
  return {qkvz.view({hidden_states.size(0), qkvz.size(-1)}).contiguous(),
          ba.view({hidden_states.size(0), ba.size(-1)}).contiguous()};
}

std::optional<
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>
Qwen3_5GatedDeltaNetImpl::project_split_inputs(
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata) {
  if (use_fused_projections_) {
    // Keep the two packed projection results intact and let the existing
    // fused_qkvzba_split_reshape_cat kernel split/reshape them in one launch.
    // Returning views here would require four device copies because slices of
    // the packed last dimension are not contiguous.
    return std::nullopt;
  }

  auto qkv = reshape_projected_tokens_with_pad(
      attn_metadata, in_proj_qkv_->forward(hidden_states));
  auto z_proj = reshape_projected_tokens_with_pad(
      attn_metadata, in_proj_z_->forward(hidden_states));
  auto b_proj = reshape_projected_tokens_with_pad(
      attn_metadata, in_proj_b_->forward(hidden_states));
  auto a_proj = reshape_projected_tokens_with_pad(
      attn_metadata, in_proj_a_->forward(hidden_states));

  const int64_t batch_size = qkv.size(0);
  const int64_t seq_len = qkv.size(1);
  auto z =
      z_proj.view({batch_size, seq_len, num_v_heads_ / tp_size_, head_v_dim_});
  auto b = b_proj.view({batch_size, seq_len, num_v_heads_ / tp_size_});
  auto a = a_proj.view({batch_size, seq_len, num_v_heads_ / tp_size_});
  return std::make_tuple(qkv, z, b, a);
}

void Qwen3_5GatedDeltaNetImpl::load_projection_state_dict(
    const StateDict& state_dict) {
  if (use_fused_projections_) {
    if (fused_qkvzba_proj_->is_weight_loaded()) {
      return;
    }
    auto remember_if_defined = [&](const char* name, torch::Tensor& pending) {
      torch::Tensor weight = state_dict.get_tensor(name);
      if (weight.defined()) {
        pending = weight;
      }
    };
    remember_if_defined("in_proj_qkv.weight", pending_qkv_weight_);
    remember_if_defined("in_proj_z.weight", pending_z_weight_);
    remember_if_defined("in_proj_b.weight", pending_b_weight_);
    remember_if_defined("in_proj_a.weight", pending_a_weight_);
    if (pending_qkv_weight_.defined() && pending_z_weight_.defined() &&
        pending_b_weight_.defined() && pending_a_weight_.defined()) {
      StateDict source_state_dict({{"in_proj_qkv.weight", pending_qkv_weight_},
                                   {"in_proj_z.weight", pending_z_weight_},
                                   {"in_proj_b.weight", pending_b_weight_},
                                   {"in_proj_a.weight", pending_a_weight_}});
      fused_qkvzba_proj_->load_state_dict(
          build_fused_qkvzba_state_dict(source_state_dict));
      pending_qkv_weight_ = torch::Tensor();
      pending_z_weight_ = torch::Tensor();
      pending_b_weight_ = torch::Tensor();
      pending_a_weight_ = torch::Tensor();
    }
    return;
  }

  auto in_proj_qkv_state_dict = state_dict.get_dict_with_prefix("in_proj_qkv.");
  if (in_proj_qkv_state_dict.size() > 0 && !in_proj_qkv_->is_weight_loaded()) {
    in_proj_qkv_->load_state_dict(
        in_proj_qkv_state_dict,
        /*shard_tensor_count=*/3,
        /*shard_sizes=*/
        {k_size_ / tp_size_, k_size_ / tp_size_, v_size_ / tp_size_});
  }

  auto in_proj_z_state_dict = state_dict.get_dict_with_prefix("in_proj_z.");
  if (in_proj_z_state_dict.size() > 0 && !in_proj_z_->is_weight_loaded()) {
    in_proj_z_->load_state_dict(in_proj_z_state_dict);
  }

  auto in_proj_b_state_dict = state_dict.get_dict_with_prefix("in_proj_b.");
  if (in_proj_b_state_dict.size() > 0 && !in_proj_b_->is_weight_loaded()) {
    in_proj_b_->load_state_dict(in_proj_b_state_dict);
  }

  auto in_proj_a_state_dict = state_dict.get_dict_with_prefix("in_proj_a.");
  if (in_proj_a_state_dict.size() > 0 && !in_proj_a_->is_weight_loaded()) {
    in_proj_a_->load_state_dict(in_proj_a_state_dict);
  }
}

void Qwen3_5GatedDeltaNetImpl::verify_projection_weights(
    const std::string& prefix) const {
  if (use_fused_projections_) {
    CHECK(fused_qkvzba_proj_ && fused_qkvzba_proj_->is_weight_loaded())
        << "Missing fused QKV/Z/B/A projection after loading split weights: "
        << prefix;
    return;
  }

  CHECK(in_proj_qkv_ && in_proj_qkv_->is_weight_loaded())
      << "Missing required weight after all shards loaded: " << prefix
      << "in_proj_qkv.weight";
  CHECK(in_proj_z_ && in_proj_z_->is_weight_loaded())
      << "Missing required weight after all shards loaded: " << prefix
      << "in_proj_z.weight";
  CHECK(in_proj_b_ && in_proj_b_->is_weight_loaded())
      << "Missing required weight after all shards loaded: " << prefix
      << "in_proj_b.weight";
  CHECK(in_proj_a_ && in_proj_a_->is_weight_loaded())
      << "Missing required weight after all shards loaded: " << prefix
      << "in_proj_a.weight";
}

StateDict Qwen3_5GatedDeltaNetImpl::build_fused_qkvzba_state_dict(
    const StateDict& state_dict) const {
  torch::Tensor qkv_weight = state_dict.get_tensor("in_proj_qkv.weight");
  torch::Tensor z_weight = state_dict.get_tensor("in_proj_z.weight");
  torch::Tensor b_weight = state_dict.get_tensor("in_proj_b.weight");
  torch::Tensor a_weight = state_dict.get_tensor("in_proj_a.weight");
  if (!qkv_weight.defined() || !z_weight.defined() || !b_weight.defined() ||
      !a_weight.defined()) {
    return StateDict({});
  }
  CHECK_EQ(qkv_weight.size(1), z_weight.size(1))
      << "Qwen3.5 QKV/Z projection input width mismatch.";
  std::vector<torch::Tensor> qkv_parts =
      torch::split(qkv_weight, {k_size_, k_size_, v_size_}, /*dim=*/0);
  const int64_t num_v_heads_per_k = num_v_heads_ / num_k_heads_;
  torch::Tensor qkvz_weight =
      torch::cat(
          {qkv_parts[0].view({num_k_heads_, head_k_dim_, -1}),
           qkv_parts[1].view({num_k_heads_, head_k_dim_, -1}),
           qkv_parts[2].view(
               {num_k_heads_, num_v_heads_per_k * head_v_dim_, -1}),
           z_weight.view({num_k_heads_, num_v_heads_per_k * head_v_dim_, -1})},
          /*dim=*/1)
          .reshape({2 * k_size_ + 2 * v_size_, qkv_weight.size(1)})
          .contiguous();
  CHECK_EQ(b_weight.sizes(), a_weight.sizes())
      << "Qwen3.5 B/A projection weight shape mismatch.";
  CHECK_EQ(b_weight.size(0), num_v_heads_)
      << "Unexpected Qwen3.5 B projection width.";
  torch::Tensor ba_weight =
      torch::cat({b_weight.view({num_k_heads_, num_v_heads_per_k, -1}),
                  a_weight.view({num_k_heads_, num_v_heads_per_k, -1})},
                 /*dim=*/1)
          .reshape({2 * num_v_heads_, b_weight.size(1)})
          .contiguous();

  // ColumnParallelLinear shards one contiguous output range per TP rank. Pack
  // each rank's QKVZ and BA rows together so the local activation is laid out
  // as [local_qkvz, local_ba] and can be consumed as two zero-copy views.
  std::vector<torch::Tensor> rank_weights;
  rank_weights.reserve(static_cast<size_t>(tp_size_));
  const auto qkvz_shards = qkvz_weight.chunk(tp_size_, /*dim=*/0);
  const auto ba_shards = ba_weight.chunk(tp_size_, /*dim=*/0);
  CHECK_EQ(qkvz_shards.size(), static_cast<size_t>(tp_size_));
  CHECK_EQ(ba_shards.size(), static_cast<size_t>(tp_size_));
  for (int64_t tp_rank = 0; tp_rank < tp_size_; ++tp_rank) {
    rank_weights.emplace_back(
        torch::cat({qkvz_shards[static_cast<size_t>(tp_rank)],
                    ba_shards[static_cast<size_t>(tp_rank)]},
                   /*dim=*/0));
  }
  return StateDict(
      {{"weight", torch::cat(rank_weights, /*dim=*/0).contiguous()}});
}

}  // namespace layer
}  // namespace xllm
