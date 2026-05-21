"""
Standalone repro captured via capture_hook.
Label: torchbench_modded_nanogpt_infer
Pattern hash: a1dae8115911
Shape hash: 95a55307
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([6144], i64), T([], i32), S([-1, 128]), S([-1, 128]))"

class Repro(torch.nn.Module):
    def forward(self, cumsum: "i64[6144]", arg4_1: "i32[]", _shape_param_0, _shape_param_1):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:842 in create_blockmasks, code: block_idx = torch.arange(NUM_BLOCKS, dtype=torch.int32, device="cuda")
        iota_default: "i32[48]" = torch.ops.prims.iota.default(48, start = 0, step = 1, dtype = torch.int32, device = device(type='cuda'), requires_grad = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:843 in create_blockmasks, code: causal_blockmask_any = block_idx[:, None] >= block_idx
        unsqueeze_default: "i32[48, 1]" = torch.ops.aten.unsqueeze.default(iota_default, 1)
        ge_tensor: "b8[48, 48]" = torch.ops.aten.ge.Tensor(unsqueeze_default, iota_default);  unsqueeze_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:845 in create_blockmasks, code: docs_low = docs.view(-1, BLOCK_SIZE)[:, 0].contiguous()
        reshape_default: "i64[48, 128]" = torch.ops.aten.reshape.default(cumsum, _shape_param_0);  _shape_param_0 = None
        select_int: "i64[48]" = torch.ops.aten.select.int(reshape_default, 1, 0);  reshape_default = None
        clone_default: "i64[48]" = torch.ops.aten.clone.default(select_int, memory_format = torch.contiguous_format);  select_int = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:847 in create_blockmasks, code: document_blockmask_any = (docs_low[:, None] <= docs_high) & (
        unsqueeze_default_1: "i64[48, 1]" = torch.ops.aten.unsqueeze.default(clone_default, 1)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:846 in create_blockmasks, code: docs_high = docs.view(-1, BLOCK_SIZE)[:, -1].contiguous()
        reshape_default_1: "i64[48, 128]" = torch.ops.aten.reshape.default(cumsum, _shape_param_1);  cumsum = _shape_param_1 = None
        select_int_1: "i64[48]" = torch.ops.aten.select.int(reshape_default_1, 1, -1);  reshape_default_1 = None
        clone_default_1: "i64[48]" = torch.ops.aten.clone.default(select_int_1, memory_format = torch.contiguous_format);  select_int_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:847 in create_blockmasks, code: document_blockmask_any = (docs_low[:, None] <= docs_high) & (
        le_tensor: "b8[48, 48]" = torch.ops.aten.le.Tensor(unsqueeze_default_1, clone_default_1);  unsqueeze_default_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:848 in create_blockmasks, code: docs_high[:, None] >= docs_low
        unsqueeze_default_2: "i64[48, 1]" = torch.ops.aten.unsqueeze.default(clone_default_1, 1)
        ge_tensor_1: "b8[48, 48]" = torch.ops.aten.ge.Tensor(unsqueeze_default_2, clone_default);  unsqueeze_default_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:847 in create_blockmasks, code: document_blockmask_any = (docs_low[:, None] <= docs_high) & (
        bitwise_and_tensor: "b8[48, 48]" = torch.ops.aten.bitwise_and.Tensor(le_tensor, ge_tensor_1);  le_tensor = ge_tensor_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:853 in create_blockmasks, code: blockmask_any = causal_blockmask_any & document_blockmask_any
        bitwise_and_tensor_1: "b8[48, 48]" = torch.ops.aten.bitwise_and.Tensor(ge_tensor, bitwise_and_tensor);  ge_tensor = bitwise_and_tensor = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:844 in create_blockmasks, code: causal_blockmask_all = block_idx[:, None] > block_idx
        unsqueeze_default_3: "i32[48, 1]" = torch.ops.aten.unsqueeze.default(iota_default, 1)
        gt_tensor: "b8[48, 48]" = torch.ops.aten.gt.Tensor(unsqueeze_default_3, iota_default);  unsqueeze_default_3 = iota_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:850 in create_blockmasks, code: document_blockmask_all = (docs_low[:, None] == docs_high) & (
        unsqueeze_default_4: "i64[48, 1]" = torch.ops.aten.unsqueeze.default(clone_default, 1)
        eq_tensor: "b8[48, 48]" = torch.ops.aten.eq.Tensor(unsqueeze_default_4, clone_default_1);  unsqueeze_default_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:851 in create_blockmasks, code: docs_high[:, None] == docs_low
        unsqueeze_default_5: "i64[48, 1]" = torch.ops.aten.unsqueeze.default(clone_default_1, 1);  clone_default_1 = None
        eq_tensor_1: "b8[48, 48]" = torch.ops.aten.eq.Tensor(unsqueeze_default_5, clone_default);  unsqueeze_default_5 = clone_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:850 in create_blockmasks, code: document_blockmask_all = (docs_low[:, None] == docs_high) & (
        bitwise_and_tensor_2: "b8[48, 48]" = torch.ops.aten.bitwise_and.Tensor(eq_tensor, eq_tensor_1);  eq_tensor = eq_tensor_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:854 in create_blockmasks, code: blockmask_all = causal_blockmask_all & document_blockmask_all
        bitwise_and_tensor_3: "b8[48, 48]" = torch.ops.aten.bitwise_and.Tensor(gt_tensor, bitwise_and_tensor_2);  gt_tensor = bitwise_and_tensor_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:856 in create_blockmasks, code: blockmask_any & ~blockmask_all
        bitwise_not_default: "b8[48, 48]" = torch.ops.aten.bitwise_not.default(bitwise_and_tensor_3)
        bitwise_and_tensor_4: "b8[48, 48]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_tensor_1, bitwise_not_default);  bitwise_and_tensor_1 = bitwise_not_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:833 in dense_to_ordered, code: dense_blockmask.argsort(dim=-1, descending=False, stable=True)
        sort_stable = torch.ops.aten.sort.stable(bitwise_and_tensor_4, stable = True)
        getitem: "i64[48, 48]" = sort_stable[1];  sort_stable = None
        sort_stable_1 = torch.ops.aten.sort.stable(bitwise_and_tensor_3, stable = True)
        getitem_1: "i64[48, 48]" = sort_stable_1[1];  sort_stable_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:400 in create_dense_one, code: dense_mask = kv_indices.new_zeros(num_rows, num_cols + 1, dtype=torch.int32)
        full_default: "i32[1, 1, 48, 49]" = torch.ops.aten.full.default([1, 1, 48, 49], 0, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:412 in create_dense_one, code: dense_mask[row_indices, valid_indices] = dense_mask.new_ones(())
        iota_default_1: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_6: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_default_1, -1);  iota_default_1 = None
        unsqueeze_default_7: "i64[1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        unsqueeze_default_8: "i64[1, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, -1);  unsqueeze_default_7 = None
        iota_default_2: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_9: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_default_2, -1);  iota_default_2 = None
        unsqueeze_default_10: "i64[1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, -1);  unsqueeze_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:402 in create_dense_one, code: row_indices = torch.arange(num_rows, dtype=torch.int, device=device).unsqueeze(
        iota_default_3: "i32[48]" = torch.ops.prims.iota.default(48, start = 0, step = 1, dtype = torch.int32, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_11: "i32[48, 1]" = torch.ops.aten.unsqueeze.default(iota_default_3, -1);  iota_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:405 in create_dense_one, code: col_range = torch.arange(num_cols, dtype=torch.int, device=device)
        iota_default_4: "i32[48]" = torch.ops.prims.iota.default(48, start = 0, step = 1, dtype = torch.int32, device = device(type='cuda', index=0), requires_grad = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:831 in dense_to_ordered, code: num_blocks = dense_blockmask.sum(dim=-1, dtype=torch.int32)
        sum_dim_int_list: "i32[48]" = torch.ops.aten.sum.dim_IntList(bitwise_and_tensor_4, [-1], dtype = torch.int32);  bitwise_and_tensor_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:837 in dense_to_ordered, code: return num_blocks[None, None].contiguous(), indices[None, None].contiguous()
        unsqueeze_default_12: "i32[1, 48]" = torch.ops.aten.unsqueeze.default(sum_dim_int_list, 0);  sum_dim_int_list = None
        unsqueeze_default_13: "i32[1, 1, 48]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, 1);  unsqueeze_default_12 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:831 in dense_to_ordered, code: num_blocks = dense_blockmask.sum(dim=-1, dtype=torch.int32)
        sum_dim_int_list_1: "i32[48]" = torch.ops.aten.sum.dim_IntList(bitwise_and_tensor_3, [-1], dtype = torch.int32);  bitwise_and_tensor_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:837 in dense_to_ordered, code: return num_blocks[None, None].contiguous(), indices[None, None].contiguous()
        unsqueeze_default_14: "i32[1, 48]" = torch.ops.aten.unsqueeze.default(sum_dim_int_list_1, 0);  sum_dim_int_list_1 = None
        unsqueeze_default_15: "i32[1, 1, 48]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 1);  unsqueeze_default_14 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:864 in build_bm, code: torch.clamp_min(window_size_blocks - full_kv_num_blocks, 1),
        sub_tensor: "i32[1, 1, 48]" = torch.ops.aten.sub.Tensor(arg4_1, unsqueeze_default_15)
        clamp_min_default: "i32[1, 1, 48]" = torch.ops.aten.clamp_min.default(sub_tensor, 1);  sub_tensor = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:862 in build_bm, code: torch.clamp_max(
        clamp_max_tensor: "i32[1, 1, 48]" = torch.ops.aten.clamp_max.Tensor(unsqueeze_default_13, clamp_min_default);  unsqueeze_default_13 = clamp_min_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:406 in create_dense_one, code: index_mask = col_range < kv_num_blocks.unsqueeze(-1)
        unsqueeze_default_16: "i32[1, 1, 48, 1]" = torch.ops.aten.unsqueeze.default(clamp_max_tensor, 3);  clamp_max_tensor = None
        lt_tensor: "b8[1, 1, 48, 48]" = torch.ops.aten.lt.Tensor(iota_default_4, unsqueeze_default_16);  iota_default_4 = unsqueeze_default_16 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:834 in dense_to_ordered, code: .flip(-1)
        rev_default: "i64[48, 48]" = torch.ops.prims.rev.default(getitem, [1]);  getitem = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:835 in dense_to_ordered, code: .to(torch.int32)
        convert_element_type_default: "i32[48, 48]" = torch.ops.prims.convert_element_type.default(rev_default, torch.int32);  rev_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:837 in dense_to_ordered, code: return num_blocks[None, None].contiguous(), indices[None, None].contiguous()
        unsqueeze_default_17: "i32[1, 48, 48]" = torch.ops.aten.unsqueeze.default(convert_element_type_default, 0);  convert_element_type_default = None
        unsqueeze_default_18: "i32[1, 1, 48, 48]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_17, 1);  unsqueeze_default_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:409 in create_dense_one, code: valid_indices = torch.where(index_mask, kv_indices, num_cols)
        full_default_1: "i32[]" = torch.ops.aten.full.default([], 48, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i32[1, 1, 48, 48]" = torch.ops.aten.where.self(lt_tensor, unsqueeze_default_18, full_default_1);  lt_tensor = unsqueeze_default_18 = full_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:412 in create_dense_one, code: dense_mask[row_indices, valid_indices] = dense_mask.new_ones(())
        full_default_2: "i32[1, 1, 1, 1]" = torch.ops.aten.full.default([1, 1, 1, 1], 1, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "i32[1, 1, 48, 49]" = torch.ops.aten.index_put.default(full_default, [unsqueeze_default_8, unsqueeze_default_10, unsqueeze_default_11, where_self], full_default_2);  full_default = unsqueeze_default_8 = unsqueeze_default_10 = unsqueeze_default_11 = where_self = full_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:413 in create_dense_one, code: return dense_mask[:, :num_cols].contiguous()
        slice_tensor: "i32[1, 1, 48, 48]" = torch.ops.aten.slice.Tensor(index_put_default, 3, 0, 48);  index_put_default = None
        clone_default_2: "i32[1, 1, 48, 48]" = torch.ops.aten.clone.default(slice_tensor, memory_format = torch.contiguous_format);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:437 in _transpose_ordered, code: return _dense_to_ordered(dense.transpose(-2, -1))
        permute_default: "i32[1, 1, 48, 48]" = torch.ops.aten.permute.default(clone_default_2, [0, 1, 3, 2]);  clone_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:426 in _dense_to_ordered, code: col_indices = torch.argsort(dense_mask, dim=-1, descending=True, stable=True)
        sort_stable_2 = torch.ops.aten.sort.stable(permute_default, stable = True, descending = True)
        getitem_2: "i64[1, 1, 48, 48]" = sort_stable_2[1];  sort_stable_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:400 in create_dense_one, code: dense_mask = kv_indices.new_zeros(num_rows, num_cols + 1, dtype=torch.int32)
        full_default_3: "i32[1, 1, 48, 49]" = torch.ops.aten.full.default([1, 1, 48, 49], 0, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:412 in create_dense_one, code: dense_mask[row_indices, valid_indices] = dense_mask.new_ones(())
        iota_default_5: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_19: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_default_5, -1);  iota_default_5 = None
        unsqueeze_default_20: "i64[1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_19, -1);  unsqueeze_default_19 = None
        unsqueeze_default_21: "i64[1, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_20, -1);  unsqueeze_default_20 = None
        iota_default_6: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_22: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_default_6, -1);  iota_default_6 = None
        unsqueeze_default_23: "i64[1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_22, -1);  unsqueeze_default_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:402 in create_dense_one, code: row_indices = torch.arange(num_rows, dtype=torch.int, device=device).unsqueeze(
        iota_default_7: "i32[48]" = torch.ops.prims.iota.default(48, start = 0, step = 1, dtype = torch.int32, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_24: "i32[48, 1]" = torch.ops.aten.unsqueeze.default(iota_default_7, -1);  iota_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:405 in create_dense_one, code: col_range = torch.arange(num_cols, dtype=torch.int, device=device)
        iota_default_8: "i32[48]" = torch.ops.prims.iota.default(48, start = 0, step = 1, dtype = torch.int32, device = device(type='cuda', index=0), requires_grad = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:867 in build_bm, code: torch.clamp_max(full_kv_num_blocks, window_size_blocks - 1),
        sub_tensor_1: "i32[]" = torch.ops.aten.sub.Tensor(arg4_1, 1);  arg4_1 = None
        clamp_max_tensor_1: "i32[1, 1, 48]" = torch.ops.aten.clamp_max.Tensor(unsqueeze_default_15, sub_tensor_1);  unsqueeze_default_15 = sub_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:406 in create_dense_one, code: index_mask = col_range < kv_num_blocks.unsqueeze(-1)
        unsqueeze_default_25: "i32[1, 1, 48, 1]" = torch.ops.aten.unsqueeze.default(clamp_max_tensor_1, 3);  clamp_max_tensor_1 = None
        lt_tensor_1: "b8[1, 1, 48, 48]" = torch.ops.aten.lt.Tensor(iota_default_8, unsqueeze_default_25);  iota_default_8 = unsqueeze_default_25 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:834 in dense_to_ordered, code: .flip(-1)
        rev_default_1: "i64[48, 48]" = torch.ops.prims.rev.default(getitem_1, [1]);  getitem_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:835 in dense_to_ordered, code: .to(torch.int32)
        convert_element_type_default_1: "i32[48, 48]" = torch.ops.prims.convert_element_type.default(rev_default_1, torch.int32);  rev_default_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:837 in dense_to_ordered, code: return num_blocks[None, None].contiguous(), indices[None, None].contiguous()
        unsqueeze_default_26: "i32[1, 48, 48]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_1, 0);  convert_element_type_default_1 = None
        unsqueeze_default_27: "i32[1, 1, 48, 48]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_26, 1);  unsqueeze_default_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:409 in create_dense_one, code: valid_indices = torch.where(index_mask, kv_indices, num_cols)
        full_default_4: "i32[]" = torch.ops.aten.full.default([], 48, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "i32[1, 1, 48, 48]" = torch.ops.aten.where.self(lt_tensor_1, unsqueeze_default_27, full_default_4);  lt_tensor_1 = unsqueeze_default_27 = full_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:412 in create_dense_one, code: dense_mask[row_indices, valid_indices] = dense_mask.new_ones(())
        full_default_5: "i32[1, 1, 1, 1]" = torch.ops.aten.full.default([1, 1, 1, 1], 1, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_1: "i32[1, 1, 48, 49]" = torch.ops.aten.index_put.default(full_default_3, [unsqueeze_default_21, unsqueeze_default_23, unsqueeze_default_24, where_self_1], full_default_5);  full_default_3 = unsqueeze_default_21 = unsqueeze_default_23 = unsqueeze_default_24 = where_self_1 = full_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:413 in create_dense_one, code: return dense_mask[:, :num_cols].contiguous()
        slice_tensor_1: "i32[1, 1, 48, 48]" = torch.ops.aten.slice.Tensor(index_put_default_1, 3, 0, 48);  index_put_default_1 = None
        clone_default_3: "i32[1, 1, 48, 48]" = torch.ops.aten.clone.default(slice_tensor_1, memory_format = torch.contiguous_format);  slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:437 in _transpose_ordered, code: return _dense_to_ordered(dense.transpose(-2, -1))
        permute_default_1: "i32[1, 1, 48, 48]" = torch.ops.aten.permute.default(clone_default_3, [0, 1, 3, 2]);  clone_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:426 in _dense_to_ordered, code: col_indices = torch.argsort(dense_mask, dim=-1, descending=True, stable=True)
        sort_stable_3 = torch.ops.aten.sort.stable(permute_default_1, stable = True, descending = True)
        getitem_3: "i64[1, 1, 48, 48]" = sort_stable_3[1];  sort_stable_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:425 in _dense_to_ordered, code: num_blocks_in_row = dense_mask.sum(dim=-1)
        sum_dim_int_list_2: "i64[1, 1, 48]" = torch.ops.aten.sum.dim_IntList(permute_default, [-1]);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:428 in _dense_to_ordered, code: num_blocks_in_row.to(torch.int32, memory_format=torch.contiguous_format),
        convert_element_type_default_2: "i32[1, 1, 48]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_2, torch.int32);  sum_dim_int_list_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:429 in _dense_to_ordered, code: col_indices.to(torch.int32, memory_format=torch.contiguous_format),
        convert_element_type_default_3: "i32[1, 1, 48, 48]" = torch.ops.prims.convert_element_type.default(getitem_2, torch.int32);  getitem_2 = None
        clone_default_4: "i32[1, 1, 48, 48]" = torch.ops.aten.clone.default(convert_element_type_default_3, memory_format = torch.contiguous_format);  convert_element_type_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:425 in _dense_to_ordered, code: num_blocks_in_row = dense_mask.sum(dim=-1)
        sum_dim_int_list_3: "i64[1, 1, 48]" = torch.ops.aten.sum.dim_IntList(permute_default_1, [-1]);  permute_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:428 in _dense_to_ordered, code: num_blocks_in_row.to(torch.int32, memory_format=torch.contiguous_format),
        convert_element_type_default_4: "i32[1, 1, 48]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_3, torch.int32);  sum_dim_int_list_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:429 in _dense_to_ordered, code: col_indices.to(torch.int32, memory_format=torch.contiguous_format),
        convert_element_type_default_5: "i32[1, 1, 48, 48]" = torch.ops.prims.convert_element_type.default(getitem_3, torch.int32);  getitem_3 = None
        clone_default_5: "i32[1, 1, 48, 48]" = torch.ops.aten.clone.default(convert_element_type_default_5, memory_format = torch.contiguous_format);  convert_element_type_default_5 = None
        return (convert_element_type_default_4, convert_element_type_default_2, clone_default_5, clone_default_4)



def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
