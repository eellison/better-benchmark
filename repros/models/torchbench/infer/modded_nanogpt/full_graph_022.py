import torch
from torch import device
from math import inf, nan

# Register nanogpt custom ops (from modded_nanogpt benchmark model)
if not hasattr(torch.ops, 'nanogpt') or not hasattr(torch.ops.nanogpt, 'mm'):
    @torch.library.custom_op("nanogpt::mm", mutates_args=())
    def _mm_op(x: torch.Tensor, w: torch.Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        w_f8 = w.div(w_s).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(x_f8, w_f8.T, out_dtype=torch.bfloat16,
                               scale_a=x.new_tensor(x_s, dtype=torch.float32),
                               scale_b=x.new_tensor(w_s, dtype=torch.float32), use_fast_accum=True)
        return out, x_f8, w_f8

    @_mm_op.register_fake
    def _mm_fake(x: torch.Tensor, w: torch.Tensor, *_):
        return x @ w.T, x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)

    @torch.library.custom_op("nanogpt::mm_backward", mutates_args=())
    def _mm_backward_op(g: torch.Tensor, x_f8: torch.Tensor, w_f8: torch.Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[torch.Tensor, torch.Tensor]:
        grad_f8 = g.div(grad_s).to(torch.float8_e5m2)
        grad_x = torch._scaled_mm(grad_f8, w_f8.T.contiguous().T, out_dtype=torch.bfloat16,
                                   scale_a=g.new_tensor(grad_s, dtype=torch.float32),
                                   scale_b=g.new_tensor(w_s, dtype=torch.float32), use_fast_accum=False)
        grad_w = torch._scaled_mm(x_f8.T.contiguous(), grad_f8.T.contiguous().T, out_dtype=torch.float32,
                                  scale_a=g.new_tensor(x_s, dtype=torch.float32),
                                  scale_b=g.new_tensor(grad_s, dtype=torch.float32), use_fast_accum=False)
        return grad_x, grad_w.T

    @_mm_backward_op.register_fake
    def _mm_backward_fake(g: torch.Tensor, x_f8: torch.Tensor, w_f8: torch.Tensor, *_):
        return x_f8.to(torch.bfloat16), w_f8.T.contiguous().T.to(torch.float32)


class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i32[6144]", arg1_1: "bf16[50304, 768]", arg2_1: "bf16[50304, 768]", arg3_1: "bf16[50304, 768]", arg4_1: "i32[]", arg5_1: "bf16[50304, 768]", arg6_1: "f32[64]", arg7_1: "f32[4, 768, 768]", arg8_1: "f32[262144, 64]", arg9_1: "f32[262144, 64]", arg10_1: "f32[6, 12]", arg11_1: "f32[768, 3072]", arg12_1: "f32[768, 3072]", arg13_1: "f32[4, 768, 768]", arg14_1: "f32[262144, 64]", arg15_1: "f32[262144, 64]", arg16_1: "f32[6, 12]", arg17_1: "f32[768, 3072]", arg18_1: "f32[768, 3072]", arg19_1: "f32[4, 768, 768]", arg20_1: "f32[262144, 64]", arg21_1: "f32[262144, 64]", arg22_1: "f32[6, 12]", arg23_1: "f32[768, 3072]", arg24_1: "f32[768, 3072]", arg25_1: "f32[4, 768, 768]", arg26_1: "f32[262144, 64]", arg27_1: "f32[262144, 64]", arg28_1: "f32[6, 12]", arg29_1: "f32[768, 3072]", arg30_1: "f32[768, 3072]", arg31_1: "f32[4, 768, 768]", arg32_1: "f32[262144, 64]", arg33_1: "f32[262144, 64]", arg34_1: "f32[6, 12]", arg35_1: "f32[768, 3072]", arg36_1: "f32[768, 3072]", arg37_1: "f32[4, 768, 768]", arg38_1: "f32[262144, 64]", arg39_1: "f32[262144, 64]", arg40_1: "f32[6, 12]", arg41_1: "f32[768, 3072]", arg42_1: "f32[768, 3072]", arg43_1: "f32[4, 768, 768]", arg44_1: "f32[262144, 64]", arg45_1: "f32[262144, 64]", arg46_1: "f32[6, 12]", arg47_1: "f32[768, 3072]", arg48_1: "f32[768, 3072]", arg49_1: "f32[768, 3072]", arg50_1: "f32[768, 3072]", arg51_1: "f32[4, 768, 768]", arg52_1: "f32[262144, 64]", arg53_1: "f32[262144, 64]", arg54_1: "f32[6, 12]", arg55_1: "f32[768, 3072]", arg56_1: "f32[768, 3072]", arg57_1: "f32[4, 768, 768]", arg58_1: "f32[262144, 64]", arg59_1: "f32[262144, 64]", arg60_1: "f32[6, 12]", arg61_1: "f32[768, 3072]", arg62_1: "f32[768, 3072]", arg63_1: "f32[4, 768, 768]", arg64_1: "f32[262144, 64]", arg65_1: "f32[262144, 64]", arg66_1: "f32[6, 12]", arg67_1: "f32[768, 3072]", arg68_1: "f32[768, 3072]", arg69_1: "f32[4, 768, 768]", arg70_1: "f32[262144, 64]", arg71_1: "f32[262144, 64]", arg72_1: "f32[6, 12]", arg73_1: "f32[768, 3072]", arg74_1: "f32[768, 3072]", arg75_1: "f32[50304, 768]", arg76_1: "i64[6144]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:842 in create_blockmasks, code: block_idx = torch.arange(NUM_BLOCKS, dtype=torch.int32, device="cuda")
        iota: "i32[48]" = torch.ops.prims.iota.default(48, start = 0, step = 1, dtype = torch.int32, device = device(type='cuda'), requires_grad = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:843 in create_blockmasks, code: causal_blockmask_any = block_idx[:, None] >= block_idx
        unsqueeze: "i32[48, 1]" = torch.ops.aten.unsqueeze.default(iota, 1)
        ge: "b8[48, 48]" = torch.ops.aten.ge.Tensor(unsqueeze, iota);  unsqueeze = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:823 in create_blockmasks, code: docs = (input_seq == 50256).cumsum(0)
        eq: "b8[6144]" = torch.ops.aten.eq.Scalar(arg0_1, 50256)
        cumsum: "i64[6144]" = torch.ops.aten.cumsum.default(eq, 0);  eq = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:845 in create_blockmasks, code: docs_low = docs.view(-1, BLOCK_SIZE)[:, 0].contiguous()
        view: "i64[48, 128]" = torch.ops.aten.reshape.default(cumsum, [-1, 128])
        select: "i64[48]" = torch.ops.aten.select.int(view, 1, 0);  view = None
        clone: "i64[48]" = torch.ops.aten.clone.default(select, memory_format = torch.contiguous_format);  select = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:847 in create_blockmasks, code: document_blockmask_any = (docs_low[:, None] <= docs_high) & (
        unsqueeze_2: "i64[48, 1]" = torch.ops.aten.unsqueeze.default(clone, 1)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:846 in create_blockmasks, code: docs_high = docs.view(-1, BLOCK_SIZE)[:, -1].contiguous()
        view_1: "i64[48, 128]" = torch.ops.aten.reshape.default(cumsum, [-1, 128])
        select_1: "i64[48]" = torch.ops.aten.select.int(view_1, 1, -1);  view_1 = None
        clone_1: "i64[48]" = torch.ops.aten.clone.default(select_1, memory_format = torch.contiguous_format);  select_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:847 in create_blockmasks, code: document_blockmask_any = (docs_low[:, None] <= docs_high) & (
        le: "b8[48, 48]" = torch.ops.aten.le.Tensor(unsqueeze_2, clone_1);  unsqueeze_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:848 in create_blockmasks, code: docs_high[:, None] >= docs_low
        unsqueeze_3: "i64[48, 1]" = torch.ops.aten.unsqueeze.default(clone_1, 1)
        ge_1: "b8[48, 48]" = torch.ops.aten.ge.Tensor(unsqueeze_3, clone);  unsqueeze_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:847 in create_blockmasks, code: document_blockmask_any = (docs_low[:, None] <= docs_high) & (
        bitwise_and: "b8[48, 48]" = torch.ops.aten.bitwise_and.Tensor(le, ge_1);  le = ge_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:853 in create_blockmasks, code: blockmask_any = causal_blockmask_any & document_blockmask_any
        bitwise_and_2: "b8[48, 48]" = torch.ops.aten.bitwise_and.Tensor(ge, bitwise_and);  ge = bitwise_and = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:844 in create_blockmasks, code: causal_blockmask_all = block_idx[:, None] > block_idx
        unsqueeze_1: "i32[48, 1]" = torch.ops.aten.unsqueeze.default(iota, 1)
        gt: "b8[48, 48]" = torch.ops.aten.gt.Tensor(unsqueeze_1, iota);  unsqueeze_1 = iota = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:850 in create_blockmasks, code: document_blockmask_all = (docs_low[:, None] == docs_high) & (
        unsqueeze_4: "i64[48, 1]" = torch.ops.aten.unsqueeze.default(clone, 1)
        eq_1: "b8[48, 48]" = torch.ops.aten.eq.Tensor(unsqueeze_4, clone_1);  unsqueeze_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:851 in create_blockmasks, code: docs_high[:, None] == docs_low
        unsqueeze_5: "i64[48, 1]" = torch.ops.aten.unsqueeze.default(clone_1, 1);  clone_1 = None
        eq_2: "b8[48, 48]" = torch.ops.aten.eq.Tensor(unsqueeze_5, clone);  unsqueeze_5 = clone = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:850 in create_blockmasks, code: document_blockmask_all = (docs_low[:, None] == docs_high) & (
        bitwise_and_1: "b8[48, 48]" = torch.ops.aten.bitwise_and.Tensor(eq_1, eq_2);  eq_1 = eq_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:854 in create_blockmasks, code: blockmask_all = causal_blockmask_all & document_blockmask_all
        bitwise_and_3: "b8[48, 48]" = torch.ops.aten.bitwise_and.Tensor(gt, bitwise_and_1);  gt = bitwise_and_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:856 in create_blockmasks, code: blockmask_any & ~blockmask_all
        bitwise_not: "b8[48, 48]" = torch.ops.aten.bitwise_not.default(bitwise_and_3)
        bitwise_and_4: "b8[48, 48]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_2, bitwise_not);  bitwise_and_2 = bitwise_not = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:833 in dense_to_ordered, code: dense_blockmask.argsort(dim=-1, descending=False, stable=True)
        sort = torch.ops.aten.sort.stable(bitwise_and_4, stable = True)
        getitem_1: "i64[48, 48]" = sort[1];  sort = None
        sort_1 = torch.ops.aten.sort.stable(bitwise_and_3, stable = True)
        getitem_3: "i64[48, 48]" = sort_1[1];  sort_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:412 in create_dense_one, code: dense_mask[row_indices, valid_indices] = dense_mask.new_ones(())
        full_1: "i32[1, 1]" = torch.ops.aten.full.default([1, 1], 1, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:400 in create_dense_one, code: dense_mask = kv_indices.new_zeros(num_rows, num_cols + 1, dtype=torch.int32)
        full_default: "i32[1, 1, 48, 49]" = torch.ops.aten.full.default([1, 1, 48, 49], 0, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:412 in create_dense_one, code: dense_mask[row_indices, valid_indices] = dense_mask.new_ones(())
        iota_4: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_18: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_4, -1);  iota_4 = None
        unsqueeze_19: "i64[1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        unsqueeze_20: "i64[1, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_19, -1);  unsqueeze_19 = None
        iota_3: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_16: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_3, -1);  iota_3 = None
        unsqueeze_17: "i64[1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:402 in create_dense_one, code: row_indices = torch.arange(num_rows, dtype=torch.int, device=device).unsqueeze(
        iota_1: "i32[48]" = torch.ops.prims.iota.default(48, start = 0, step = 1, dtype = torch.int32, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_14: "i32[48, 1]" = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:405 in create_dense_one, code: col_range = torch.arange(num_cols, dtype=torch.int, device=device)
        iota_2: "i32[48]" = torch.ops.prims.iota.default(48, start = 0, step = 1, dtype = torch.int32, device = device(type='cuda', index=0), requires_grad = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:831 in dense_to_ordered, code: num_blocks = dense_blockmask.sum(dim=-1, dtype=torch.int32)
        sum_1: "i32[48]" = torch.ops.aten.sum.dim_IntList(bitwise_and_4, [-1], dtype = torch.int32);  bitwise_and_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:837 in dense_to_ordered, code: return num_blocks[None, None].contiguous(), indices[None, None].contiguous()
        unsqueeze_6: "i32[1, 48]" = torch.ops.aten.unsqueeze.default(sum_1, 0);  sum_1 = None
        unsqueeze_7: "i32[1, 1, 48]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 1);  unsqueeze_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:831 in dense_to_ordered, code: num_blocks = dense_blockmask.sum(dim=-1, dtype=torch.int32)
        sum_2: "i32[48]" = torch.ops.aten.sum.dim_IntList(bitwise_and_3, [-1], dtype = torch.int32);  bitwise_and_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:837 in dense_to_ordered, code: return num_blocks[None, None].contiguous(), indices[None, None].contiguous()
        unsqueeze_10: "i32[1, 48]" = torch.ops.aten.unsqueeze.default(sum_2, 0);  sum_2 = None
        unsqueeze_11: "i32[1, 1, 48]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 1);  unsqueeze_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:864 in build_bm, code: torch.clamp_min(window_size_blocks - full_kv_num_blocks, 1),
        sub: "i32[1, 1, 48]" = torch.ops.aten.sub.Tensor(arg4_1, unsqueeze_11)
        clamp_min: "i32[1, 1, 48]" = torch.ops.aten.clamp_min.default(sub, 1);  sub = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:862 in build_bm, code: torch.clamp_max(
        clamp_max: "i32[1, 1, 48]" = torch.ops.aten.clamp_max.Tensor(unsqueeze_7, clamp_min);  clamp_min = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:406 in create_dense_one, code: index_mask = col_range < kv_num_blocks.unsqueeze(-1)
        unsqueeze_15: "i32[1, 1, 48, 1]" = torch.ops.aten.unsqueeze.default(clamp_max, 3)
        lt: "b8[1, 1, 48, 48]" = torch.ops.aten.lt.Tensor(iota_2, unsqueeze_15);  iota_2 = unsqueeze_15 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:834 in dense_to_ordered, code: .flip(-1)
        rev: "i64[48, 48]" = torch.ops.prims.rev.default(getitem_1, [1]);  getitem_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:835 in dense_to_ordered, code: .to(torch.int32)
        convert_element_type: "i32[48, 48]" = torch.ops.prims.convert_element_type.default(rev, torch.int32);  rev = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:837 in dense_to_ordered, code: return num_blocks[None, None].contiguous(), indices[None, None].contiguous()
        unsqueeze_8: "i32[1, 48, 48]" = torch.ops.aten.unsqueeze.default(convert_element_type, 0);  convert_element_type = None
        unsqueeze_9: "i32[1, 1, 48, 48]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, 1);  unsqueeze_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:409 in create_dense_one, code: valid_indices = torch.where(index_mask, kv_indices, num_cols)
        full_default_1: "i32[]" = torch.ops.aten.full.default([], 48, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i32[1, 1, 48, 48]" = torch.ops.aten.where.self(lt, unsqueeze_9, full_default_1);  lt = full_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:412 in create_dense_one, code: dense_mask[row_indices, valid_indices] = dense_mask.new_ones(())
        full_default_2: "i32[1, 1, 1, 1]" = torch.ops.aten.full.default([1, 1, 1, 1], 1, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put: "i32[1, 1, 48, 49]" = torch.ops.aten.index_put.default(full_default, [unsqueeze_20, unsqueeze_17, unsqueeze_14, where], full_default_2);  full_default = unsqueeze_20 = unsqueeze_17 = unsqueeze_14 = where = full_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:413 in create_dense_one, code: return dense_mask[:, :num_cols].contiguous()
        slice_2: "i32[1, 1, 48, 48]" = torch.ops.aten.slice.Tensor(index_put, 3, 0, 48);  index_put = None
        clone_2: "i32[1, 1, 48, 48]" = torch.ops.aten.clone.default(slice_2, memory_format = torch.contiguous_format);  slice_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:437 in _transpose_ordered, code: return _dense_to_ordered(dense.transpose(-2, -1))
        permute: "i32[1, 1, 48, 48]" = torch.ops.aten.permute.default(clone_2, [0, 1, 3, 2]);  clone_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:426 in _dense_to_ordered, code: col_indices = torch.argsort(dense_mask, dim=-1, descending=True, stable=True)
        sort_2 = torch.ops.aten.sort.stable(permute, stable = True, descending = True)
        getitem_5: "i64[1, 1, 48, 48]" = sort_2[1];  sort_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:412 in create_dense_one, code: dense_mask[row_indices, valid_indices] = dense_mask.new_ones(())
        full_3: "i32[1, 1]" = torch.ops.aten.full.default([1, 1], 1, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:400 in create_dense_one, code: dense_mask = kv_indices.new_zeros(num_rows, num_cols + 1, dtype=torch.int32)
        full_default_3: "i32[1, 1, 48, 49]" = torch.ops.aten.full.default([1, 1, 48, 49], 0, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:412 in create_dense_one, code: dense_mask[row_indices, valid_indices] = dense_mask.new_ones(())
        iota_8: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_25: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_8, -1);  iota_8 = None
        unsqueeze_26: "i64[1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_25, -1);  unsqueeze_25 = None
        unsqueeze_27: "i64[1, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
        iota_7: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_23: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_7, -1);  iota_7 = None
        unsqueeze_24: "i64[1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_23, -1);  unsqueeze_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:402 in create_dense_one, code: row_indices = torch.arange(num_rows, dtype=torch.int, device=device).unsqueeze(
        iota_5: "i32[48]" = torch.ops.prims.iota.default(48, start = 0, step = 1, dtype = torch.int32, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_21: "i32[48, 1]" = torch.ops.aten.unsqueeze.default(iota_5, -1);  iota_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:405 in create_dense_one, code: col_range = torch.arange(num_cols, dtype=torch.int, device=device)
        iota_6: "i32[48]" = torch.ops.prims.iota.default(48, start = 0, step = 1, dtype = torch.int32, device = device(type='cuda', index=0), requires_grad = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:867 in build_bm, code: torch.clamp_max(full_kv_num_blocks, window_size_blocks - 1),
        sub_1: "i32[]" = torch.ops.aten.sub.Tensor(arg4_1, 1)
        clamp_max_1: "i32[1, 1, 48]" = torch.ops.aten.clamp_max.Tensor(unsqueeze_11, sub_1);  sub_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:406 in create_dense_one, code: index_mask = col_range < kv_num_blocks.unsqueeze(-1)
        unsqueeze_22: "i32[1, 1, 48, 1]" = torch.ops.aten.unsqueeze.default(clamp_max_1, 3)
        lt_1: "b8[1, 1, 48, 48]" = torch.ops.aten.lt.Tensor(iota_6, unsqueeze_22);  iota_6 = unsqueeze_22 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:834 in dense_to_ordered, code: .flip(-1)
        rev_1: "i64[48, 48]" = torch.ops.prims.rev.default(getitem_3, [1]);  getitem_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:835 in dense_to_ordered, code: .to(torch.int32)
        convert_element_type_1: "i32[48, 48]" = torch.ops.prims.convert_element_type.default(rev_1, torch.int32);  rev_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:837 in dense_to_ordered, code: return num_blocks[None, None].contiguous(), indices[None, None].contiguous()
        unsqueeze_12: "i32[1, 48, 48]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 0);  convert_element_type_1 = None
        unsqueeze_13: "i32[1, 1, 48, 48]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, 1);  unsqueeze_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:409 in create_dense_one, code: valid_indices = torch.where(index_mask, kv_indices, num_cols)
        full_default_4: "i32[]" = torch.ops.aten.full.default([], 48, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "i32[1, 1, 48, 48]" = torch.ops.aten.where.self(lt_1, unsqueeze_13, full_default_4);  lt_1 = full_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:412 in create_dense_one, code: dense_mask[row_indices, valid_indices] = dense_mask.new_ones(())
        full_default_5: "i32[1, 1, 1, 1]" = torch.ops.aten.full.default([1, 1, 1, 1], 1, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_1: "i32[1, 1, 48, 49]" = torch.ops.aten.index_put.default(full_default_3, [unsqueeze_27, unsqueeze_24, unsqueeze_21, where_1], full_default_5);  full_default_3 = unsqueeze_27 = unsqueeze_24 = unsqueeze_21 = where_1 = full_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:413 in create_dense_one, code: return dense_mask[:, :num_cols].contiguous()
        slice_4: "i32[1, 1, 48, 48]" = torch.ops.aten.slice.Tensor(index_put_1, 3, 0, 48);  index_put_1 = None
        clone_5: "i32[1, 1, 48, 48]" = torch.ops.aten.clone.default(slice_4, memory_format = torch.contiguous_format);  slice_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:437 in _transpose_ordered, code: return _dense_to_ordered(dense.transpose(-2, -1))
        permute_1: "i32[1, 1, 48, 48]" = torch.ops.aten.permute.default(clone_5, [0, 1, 3, 2]);  clone_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:426 in _dense_to_ordered, code: col_indices = torch.argsort(dense_mask, dim=-1, descending=True, stable=True)
        sort_3 = torch.ops.aten.sort.stable(permute_1, stable = True, descending = True)
        getitem_7: "i64[1, 1, 48, 48]" = sort_3[1];  sort_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:412 in create_dense_one, code: dense_mask[row_indices, valid_indices] = dense_mask.new_ones(())
        full_5: "i32[1, 1]" = torch.ops.aten.full.default([1, 1], 1, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:400 in create_dense_one, code: dense_mask = kv_indices.new_zeros(num_rows, num_cols + 1, dtype=torch.int32)
        full_default_6: "i32[1, 1, 48, 49]" = torch.ops.aten.full.default([1, 1, 48, 49], 0, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:412 in create_dense_one, code: dense_mask[row_indices, valid_indices] = dense_mask.new_ones(())
        iota_12: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_32: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_12, -1);  iota_12 = None
        unsqueeze_33: "i64[1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        unsqueeze_34: "i64[1, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_33, -1);  unsqueeze_33 = None
        iota_11: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_30: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_11, -1);  iota_11 = None
        unsqueeze_31: "i64[1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:402 in create_dense_one, code: row_indices = torch.arange(num_rows, dtype=torch.int, device=device).unsqueeze(
        iota_9: "i32[48]" = torch.ops.prims.iota.default(48, start = 0, step = 1, dtype = torch.int32, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_28: "i32[48, 1]" = torch.ops.aten.unsqueeze.default(iota_9, -1);  iota_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:405 in create_dense_one, code: col_range = torch.arange(num_cols, dtype=torch.int, device=device)
        iota_10: "i32[48]" = torch.ops.prims.iota.default(48, start = 0, step = 1, dtype = torch.int32, device = device(type='cuda', index=0), requires_grad = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:875 in create_blockmasks, code: sliding_window_num_blocks // 2
        div: "i32[]" = torch.ops.aten.div.Tensor_mode(arg4_1, 2, rounding_mode = 'floor');  arg4_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:864 in build_bm, code: torch.clamp_min(window_size_blocks - full_kv_num_blocks, 1),
        sub_2: "i32[1, 1, 48]" = torch.ops.aten.sub.Tensor(div, unsqueeze_11)
        clamp_min_1: "i32[1, 1, 48]" = torch.ops.aten.clamp_min.default(sub_2, 1);  sub_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:862 in build_bm, code: torch.clamp_max(
        clamp_max_2: "i32[1, 1, 48]" = torch.ops.aten.clamp_max.Tensor(unsqueeze_7, clamp_min_1);  unsqueeze_7 = clamp_min_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:406 in create_dense_one, code: index_mask = col_range < kv_num_blocks.unsqueeze(-1)
        unsqueeze_29: "i32[1, 1, 48, 1]" = torch.ops.aten.unsqueeze.default(clamp_max_2, 3)
        lt_2: "b8[1, 1, 48, 48]" = torch.ops.aten.lt.Tensor(iota_10, unsqueeze_29);  iota_10 = unsqueeze_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:409 in create_dense_one, code: valid_indices = torch.where(index_mask, kv_indices, num_cols)
        full_default_7: "i32[]" = torch.ops.aten.full.default([], 48, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "i32[1, 1, 48, 48]" = torch.ops.aten.where.self(lt_2, unsqueeze_9, full_default_7);  lt_2 = full_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:412 in create_dense_one, code: dense_mask[row_indices, valid_indices] = dense_mask.new_ones(())
        full_default_8: "i32[1, 1, 1, 1]" = torch.ops.aten.full.default([1, 1, 1, 1], 1, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_2: "i32[1, 1, 48, 49]" = torch.ops.aten.index_put.default(full_default_6, [unsqueeze_34, unsqueeze_31, unsqueeze_28, where_2], full_default_8);  full_default_6 = unsqueeze_34 = unsqueeze_31 = unsqueeze_28 = where_2 = full_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:413 in create_dense_one, code: return dense_mask[:, :num_cols].contiguous()
        slice_6: "i32[1, 1, 48, 48]" = torch.ops.aten.slice.Tensor(index_put_2, 3, 0, 48);  index_put_2 = None
        clone_8: "i32[1, 1, 48, 48]" = torch.ops.aten.clone.default(slice_6, memory_format = torch.contiguous_format);  slice_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:437 in _transpose_ordered, code: return _dense_to_ordered(dense.transpose(-2, -1))
        permute_2: "i32[1, 1, 48, 48]" = torch.ops.aten.permute.default(clone_8, [0, 1, 3, 2]);  clone_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:426 in _dense_to_ordered, code: col_indices = torch.argsort(dense_mask, dim=-1, descending=True, stable=True)
        sort_4 = torch.ops.aten.sort.stable(permute_2, stable = True, descending = True)
        getitem_9: "i64[1, 1, 48, 48]" = sort_4[1];  sort_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:412 in create_dense_one, code: dense_mask[row_indices, valid_indices] = dense_mask.new_ones(())
        full_7: "i32[1, 1]" = torch.ops.aten.full.default([1, 1], 1, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:400 in create_dense_one, code: dense_mask = kv_indices.new_zeros(num_rows, num_cols + 1, dtype=torch.int32)
        full_default_9: "i32[1, 1, 48, 49]" = torch.ops.aten.full.default([1, 1, 48, 49], 0, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:412 in create_dense_one, code: dense_mask[row_indices, valid_indices] = dense_mask.new_ones(())
        iota_16: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_39: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_16, -1);  iota_16 = None
        unsqueeze_40: "i64[1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_39, -1);  unsqueeze_39 = None
        unsqueeze_41: "i64[1, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        iota_15: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_37: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_15, -1);  iota_15 = None
        unsqueeze_38: "i64[1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_37, -1);  unsqueeze_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:402 in create_dense_one, code: row_indices = torch.arange(num_rows, dtype=torch.int, device=device).unsqueeze(
        iota_13: "i32[48]" = torch.ops.prims.iota.default(48, start = 0, step = 1, dtype = torch.int32, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_35: "i32[48, 1]" = torch.ops.aten.unsqueeze.default(iota_13, -1);  iota_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:405 in create_dense_one, code: col_range = torch.arange(num_cols, dtype=torch.int, device=device)
        iota_14: "i32[48]" = torch.ops.prims.iota.default(48, start = 0, step = 1, dtype = torch.int32, device = device(type='cuda', index=0), requires_grad = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:867 in build_bm, code: torch.clamp_max(full_kv_num_blocks, window_size_blocks - 1),
        sub_3: "i32[]" = torch.ops.aten.sub.Tensor(div, 1);  div = None
        clamp_max_3: "i32[1, 1, 48]" = torch.ops.aten.clamp_max.Tensor(unsqueeze_11, sub_3);  unsqueeze_11 = sub_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:406 in create_dense_one, code: index_mask = col_range < kv_num_blocks.unsqueeze(-1)
        unsqueeze_36: "i32[1, 1, 48, 1]" = torch.ops.aten.unsqueeze.default(clamp_max_3, 3)
        lt_3: "b8[1, 1, 48, 48]" = torch.ops.aten.lt.Tensor(iota_14, unsqueeze_36);  iota_14 = unsqueeze_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:409 in create_dense_one, code: valid_indices = torch.where(index_mask, kv_indices, num_cols)
        full_default_10: "i32[]" = torch.ops.aten.full.default([], 48, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "i32[1, 1, 48, 48]" = torch.ops.aten.where.self(lt_3, unsqueeze_13, full_default_10);  lt_3 = full_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:412 in create_dense_one, code: dense_mask[row_indices, valid_indices] = dense_mask.new_ones(())
        full_default_11: "i32[1, 1, 1, 1]" = torch.ops.aten.full.default([1, 1, 1, 1], 1, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_3: "i32[1, 1, 48, 49]" = torch.ops.aten.index_put.default(full_default_9, [unsqueeze_41, unsqueeze_38, unsqueeze_35, where_3], full_default_11);  full_default_9 = unsqueeze_41 = unsqueeze_38 = unsqueeze_35 = where_3 = full_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:413 in create_dense_one, code: return dense_mask[:, :num_cols].contiguous()
        slice_8: "i32[1, 1, 48, 48]" = torch.ops.aten.slice.Tensor(index_put_3, 3, 0, 48);  index_put_3 = None
        clone_11: "i32[1, 1, 48, 48]" = torch.ops.aten.clone.default(slice_8, memory_format = torch.contiguous_format);  slice_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:437 in _transpose_ordered, code: return _dense_to_ordered(dense.transpose(-2, -1))
        permute_3: "i32[1, 1, 48, 48]" = torch.ops.aten.permute.default(clone_11, [0, 1, 3, 2]);  clone_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:426 in _dense_to_ordered, code: col_indices = torch.argsort(dense_mask, dim=-1, descending=True, stable=True)
        sort_5 = torch.ops.aten.sort.stable(permute_3, stable = True, descending = True)
        getitem_11: "i64[1, 1, 48, 48]" = sort_5[1];  sort_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:914 in forward, code: lambdas = self.scalars[1 * len(self.blocks) : 3 * len(self.blocks)].view(-1, 2)
        slice_10: "f32[24]" = torch.ops.aten.slice.Tensor(arg6_1, 0, 12, 36)
        view_6: "f32[12, 2]" = torch.ops.aten.reshape.default(slice_10, [-1, 2]);  slice_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_2: "f32[2]" = torch.ops.aten.select.int(view_6, 0, 0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        select_4: "f32[]" = torch.ops.aten.select.int(select_2, 0, 0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:909 in forward, code: x = x0 = norm(self.embed(input_seq)[None])  # use of norm here by @Grad62304977
        embedding_3: "bf16[6144, 768]" = torch.ops.aten.embedding.default(arg5_1, arg0_1);  arg5_1 = None
        unsqueeze_42: "bf16[1, 6144, 768]" = torch.ops.aten.unsqueeze.default(embedding_3, 0);  embedding_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_10: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(unsqueeze_42, torch.float32);  unsqueeze_42 = None
        pow_1: "f32[1, 6144, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_10, 2)
        mean: "f32[1, 6144, 1]" = torch.ops.aten.mean.dim(pow_1, [2], True);  pow_1 = None
        add: "f32[1, 6144, 1]" = torch.ops.aten.add.Scalar(mean, 1.1920928955078125e-07);  mean = None
        rsqrt: "f32[1, 6144, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_10, rsqrt);  convert_element_type_10 = rsqrt = None
        convert_element_type_11: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        mul_1: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(select_4, convert_element_type_11);  select_4 = None
        select_5: "f32[]" = torch.ops.aten.select.int(select_2, 0, 1);  select_2 = None
        mul_2: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(select_5, convert_element_type_11);  select_5 = None
        add_1: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_12: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_1, torch.float32)
        pow_2: "f32[1, 6144, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_12, 2)
        mean_1: "f32[1, 6144, 1]" = torch.ops.aten.mean.dim(pow_2, [2], True);  pow_2 = None
        add_2: "f32[1, 6144, 1]" = torch.ops.aten.add.Scalar(mean_1, 1.1920928955078125e-07);  mean_1 = None
        rsqrt_1: "f32[1, 6144, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul_3: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_12, rsqrt_1);  convert_element_type_12 = rsqrt_1 = None
        convert_element_type_13: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        view_9: "bf16[6144, 768]" = torch.ops.aten.reshape.default(convert_element_type_13, [6144, 768])
        slice_12: "f32[3, 768, 768]" = torch.ops.aten.slice.Tensor(arg7_1, 0, 0, 3)
        view_8: "f32[2304, 768]" = torch.ops.aten.reshape.default(slice_12, [2304, 768]);  slice_12 = None
        convert_element_type_14: "bf16[2304, 768]" = torch.ops.prims.convert_element_type.default(view_8, torch.bfloat16);  view_8 = None
        permute_4: "bf16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_14, [1, 0]);  convert_element_type_14 = None
        mm: "bf16[6144, 2304]" = torch.ops.aten.mm.default(view_9, permute_4);  view_9 = permute_4 = None
        view_10: "bf16[1, 6144, 2304]" = torch.ops.aten.reshape.default(mm, [1, 6144, 2304]);  mm = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:679 in forward, code: .view(B, T, 3 * self.num_heads, self.head_dim)
        view_11: "bf16[1, 6144, 18, 128]" = torch.ops.aten.reshape.default(view_10, [1, 6144, 18, 128]);  view_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:680 in forward, code: .chunk(3, dim=-2)
        split = torch.ops.aten.split.Tensor(view_11, 6, -2);  view_11 = None
        getitem_12: "bf16[1, 6144, 6, 128]" = split[0]
        getitem_13: "bf16[1, 6144, 6, 128]" = split[1]
        getitem_14: "bf16[1, 6144, 6, 128]" = split[2];  split = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_17: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_12, torch.float32);  getitem_12 = None
        pow_3: "f32[1, 6144, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_17, 2)
        mean_2: "f32[1, 6144, 6, 1]" = torch.ops.aten.mean.dim(pow_3, [3], True);  pow_3 = None
        add_3: "f32[1, 6144, 6, 1]" = torch.ops.aten.add.Scalar(mean_2, 1.1920928955078125e-07);  mean_2 = None
        rsqrt_2: "f32[1, 6144, 6, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        mul_4: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_17, rsqrt_2);  convert_element_type_17 = rsqrt_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        convert_element_type_default_21: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_4, torch.float32);  mul_4 = None
        split_1 = torch.ops.aten.split.Tensor(convert_element_type_default_21, 64, -1);  convert_element_type_default_21 = None
        getitem_15: "f32[1, 6144, 6, 64]" = split_1[0]
        getitem_16: "f32[1, 6144, 6, 64]" = split_1[1];  split_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_19: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_13, torch.float32);  getitem_13 = None
        pow_4: "f32[1, 6144, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_19, 2)
        mean_3: "f32[1, 6144, 6, 1]" = torch.ops.aten.mean.dim(pow_4, [3], True);  pow_4 = None
        add_4: "f32[1, 6144, 6, 1]" = torch.ops.aten.add.Scalar(mean_3, 1.1920928955078125e-07);  mean_3 = None
        rsqrt_3: "f32[1, 6144, 6, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        mul_5: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_19, rsqrt_3);  convert_element_type_19 = rsqrt_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        convert_element_type_default_20: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_5, torch.float32);  mul_5 = None
        split_2 = torch.ops.aten.split.Tensor(convert_element_type_default_20, 64, -1);  convert_element_type_default_20 = None
        getitem_17: "f32[1, 6144, 6, 64]" = split_2[0]
        getitem_18: "f32[1, 6144, 6, 64]" = split_2[1];  split_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:2041 in flex_attention, code: out, lse, max_scores = flex_attention_hop(
        sdpa_score0 = self.sdpa_score0
        sdpa_mask0 = self.sdpa_mask0

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:637 in forward, code: self.cos[None, : x_BTHD.size(-3), None, :],
        unsqueeze_43: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg8_1, 0)
        slice_13: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_43, 1, 0, 6144);  unsqueeze_43 = None
        unsqueeze_44: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_13, 2);  slice_13 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_6: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_15, unsqueeze_44)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:638 in forward, code: self.sin[None, : x_BTHD.size(-3), None, :],
        unsqueeze_45: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg9_1, 0)
        slice_14: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_45, 1, 0, 6144);  unsqueeze_45 = None
        unsqueeze_46: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_14, 2);  slice_14 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_7: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_16, unsqueeze_46)
        add_5: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_6, mul_7);  mul_6 = mul_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        neg: "f32[1, 6144, 1, 64]" = torch.ops.aten.neg.default(unsqueeze_46);  unsqueeze_46 = None
        mul_8: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_15, neg);  getitem_15 = neg = None
        mul_9: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_16, unsqueeze_44);  getitem_16 = unsqueeze_44 = None
        add_6: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        cat: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_5, add_6], 3);  add_5 = add_6 = None
        convert_element_type_22: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat, torch.bfloat16);  cat = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:691 in forward, code: q.transpose(1, 2),
        permute_5: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(convert_element_type_22, [0, 2, 1, 3]);  convert_element_type_22 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:637 in forward, code: self.cos[None, : x_BTHD.size(-3), None, :],
        unsqueeze_47: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg8_1, 0);  arg8_1 = None
        slice_15: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_47, 1, 0, 6144);  unsqueeze_47 = None
        unsqueeze_48: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_15, 2);  slice_15 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_10: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_17, unsqueeze_48)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:638 in forward, code: self.sin[None, : x_BTHD.size(-3), None, :],
        unsqueeze_49: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg9_1, 0);  arg9_1 = None
        slice_16: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_49, 1, 0, 6144);  unsqueeze_49 = None
        unsqueeze_50: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_16, 2);  slice_16 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_11: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_18, unsqueeze_50)
        add_7: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_10, mul_11);  mul_10 = mul_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        neg_1: "f32[1, 6144, 1, 64]" = torch.ops.aten.neg.default(unsqueeze_50);  unsqueeze_50 = None
        mul_12: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_17, neg_1);  getitem_17 = neg_1 = None
        mul_13: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_18, unsqueeze_48);  getitem_18 = unsqueeze_48 = None
        add_8: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_12, mul_13);  mul_12 = mul_13 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        cat_1: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_7, add_8], 3);  add_7 = add_8 = None
        convert_element_type_24: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_1, torch.bfloat16);  cat_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:692 in forward, code: k.transpose(1, 2),
        permute_6: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(convert_element_type_24, [0, 2, 1, 3]);  convert_element_type_24 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:915 in forward, code: sa_lambdas = self.scalars[3 * len(self.blocks) : 5 * len(self.blocks)].view(
        slice_11: "f32[24]" = torch.ops.aten.slice.Tensor(arg6_1, 0, 36, 60)
        view_7: "f32[12, 2]" = torch.ops.aten.reshape.default(slice_11, [-1, 2]);  slice_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_3: "f32[2]" = torch.ops.aten.select.int(view_7, 0, 0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:685 in forward, code: v = lambdas[0] * v + lambdas[1] * ve.view_as(
        select_6: "f32[]" = torch.ops.aten.select.int(select_3, 0, 0)
        mul_14: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(select_6, getitem_14);  select_6 = getitem_14 = None
        select_7: "f32[]" = torch.ops.aten.select.int(select_3, 0, 1);  select_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:883 in forward, code: ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        embedding: "bf16[6144, 768]" = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg1_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:685 in forward, code: v = lambdas[0] * v + lambdas[1] * ve.view_as(
        view_12: "bf16[1, 6144, 6, 128]" = torch.ops.aten.reshape.default(embedding, [1, 6144, 6, 128])
        mul_15: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(select_7, view_12);  select_7 = view_12 = None
        add_9: "bf16[1, 6144, 6, 128]" = torch.ops.aten.add.Tensor(mul_14, mul_15);  mul_14 = mul_15 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:693 in forward, code: v.transpose(1, 2),
        permute_7: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(add_9, [0, 2, 1, 3]);  add_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:425 in _dense_to_ordered, code: num_blocks_in_row = dense_mask.sum(dim=-1)
        sum_3: "i64[1, 1, 48]" = torch.ops.aten.sum.dim_IntList(permute, [-1]);  permute = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:428 in _dense_to_ordered, code: num_blocks_in_row.to(torch.int32, memory_format=torch.contiguous_format),
        convert_element_type_2: "i32[1, 1, 48]" = torch.ops.prims.convert_element_type.default(sum_3, torch.int32);  sum_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:429 in _dense_to_ordered, code: col_indices.to(torch.int32, memory_format=torch.contiguous_format),
        convert_element_type_3: "i32[1, 1, 48, 48]" = torch.ops.prims.convert_element_type.default(getitem_5, torch.int32);  getitem_5 = None
        clone_4: "i32[1, 1, 48, 48]" = torch.ops.aten.clone.default(convert_element_type_3, memory_format = torch.contiguous_format);  convert_element_type_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:425 in _dense_to_ordered, code: num_blocks_in_row = dense_mask.sum(dim=-1)
        sum_4: "i64[1, 1, 48]" = torch.ops.aten.sum.dim_IntList(permute_1, [-1]);  permute_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:428 in _dense_to_ordered, code: num_blocks_in_row.to(torch.int32, memory_format=torch.contiguous_format),
        convert_element_type_4: "i32[1, 1, 48]" = torch.ops.prims.convert_element_type.default(sum_4, torch.int32);  sum_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:429 in _dense_to_ordered, code: col_indices.to(torch.int32, memory_format=torch.contiguous_format),
        convert_element_type_5: "i32[1, 1, 48, 48]" = torch.ops.prims.convert_element_type.default(getitem_7, torch.int32);  getitem_7 = None
        clone_7: "i32[1, 1, 48, 48]" = torch.ops.aten.clone.default(convert_element_type_5, memory_format = torch.contiguous_format);  convert_element_type_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:2041 in flex_attention, code: out, lse, max_scores = flex_attention_hop(
        flex_attention = torch.ops.higher_order.flex_attention(permute_5, permute_6, permute_7, sdpa_score0, (6144, 6144, clamp_max, unsqueeze_9, clamp_max_1, unsqueeze_13, convert_element_type_2, clone_4, convert_element_type_4, clone_7, 128, 128, sdpa_mask0), 0.12, {'BACKEND': 'AUTO', 'PRESCALE_QK': False, 'ROWS_GUARANTEED_SAFE': False, 'BLOCKS_ARE_CONTIGUOUS': False, 'WRITE_DQ': True, 'OUTPUT_LOGSUMEXP': False, 'OUTPUT_MAX': False}, (), (cumsum,));  permute_5 = permute_6 = permute_7 = sdpa_score0 = sdpa_mask0 = None
        getitem_19: "bf16[1, 6, 6144, 128]" = flex_attention[0];  flex_attention = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_9: "f32[2]" = torch.ops.aten.select.int(view_6, 0, 1)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        select_11: "f32[]" = torch.ops.aten.select.int(select_9, 0, 0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:696 in forward, code: ).transpose(1, 2)
        permute_8: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_19, [0, 2, 1, 3]);  getitem_19 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        slice_17: "bf16[1, 6144, 12]" = torch.ops.aten.slice.Tensor(convert_element_type_13, 2, 0, 12);  convert_element_type_13 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:618 in forward, code: return F.linear(x, self.weight.type_as(x))
        expand_20: "bf16[1, 6144, 12]" = torch.ops.aten.expand.default(slice_17, [1, 6144, 12]);  slice_17 = None
        squeeze_dim_20: "bf16[6144, 12]" = torch.ops.aten.squeeze.dim(expand_20, 0);  expand_20 = None
        convert_element_type_25: "bf16[6, 12]" = torch.ops.prims.convert_element_type.default(arg10_1, torch.bfloat16);  arg10_1 = None
        permute_9: "bf16[12, 6]" = torch.ops.aten.permute.default(convert_element_type_25, [1, 0]);  convert_element_type_25 = None
        expand_21: "bf16[1, 12, 6]" = torch.ops.aten.expand.default(permute_9, [1, 12, 6]);  permute_9 = None
        squeeze_dim_21: "bf16[12, 6]" = torch.ops.aten.squeeze.dim(expand_21, 0);  expand_21 = None
        mm_default_10: "bf16[6144, 6]" = torch.ops.aten.mm.default(squeeze_dim_20, squeeze_dim_21);  squeeze_dim_20 = squeeze_dim_21 = None
        unsqueeze_default_10: "bf16[1, 6144, 6]" = torch.ops.aten.unsqueeze.default(mm_default_10, 0);  mm_default_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        sigmoid: "bf16[1, 6144, 6]" = torch.ops.aten.sigmoid.default(unsqueeze_default_10);  unsqueeze_default_10 = None
        view_17: "bf16[1, 6144, 6, 1]" = torch.ops.aten.reshape.default(sigmoid, [1, 6144, 6, 1]);  sigmoid = None
        mul_16: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_8, view_17);  permute_8 = view_17 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:701 in forward, code: y = y.contiguous().view(
        view_18: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mul_16, [1, 6144, 768]);  mul_16 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:704 in forward, code: y = F.linear(y, self.qkvo_w[3].type_as(y))
        view_19: "bf16[6144, 768]" = torch.ops.aten.reshape.default(view_18, [6144, 768]);  view_18 = None
        select_8: "f32[768, 768]" = torch.ops.aten.select.int(arg7_1, 0, 3);  arg7_1 = None
        convert_element_type_28: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(select_8, torch.bfloat16);  select_8 = None
        permute_10: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_28, [1, 0]);  convert_element_type_28 = None
        mm_1: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_19, permute_10);  view_19 = permute_10 = None
        view_20: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_1, [1, 6144, 768]);  mm_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:751 in forward, code: x = x + self.attn(norm(x), ve, sa_lambdas, block_mask)
        add_10: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_1, view_20);  add_1 = view_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_31: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_10, torch.float32)
        pow_5: "f32[1, 6144, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_31, 2)
        mean_4: "f32[1, 6144, 1]" = torch.ops.aten.mean.dim(pow_5, [2], True);  pow_5 = None
        add_11: "f32[1, 6144, 1]" = torch.ops.aten.add.Scalar(mean_4, 1.1920928955078125e-07);  mean_4 = None
        rsqrt_4: "f32[1, 6144, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        mul_17: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_31, rsqrt_4);  convert_element_type_31 = rsqrt_4 = None
        convert_element_type_32: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_17, torch.bfloat16);  mul_17 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:723 in forward, code: x = F.linear(x, self.c_fc.T.type_as(x))
        view_21: "bf16[6144, 768]" = torch.ops.aten.reshape.default(convert_element_type_32, [6144, 768]);  convert_element_type_32 = None
        permute_11: "f32[3072, 768]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        convert_element_type_33: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(permute_11, torch.bfloat16);  permute_11 = None
        permute_12: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_33, [1, 0]);  convert_element_type_33 = None
        mm_2: "bf16[6144, 3072]" = torch.ops.aten.mm.default(view_21, permute_12);  view_21 = permute_12 = None
        view_22: "bf16[1, 6144, 3072]" = torch.ops.aten.reshape.default(mm_2, [1, 6144, 3072]);  mm_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:724 in forward, code: x = F.relu(
        relu: "bf16[1, 6144, 3072]" = torch.ops.aten.relu.default(view_22);  view_22 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:726 in forward, code: ).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        pow_6: "bf16[1, 6144, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu, 2);  relu = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:727 in forward, code: x = F.linear(x, self.c_proj.type_as(x))
        view_23: "bf16[6144, 3072]" = torch.ops.aten.reshape.default(pow_6, [6144, 3072]);  pow_6 = None
        convert_element_type_36: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg12_1, torch.bfloat16);  arg12_1 = None
        permute_13: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_36, [1, 0]);  convert_element_type_36 = None
        mm_3: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_23, permute_13);  view_23 = permute_13 = None
        view_24: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_3, [1, 6144, 768]);  mm_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:752 in forward, code: x = x + self.mlp(norm(x))
        add_12: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_10, view_24);  add_10 = view_24 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        mul_18: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(select_11, add_12);  select_11 = None
        select_12: "f32[]" = torch.ops.aten.select.int(select_9, 0, 1);  select_9 = None
        mul_19: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(select_12, convert_element_type_11);  select_12 = None
        add_13: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_39: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_13, torch.float32)
        pow_7: "f32[1, 6144, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_39, 2)
        mean_5: "f32[1, 6144, 1]" = torch.ops.aten.mean.dim(pow_7, [2], True);  pow_7 = None
        add_14: "f32[1, 6144, 1]" = torch.ops.aten.add.Scalar(mean_5, 1.1920928955078125e-07);  mean_5 = None
        rsqrt_5: "f32[1, 6144, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        mul_20: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_39, rsqrt_5);  convert_element_type_39 = rsqrt_5 = None
        convert_element_type_40: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_20, torch.bfloat16);  mul_20 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        view_26: "bf16[6144, 768]" = torch.ops.aten.reshape.default(convert_element_type_40, [6144, 768])
        slice_18: "f32[3, 768, 768]" = torch.ops.aten.slice.Tensor(arg13_1, 0, 0, 3)
        view_25: "f32[2304, 768]" = torch.ops.aten.reshape.default(slice_18, [2304, 768]);  slice_18 = None
        convert_element_type_41: "bf16[2304, 768]" = torch.ops.prims.convert_element_type.default(view_25, torch.bfloat16);  view_25 = None
        permute_14: "bf16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_41, [1, 0]);  convert_element_type_41 = None
        mm_4: "bf16[6144, 2304]" = torch.ops.aten.mm.default(view_26, permute_14);  view_26 = permute_14 = None
        view_27: "bf16[1, 6144, 2304]" = torch.ops.aten.reshape.default(mm_4, [1, 6144, 2304]);  mm_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:679 in forward, code: .view(B, T, 3 * self.num_heads, self.head_dim)
        view_28: "bf16[1, 6144, 18, 128]" = torch.ops.aten.reshape.default(view_27, [1, 6144, 18, 128]);  view_27 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:680 in forward, code: .chunk(3, dim=-2)
        split_3 = torch.ops.aten.split.Tensor(view_28, 6, -2);  view_28 = None
        getitem_22: "bf16[1, 6144, 6, 128]" = split_3[0]
        getitem_23: "bf16[1, 6144, 6, 128]" = split_3[1]
        getitem_24: "bf16[1, 6144, 6, 128]" = split_3[2];  split_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_44: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_22, torch.float32);  getitem_22 = None
        pow_8: "f32[1, 6144, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_44, 2)
        mean_6: "f32[1, 6144, 6, 1]" = torch.ops.aten.mean.dim(pow_8, [3], True);  pow_8 = None
        add_15: "f32[1, 6144, 6, 1]" = torch.ops.aten.add.Scalar(mean_6, 1.1920928955078125e-07);  mean_6 = None
        rsqrt_6: "f32[1, 6144, 6, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
        mul_21: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_44, rsqrt_6);  convert_element_type_44 = rsqrt_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        convert_element_type_default_19: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_21, torch.float32);  mul_21 = None
        split_4 = torch.ops.aten.split.Tensor(convert_element_type_default_19, 64, -1);  convert_element_type_default_19 = None
        getitem_25: "f32[1, 6144, 6, 64]" = split_4[0]
        getitem_26: "f32[1, 6144, 6, 64]" = split_4[1];  split_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_46: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_23, torch.float32);  getitem_23 = None
        pow_9: "f32[1, 6144, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_46, 2)
        mean_7: "f32[1, 6144, 6, 1]" = torch.ops.aten.mean.dim(pow_9, [3], True);  pow_9 = None
        add_16: "f32[1, 6144, 6, 1]" = torch.ops.aten.add.Scalar(mean_7, 1.1920928955078125e-07);  mean_7 = None
        rsqrt_7: "f32[1, 6144, 6, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        mul_22: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_46, rsqrt_7);  convert_element_type_46 = rsqrt_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        convert_element_type_default_18: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_22, torch.float32);  mul_22 = None
        split_5 = torch.ops.aten.split.Tensor(convert_element_type_default_18, 64, -1);  convert_element_type_default_18 = None
        getitem_27: "f32[1, 6144, 6, 64]" = split_5[0]
        getitem_28: "f32[1, 6144, 6, 64]" = split_5[1];  split_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:2041 in flex_attention, code: out, lse, max_scores = flex_attention_hop(
        sdpa_score1 = self.sdpa_score1
        sdpa_mask1 = self.sdpa_mask1

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:637 in forward, code: self.cos[None, : x_BTHD.size(-3), None, :],
        unsqueeze_51: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg14_1, 0)
        slice_19: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_51, 1, 0, 6144);  unsqueeze_51 = None
        unsqueeze_52: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_19, 2);  slice_19 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_23: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_25, unsqueeze_52)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:638 in forward, code: self.sin[None, : x_BTHD.size(-3), None, :],
        unsqueeze_53: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg15_1, 0)
        slice_20: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_53, 1, 0, 6144);  unsqueeze_53 = None
        unsqueeze_54: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_20, 2);  slice_20 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_24: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_26, unsqueeze_54)
        add_17: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_23, mul_24);  mul_23 = mul_24 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        neg_2: "f32[1, 6144, 1, 64]" = torch.ops.aten.neg.default(unsqueeze_54);  unsqueeze_54 = None
        mul_25: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_25, neg_2);  getitem_25 = neg_2 = None
        mul_26: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_26, unsqueeze_52);  getitem_26 = unsqueeze_52 = None
        add_18: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        cat_2: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_17, add_18], 3);  add_17 = add_18 = None
        convert_element_type_49: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_2, torch.bfloat16);  cat_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:691 in forward, code: q.transpose(1, 2),
        permute_15: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(convert_element_type_49, [0, 2, 1, 3]);  convert_element_type_49 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:637 in forward, code: self.cos[None, : x_BTHD.size(-3), None, :],
        unsqueeze_55: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg14_1, 0);  arg14_1 = None
        slice_21: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_55, 1, 0, 6144);  unsqueeze_55 = None
        unsqueeze_56: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_21, 2);  slice_21 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_27: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_27, unsqueeze_56)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:638 in forward, code: self.sin[None, : x_BTHD.size(-3), None, :],
        unsqueeze_57: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg15_1, 0);  arg15_1 = None
        slice_22: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_57, 1, 0, 6144);  unsqueeze_57 = None
        unsqueeze_58: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_22, 2);  slice_22 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_28: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_28, unsqueeze_58)
        add_19: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_27, mul_28);  mul_27 = mul_28 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        neg_3: "f32[1, 6144, 1, 64]" = torch.ops.aten.neg.default(unsqueeze_58);  unsqueeze_58 = None
        mul_29: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_27, neg_3);  getitem_27 = neg_3 = None
        mul_30: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_28, unsqueeze_56);  getitem_28 = unsqueeze_56 = None
        add_20: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_29, mul_30);  mul_29 = mul_30 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        cat_3: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_19, add_20], 3);  add_19 = add_20 = None
        convert_element_type_51: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_3, torch.bfloat16);  cat_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:692 in forward, code: k.transpose(1, 2),
        permute_16: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(convert_element_type_51, [0, 2, 1, 3]);  convert_element_type_51 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_10: "f32[2]" = torch.ops.aten.select.int(view_7, 0, 1)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:685 in forward, code: v = lambdas[0] * v + lambdas[1] * ve.view_as(
        select_13: "f32[]" = torch.ops.aten.select.int(select_10, 0, 0)
        mul_31: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(select_13, getitem_24);  select_13 = getitem_24 = None
        select_14: "f32[]" = torch.ops.aten.select.int(select_10, 0, 1);  select_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:883 in forward, code: ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        embedding_1: "bf16[6144, 768]" = torch.ops.aten.embedding.default(arg2_1, arg0_1);  arg2_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:685 in forward, code: v = lambdas[0] * v + lambdas[1] * ve.view_as(
        view_29: "bf16[1, 6144, 6, 128]" = torch.ops.aten.reshape.default(embedding_1, [1, 6144, 6, 128])
        mul_32: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(select_14, view_29);  select_14 = view_29 = None
        add_21: "bf16[1, 6144, 6, 128]" = torch.ops.aten.add.Tensor(mul_31, mul_32);  mul_31 = mul_32 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:693 in forward, code: v.transpose(1, 2),
        permute_17: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(add_21, [0, 2, 1, 3]);  add_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:425 in _dense_to_ordered, code: num_blocks_in_row = dense_mask.sum(dim=-1)
        sum_5: "i64[1, 1, 48]" = torch.ops.aten.sum.dim_IntList(permute_2, [-1]);  permute_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:428 in _dense_to_ordered, code: num_blocks_in_row.to(torch.int32, memory_format=torch.contiguous_format),
        convert_element_type_6: "i32[1, 1, 48]" = torch.ops.prims.convert_element_type.default(sum_5, torch.int32);  sum_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:429 in _dense_to_ordered, code: col_indices.to(torch.int32, memory_format=torch.contiguous_format),
        convert_element_type_7: "i32[1, 1, 48, 48]" = torch.ops.prims.convert_element_type.default(getitem_9, torch.int32);  getitem_9 = None
        clone_10: "i32[1, 1, 48, 48]" = torch.ops.aten.clone.default(convert_element_type_7, memory_format = torch.contiguous_format);  convert_element_type_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:425 in _dense_to_ordered, code: num_blocks_in_row = dense_mask.sum(dim=-1)
        sum_6: "i64[1, 1, 48]" = torch.ops.aten.sum.dim_IntList(permute_3, [-1]);  permute_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:428 in _dense_to_ordered, code: num_blocks_in_row.to(torch.int32, memory_format=torch.contiguous_format),
        convert_element_type_8: "i32[1, 1, 48]" = torch.ops.prims.convert_element_type.default(sum_6, torch.int32);  sum_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:429 in _dense_to_ordered, code: col_indices.to(torch.int32, memory_format=torch.contiguous_format),
        convert_element_type_9: "i32[1, 1, 48, 48]" = torch.ops.prims.convert_element_type.default(getitem_11, torch.int32);  getitem_11 = None
        clone_13: "i32[1, 1, 48, 48]" = torch.ops.aten.clone.default(convert_element_type_9, memory_format = torch.contiguous_format);  convert_element_type_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:2041 in flex_attention, code: out, lse, max_scores = flex_attention_hop(
        flex_attention_1 = torch.ops.higher_order.flex_attention(permute_15, permute_16, permute_17, sdpa_score1, (6144, 6144, clamp_max_2, unsqueeze_9, clamp_max_3, unsqueeze_13, convert_element_type_6, clone_10, convert_element_type_8, clone_13, 128, 128, sdpa_mask1), 0.12, {'BACKEND': 'AUTO', 'PRESCALE_QK': False, 'ROWS_GUARANTEED_SAFE': False, 'BLOCKS_ARE_CONTIGUOUS': False, 'WRITE_DQ': True, 'OUTPUT_LOGSUMEXP': False, 'OUTPUT_MAX': False}, (), (cumsum,));  permute_15 = permute_16 = permute_17 = sdpa_score1 = sdpa_mask1 = None
        getitem_29: "bf16[1, 6, 6144, 128]" = flex_attention_1[0];  flex_attention_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_16: "f32[2]" = torch.ops.aten.select.int(view_6, 0, 2)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        select_18: "f32[]" = torch.ops.aten.select.int(select_16, 0, 0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:696 in forward, code: ).transpose(1, 2)
        permute_18: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_29, [0, 2, 1, 3]);  getitem_29 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        slice_23: "bf16[1, 6144, 12]" = torch.ops.aten.slice.Tensor(convert_element_type_40, 2, 0, 12);  convert_element_type_40 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:618 in forward, code: return F.linear(x, self.weight.type_as(x))
        expand_22: "bf16[1, 6144, 12]" = torch.ops.aten.expand.default(slice_23, [1, 6144, 12]);  slice_23 = None
        squeeze_dim_18: "bf16[6144, 12]" = torch.ops.aten.squeeze.dim(expand_22, 0);  expand_22 = None
        convert_element_type_52: "bf16[6, 12]" = torch.ops.prims.convert_element_type.default(arg16_1, torch.bfloat16);  arg16_1 = None
        permute_19: "bf16[12, 6]" = torch.ops.aten.permute.default(convert_element_type_52, [1, 0]);  convert_element_type_52 = None
        expand_23: "bf16[1, 12, 6]" = torch.ops.aten.expand.default(permute_19, [1, 12, 6]);  permute_19 = None
        squeeze_dim_19: "bf16[12, 6]" = torch.ops.aten.squeeze.dim(expand_23, 0);  expand_23 = None
        mm_default_9: "bf16[6144, 6]" = torch.ops.aten.mm.default(squeeze_dim_18, squeeze_dim_19);  squeeze_dim_18 = squeeze_dim_19 = None
        unsqueeze_default_9: "bf16[1, 6144, 6]" = torch.ops.aten.unsqueeze.default(mm_default_9, 0);  mm_default_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        sigmoid_1: "bf16[1, 6144, 6]" = torch.ops.aten.sigmoid.default(unsqueeze_default_9);  unsqueeze_default_9 = None
        view_34: "bf16[1, 6144, 6, 1]" = torch.ops.aten.reshape.default(sigmoid_1, [1, 6144, 6, 1]);  sigmoid_1 = None
        mul_33: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_18, view_34);  permute_18 = view_34 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:701 in forward, code: y = y.contiguous().view(
        view_35: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mul_33, [1, 6144, 768]);  mul_33 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:704 in forward, code: y = F.linear(y, self.qkvo_w[3].type_as(y))
        view_36: "bf16[6144, 768]" = torch.ops.aten.reshape.default(view_35, [6144, 768]);  view_35 = None
        select_15: "f32[768, 768]" = torch.ops.aten.select.int(arg13_1, 0, 3);  arg13_1 = None
        convert_element_type_55: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(select_15, torch.bfloat16);  select_15 = None
        permute_20: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_55, [1, 0]);  convert_element_type_55 = None
        mm_5: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_36, permute_20);  view_36 = permute_20 = None
        view_37: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_5, [1, 6144, 768]);  mm_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:751 in forward, code: x = x + self.attn(norm(x), ve, sa_lambdas, block_mask)
        add_22: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_13, view_37);  add_13 = view_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_58: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_22, torch.float32)
        pow_10: "f32[1, 6144, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_58, 2)
        mean_8: "f32[1, 6144, 1]" = torch.ops.aten.mean.dim(pow_10, [2], True);  pow_10 = None
        add_23: "f32[1, 6144, 1]" = torch.ops.aten.add.Scalar(mean_8, 1.1920928955078125e-07);  mean_8 = None
        rsqrt_8: "f32[1, 6144, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        mul_34: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_58, rsqrt_8);  convert_element_type_58 = rsqrt_8 = None
        convert_element_type_59: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_34, torch.bfloat16);  mul_34 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:723 in forward, code: x = F.linear(x, self.c_fc.T.type_as(x))
        view_38: "bf16[6144, 768]" = torch.ops.aten.reshape.default(convert_element_type_59, [6144, 768]);  convert_element_type_59 = None
        permute_21: "f32[3072, 768]" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        convert_element_type_60: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(permute_21, torch.bfloat16);  permute_21 = None
        permute_22: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_60, [1, 0]);  convert_element_type_60 = None
        mm_6: "bf16[6144, 3072]" = torch.ops.aten.mm.default(view_38, permute_22);  view_38 = permute_22 = None
        view_39: "bf16[1, 6144, 3072]" = torch.ops.aten.reshape.default(mm_6, [1, 6144, 3072]);  mm_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:724 in forward, code: x = F.relu(
        relu_1: "bf16[1, 6144, 3072]" = torch.ops.aten.relu.default(view_39);  view_39 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:726 in forward, code: ).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        pow_11: "bf16[1, 6144, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu_1, 2);  relu_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:727 in forward, code: x = F.linear(x, self.c_proj.type_as(x))
        view_40: "bf16[6144, 3072]" = torch.ops.aten.reshape.default(pow_11, [6144, 3072]);  pow_11 = None
        convert_element_type_63: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg18_1, torch.bfloat16);  arg18_1 = None
        permute_23: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_63, [1, 0]);  convert_element_type_63 = None
        mm_7: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_40, permute_23);  view_40 = permute_23 = None
        view_41: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_7, [1, 6144, 768]);  mm_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:752 in forward, code: x = x + self.mlp(norm(x))
        add_24: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_22, view_41);  add_22 = view_41 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        mul_35: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(select_18, add_24);  select_18 = None
        select_19: "f32[]" = torch.ops.aten.select.int(select_16, 0, 1);  select_16 = None
        mul_36: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(select_19, convert_element_type_11);  select_19 = None
        add_25: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(mul_35, mul_36);  mul_35 = mul_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_66: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_25, torch.float32)
        pow_12: "f32[1, 6144, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_66, 2)
        mean_9: "f32[1, 6144, 1]" = torch.ops.aten.mean.dim(pow_12, [2], True);  pow_12 = None
        add_26: "f32[1, 6144, 1]" = torch.ops.aten.add.Scalar(mean_9, 1.1920928955078125e-07);  mean_9 = None
        rsqrt_9: "f32[1, 6144, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        mul_37: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_66, rsqrt_9);  convert_element_type_66 = rsqrt_9 = None
        convert_element_type_67: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_37, torch.bfloat16);  mul_37 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        view_43: "bf16[6144, 768]" = torch.ops.aten.reshape.default(convert_element_type_67, [6144, 768])
        slice_24: "f32[3, 768, 768]" = torch.ops.aten.slice.Tensor(arg19_1, 0, 0, 3)
        view_42: "f32[2304, 768]" = torch.ops.aten.reshape.default(slice_24, [2304, 768]);  slice_24 = None
        convert_element_type_68: "bf16[2304, 768]" = torch.ops.prims.convert_element_type.default(view_42, torch.bfloat16);  view_42 = None
        permute_24: "bf16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_68, [1, 0]);  convert_element_type_68 = None
        mm_8: "bf16[6144, 2304]" = torch.ops.aten.mm.default(view_43, permute_24);  view_43 = permute_24 = None
        view_44: "bf16[1, 6144, 2304]" = torch.ops.aten.reshape.default(mm_8, [1, 6144, 2304]);  mm_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:679 in forward, code: .view(B, T, 3 * self.num_heads, self.head_dim)
        view_45: "bf16[1, 6144, 18, 128]" = torch.ops.aten.reshape.default(view_44, [1, 6144, 18, 128]);  view_44 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:680 in forward, code: .chunk(3, dim=-2)
        split_6 = torch.ops.aten.split.Tensor(view_45, 6, -2);  view_45 = None
        getitem_32: "bf16[1, 6144, 6, 128]" = split_6[0]
        getitem_33: "bf16[1, 6144, 6, 128]" = split_6[1]
        getitem_34: "bf16[1, 6144, 6, 128]" = split_6[2];  split_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_71: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_32, torch.float32);  getitem_32 = None
        pow_13: "f32[1, 6144, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_71, 2)
        mean_10: "f32[1, 6144, 6, 1]" = torch.ops.aten.mean.dim(pow_13, [3], True);  pow_13 = None
        add_27: "f32[1, 6144, 6, 1]" = torch.ops.aten.add.Scalar(mean_10, 1.1920928955078125e-07);  mean_10 = None
        rsqrt_10: "f32[1, 6144, 6, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        mul_38: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_71, rsqrt_10);  convert_element_type_71 = rsqrt_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        convert_element_type_default_17: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_38, torch.float32);  mul_38 = None
        split_7 = torch.ops.aten.split.Tensor(convert_element_type_default_17, 64, -1);  convert_element_type_default_17 = None
        getitem_35: "f32[1, 6144, 6, 64]" = split_7[0]
        getitem_36: "f32[1, 6144, 6, 64]" = split_7[1];  split_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_73: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_33, torch.float32);  getitem_33 = None
        pow_14: "f32[1, 6144, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_73, 2)
        mean_11: "f32[1, 6144, 6, 1]" = torch.ops.aten.mean.dim(pow_14, [3], True);  pow_14 = None
        add_28: "f32[1, 6144, 6, 1]" = torch.ops.aten.add.Scalar(mean_11, 1.1920928955078125e-07);  mean_11 = None
        rsqrt_11: "f32[1, 6144, 6, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        mul_39: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_73, rsqrt_11);  convert_element_type_73 = rsqrt_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        convert_element_type_default_16: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_39, torch.float32);  mul_39 = None
        split_8 = torch.ops.aten.split.Tensor(convert_element_type_default_16, 64, -1);  convert_element_type_default_16 = None
        getitem_37: "f32[1, 6144, 6, 64]" = split_8[0]
        getitem_38: "f32[1, 6144, 6, 64]" = split_8[1];  split_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:2041 in flex_attention, code: out, lse, max_scores = flex_attention_hop(
        sdpa_score2 = self.sdpa_score2
        sdpa_mask2 = self.sdpa_mask2

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:637 in forward, code: self.cos[None, : x_BTHD.size(-3), None, :],
        unsqueeze_59: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg20_1, 0)
        slice_25: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_59, 1, 0, 6144);  unsqueeze_59 = None
        unsqueeze_60: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_25, 2);  slice_25 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_40: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_35, unsqueeze_60)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:638 in forward, code: self.sin[None, : x_BTHD.size(-3), None, :],
        unsqueeze_61: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg21_1, 0)
        slice_26: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_61, 1, 0, 6144);  unsqueeze_61 = None
        unsqueeze_62: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_26, 2);  slice_26 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_41: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_36, unsqueeze_62)
        add_29: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_40, mul_41);  mul_40 = mul_41 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        neg_4: "f32[1, 6144, 1, 64]" = torch.ops.aten.neg.default(unsqueeze_62);  unsqueeze_62 = None
        mul_42: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_35, neg_4);  getitem_35 = neg_4 = None
        mul_43: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_36, unsqueeze_60);  getitem_36 = unsqueeze_60 = None
        add_30: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_42, mul_43);  mul_42 = mul_43 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        cat_4: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_29, add_30], 3);  add_29 = add_30 = None
        convert_element_type_76: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_4, torch.bfloat16);  cat_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:691 in forward, code: q.transpose(1, 2),
        permute_25: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(convert_element_type_76, [0, 2, 1, 3]);  convert_element_type_76 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:637 in forward, code: self.cos[None, : x_BTHD.size(-3), None, :],
        unsqueeze_63: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg20_1, 0);  arg20_1 = None
        slice_27: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_63, 1, 0, 6144);  unsqueeze_63 = None
        unsqueeze_64: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_27, 2);  slice_27 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_44: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_37, unsqueeze_64)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:638 in forward, code: self.sin[None, : x_BTHD.size(-3), None, :],
        unsqueeze_65: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg21_1, 0);  arg21_1 = None
        slice_28: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_65, 1, 0, 6144);  unsqueeze_65 = None
        unsqueeze_66: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_28, 2);  slice_28 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_45: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_38, unsqueeze_66)
        add_31: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_44, mul_45);  mul_44 = mul_45 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        neg_5: "f32[1, 6144, 1, 64]" = torch.ops.aten.neg.default(unsqueeze_66);  unsqueeze_66 = None
        mul_46: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_37, neg_5);  getitem_37 = neg_5 = None
        mul_47: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_38, unsqueeze_64);  getitem_38 = unsqueeze_64 = None
        add_32: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_46, mul_47);  mul_46 = mul_47 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        cat_5: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_31, add_32], 3);  add_31 = add_32 = None
        convert_element_type_78: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_5, torch.bfloat16);  cat_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:692 in forward, code: k.transpose(1, 2),
        permute_26: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(convert_element_type_78, [0, 2, 1, 3]);  convert_element_type_78 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_17: "f32[2]" = torch.ops.aten.select.int(view_7, 0, 2)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:685 in forward, code: v = lambdas[0] * v + lambdas[1] * ve.view_as(
        select_20: "f32[]" = torch.ops.aten.select.int(select_17, 0, 0)
        mul_48: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(select_20, getitem_34);  select_20 = getitem_34 = None
        select_21: "f32[]" = torch.ops.aten.select.int(select_17, 0, 1);  select_17 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:883 in forward, code: ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        embedding_2: "bf16[6144, 768]" = torch.ops.aten.embedding.default(arg3_1, arg0_1);  arg3_1 = arg0_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:685 in forward, code: v = lambdas[0] * v + lambdas[1] * ve.view_as(
        view_46: "bf16[1, 6144, 6, 128]" = torch.ops.aten.reshape.default(embedding_2, [1, 6144, 6, 128])
        mul_49: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(select_21, view_46);  select_21 = view_46 = None
        add_33: "bf16[1, 6144, 6, 128]" = torch.ops.aten.add.Tensor(mul_48, mul_49);  mul_48 = mul_49 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:693 in forward, code: v.transpose(1, 2),
        permute_27: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(add_33, [0, 2, 1, 3]);  add_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:2041 in flex_attention, code: out, lse, max_scores = flex_attention_hop(
        flex_attention_2 = torch.ops.higher_order.flex_attention(permute_25, permute_26, permute_27, sdpa_score2, (6144, 6144, clamp_max_2, unsqueeze_9, clamp_max_3, unsqueeze_13, convert_element_type_6, clone_10, convert_element_type_8, clone_13, 128, 128, sdpa_mask2), 0.12, {'BACKEND': 'AUTO', 'PRESCALE_QK': False, 'ROWS_GUARANTEED_SAFE': False, 'BLOCKS_ARE_CONTIGUOUS': False, 'WRITE_DQ': True, 'OUTPUT_LOGSUMEXP': False, 'OUTPUT_MAX': False}, (), (cumsum,));  permute_25 = permute_26 = permute_27 = sdpa_score2 = sdpa_mask2 = None
        getitem_39: "bf16[1, 6, 6144, 128]" = flex_attention_2[0];  flex_attention_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_23: "f32[2]" = torch.ops.aten.select.int(view_6, 0, 3)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        select_25: "f32[]" = torch.ops.aten.select.int(select_23, 0, 0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:696 in forward, code: ).transpose(1, 2)
        permute_28: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_39, [0, 2, 1, 3]);  getitem_39 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        slice_29: "bf16[1, 6144, 12]" = torch.ops.aten.slice.Tensor(convert_element_type_67, 2, 0, 12);  convert_element_type_67 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:618 in forward, code: return F.linear(x, self.weight.type_as(x))
        expand_24: "bf16[1, 6144, 12]" = torch.ops.aten.expand.default(slice_29, [1, 6144, 12]);  slice_29 = None
        squeeze_dim_16: "bf16[6144, 12]" = torch.ops.aten.squeeze.dim(expand_24, 0);  expand_24 = None
        convert_element_type_79: "bf16[6, 12]" = torch.ops.prims.convert_element_type.default(arg22_1, torch.bfloat16);  arg22_1 = None
        permute_29: "bf16[12, 6]" = torch.ops.aten.permute.default(convert_element_type_79, [1, 0]);  convert_element_type_79 = None
        expand_25: "bf16[1, 12, 6]" = torch.ops.aten.expand.default(permute_29, [1, 12, 6]);  permute_29 = None
        squeeze_dim_17: "bf16[12, 6]" = torch.ops.aten.squeeze.dim(expand_25, 0);  expand_25 = None
        mm_default_8: "bf16[6144, 6]" = torch.ops.aten.mm.default(squeeze_dim_16, squeeze_dim_17);  squeeze_dim_16 = squeeze_dim_17 = None
        unsqueeze_default_8: "bf16[1, 6144, 6]" = torch.ops.aten.unsqueeze.default(mm_default_8, 0);  mm_default_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        sigmoid_2: "bf16[1, 6144, 6]" = torch.ops.aten.sigmoid.default(unsqueeze_default_8);  unsqueeze_default_8 = None
        view_51: "bf16[1, 6144, 6, 1]" = torch.ops.aten.reshape.default(sigmoid_2, [1, 6144, 6, 1]);  sigmoid_2 = None
        mul_50: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_28, view_51);  permute_28 = view_51 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:701 in forward, code: y = y.contiguous().view(
        view_52: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mul_50, [1, 6144, 768]);  mul_50 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:704 in forward, code: y = F.linear(y, self.qkvo_w[3].type_as(y))
        view_53: "bf16[6144, 768]" = torch.ops.aten.reshape.default(view_52, [6144, 768]);  view_52 = None
        select_22: "f32[768, 768]" = torch.ops.aten.select.int(arg19_1, 0, 3);  arg19_1 = None
        convert_element_type_82: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(select_22, torch.bfloat16);  select_22 = None
        permute_30: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_82, [1, 0]);  convert_element_type_82 = None
        mm_9: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_53, permute_30);  view_53 = permute_30 = None
        view_54: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_9, [1, 6144, 768]);  mm_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:751 in forward, code: x = x + self.attn(norm(x), ve, sa_lambdas, block_mask)
        add_34: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_25, view_54);  add_25 = view_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_85: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_34, torch.float32)
        pow_15: "f32[1, 6144, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_85, 2)
        mean_12: "f32[1, 6144, 1]" = torch.ops.aten.mean.dim(pow_15, [2], True);  pow_15 = None
        add_35: "f32[1, 6144, 1]" = torch.ops.aten.add.Scalar(mean_12, 1.1920928955078125e-07);  mean_12 = None
        rsqrt_12: "f32[1, 6144, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
        mul_51: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_85, rsqrt_12);  convert_element_type_85 = rsqrt_12 = None
        convert_element_type_86: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_51, torch.bfloat16);  mul_51 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:723 in forward, code: x = F.linear(x, self.c_fc.T.type_as(x))
        view_55: "bf16[6144, 768]" = torch.ops.aten.reshape.default(convert_element_type_86, [6144, 768]);  convert_element_type_86 = None
        permute_31: "f32[3072, 768]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        convert_element_type_87: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(permute_31, torch.bfloat16);  permute_31 = None
        permute_32: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_87, [1, 0]);  convert_element_type_87 = None
        mm_10: "bf16[6144, 3072]" = torch.ops.aten.mm.default(view_55, permute_32);  view_55 = permute_32 = None
        view_56: "bf16[1, 6144, 3072]" = torch.ops.aten.reshape.default(mm_10, [1, 6144, 3072]);  mm_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:724 in forward, code: x = F.relu(
        relu_2: "bf16[1, 6144, 3072]" = torch.ops.aten.relu.default(view_56);  view_56 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:726 in forward, code: ).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        pow_16: "bf16[1, 6144, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu_2, 2);  relu_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:727 in forward, code: x = F.linear(x, self.c_proj.type_as(x))
        view_57: "bf16[6144, 3072]" = torch.ops.aten.reshape.default(pow_16, [6144, 3072]);  pow_16 = None
        convert_element_type_90: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg24_1, torch.bfloat16);  arg24_1 = None
        permute_33: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_90, [1, 0]);  convert_element_type_90 = None
        mm_11: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_57, permute_33);  view_57 = permute_33 = None
        view_58: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_11, [1, 6144, 768]);  mm_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:752 in forward, code: x = x + self.mlp(norm(x))
        add_36: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_34, view_58);  add_34 = view_58 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        mul_52: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(select_25, add_36);  select_25 = None
        select_26: "f32[]" = torch.ops.aten.select.int(select_23, 0, 1);  select_23 = None
        mul_53: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(select_26, convert_element_type_11);  select_26 = None
        add_37: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(mul_52, mul_53);  mul_52 = mul_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_93: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_37, torch.float32)
        pow_17: "f32[1, 6144, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_93, 2)
        mean_13: "f32[1, 6144, 1]" = torch.ops.aten.mean.dim(pow_17, [2], True);  pow_17 = None
        add_38: "f32[1, 6144, 1]" = torch.ops.aten.add.Scalar(mean_13, 1.1920928955078125e-07);  mean_13 = None
        rsqrt_13: "f32[1, 6144, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        mul_54: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_93, rsqrt_13);  convert_element_type_93 = rsqrt_13 = None
        convert_element_type_94: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_54, torch.bfloat16);  mul_54 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        view_60: "bf16[6144, 768]" = torch.ops.aten.reshape.default(convert_element_type_94, [6144, 768])
        slice_30: "f32[3, 768, 768]" = torch.ops.aten.slice.Tensor(arg25_1, 0, 0, 3)
        view_59: "f32[2304, 768]" = torch.ops.aten.reshape.default(slice_30, [2304, 768]);  slice_30 = None
        convert_element_type_95: "bf16[2304, 768]" = torch.ops.prims.convert_element_type.default(view_59, torch.bfloat16);  view_59 = None
        permute_34: "bf16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_95, [1, 0]);  convert_element_type_95 = None
        mm_12: "bf16[6144, 2304]" = torch.ops.aten.mm.default(view_60, permute_34);  view_60 = permute_34 = None
        view_61: "bf16[1, 6144, 2304]" = torch.ops.aten.reshape.default(mm_12, [1, 6144, 2304]);  mm_12 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:679 in forward, code: .view(B, T, 3 * self.num_heads, self.head_dim)
        view_62: "bf16[1, 6144, 18, 128]" = torch.ops.aten.reshape.default(view_61, [1, 6144, 18, 128]);  view_61 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:680 in forward, code: .chunk(3, dim=-2)
        split_9 = torch.ops.aten.split.Tensor(view_62, 6, -2);  view_62 = None
        getitem_42: "bf16[1, 6144, 6, 128]" = split_9[0]
        getitem_43: "bf16[1, 6144, 6, 128]" = split_9[1]
        getitem_44: "bf16[1, 6144, 6, 128]" = split_9[2];  split_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_98: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_42, torch.float32);  getitem_42 = None
        pow_18: "f32[1, 6144, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_98, 2)
        mean_14: "f32[1, 6144, 6, 1]" = torch.ops.aten.mean.dim(pow_18, [3], True);  pow_18 = None
        add_39: "f32[1, 6144, 6, 1]" = torch.ops.aten.add.Scalar(mean_14, 1.1920928955078125e-07);  mean_14 = None
        rsqrt_14: "f32[1, 6144, 6, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        mul_55: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_98, rsqrt_14);  convert_element_type_98 = rsqrt_14 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        convert_element_type_default_15: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_55, torch.float32);  mul_55 = None
        split_10 = torch.ops.aten.split.Tensor(convert_element_type_default_15, 64, -1);  convert_element_type_default_15 = None
        getitem_45: "f32[1, 6144, 6, 64]" = split_10[0]
        getitem_46: "f32[1, 6144, 6, 64]" = split_10[1];  split_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_100: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_43, torch.float32);  getitem_43 = None
        pow_19: "f32[1, 6144, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_100, 2)
        mean_15: "f32[1, 6144, 6, 1]" = torch.ops.aten.mean.dim(pow_19, [3], True);  pow_19 = None
        add_40: "f32[1, 6144, 6, 1]" = torch.ops.aten.add.Scalar(mean_15, 1.1920928955078125e-07);  mean_15 = None
        rsqrt_15: "f32[1, 6144, 6, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        mul_56: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_100, rsqrt_15);  convert_element_type_100 = rsqrt_15 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        convert_element_type_default_14: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_56, torch.float32);  mul_56 = None
        split_11 = torch.ops.aten.split.Tensor(convert_element_type_default_14, 64, -1);  convert_element_type_default_14 = None
        getitem_47: "f32[1, 6144, 6, 64]" = split_11[0]
        getitem_48: "f32[1, 6144, 6, 64]" = split_11[1];  split_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:2041 in flex_attention, code: out, lse, max_scores = flex_attention_hop(
        sdpa_score3 = self.sdpa_score3
        sdpa_mask3 = self.sdpa_mask3

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:637 in forward, code: self.cos[None, : x_BTHD.size(-3), None, :],
        unsqueeze_67: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg26_1, 0)
        slice_31: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_67, 1, 0, 6144);  unsqueeze_67 = None
        unsqueeze_68: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_31, 2);  slice_31 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_57: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_45, unsqueeze_68)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:638 in forward, code: self.sin[None, : x_BTHD.size(-3), None, :],
        unsqueeze_69: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg27_1, 0)
        slice_32: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_69, 1, 0, 6144);  unsqueeze_69 = None
        unsqueeze_70: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_32, 2);  slice_32 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_58: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_46, unsqueeze_70)
        add_41: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_57, mul_58);  mul_57 = mul_58 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        neg_6: "f32[1, 6144, 1, 64]" = torch.ops.aten.neg.default(unsqueeze_70);  unsqueeze_70 = None
        mul_59: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_45, neg_6);  getitem_45 = neg_6 = None
        mul_60: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_46, unsqueeze_68);  getitem_46 = unsqueeze_68 = None
        add_42: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_59, mul_60);  mul_59 = mul_60 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        cat_6: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_41, add_42], 3);  add_41 = add_42 = None
        convert_element_type_103: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_6, torch.bfloat16);  cat_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:691 in forward, code: q.transpose(1, 2),
        permute_35: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(convert_element_type_103, [0, 2, 1, 3]);  convert_element_type_103 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:637 in forward, code: self.cos[None, : x_BTHD.size(-3), None, :],
        unsqueeze_71: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg26_1, 0);  arg26_1 = None
        slice_33: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_71, 1, 0, 6144);  unsqueeze_71 = None
        unsqueeze_72: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_33, 2);  slice_33 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_61: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_47, unsqueeze_72)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:638 in forward, code: self.sin[None, : x_BTHD.size(-3), None, :],
        unsqueeze_73: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg27_1, 0);  arg27_1 = None
        slice_34: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_73, 1, 0, 6144);  unsqueeze_73 = None
        unsqueeze_74: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_34, 2);  slice_34 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_62: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_48, unsqueeze_74)
        add_43: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_61, mul_62);  mul_61 = mul_62 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        neg_7: "f32[1, 6144, 1, 64]" = torch.ops.aten.neg.default(unsqueeze_74);  unsqueeze_74 = None
        mul_63: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_47, neg_7);  getitem_47 = neg_7 = None
        mul_64: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_48, unsqueeze_72);  getitem_48 = unsqueeze_72 = None
        add_44: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_63, mul_64);  mul_63 = mul_64 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        cat_7: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_43, add_44], 3);  add_43 = add_44 = None
        convert_element_type_105: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_7, torch.bfloat16);  cat_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:692 in forward, code: k.transpose(1, 2),
        permute_36: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(convert_element_type_105, [0, 2, 1, 3]);  convert_element_type_105 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_24: "f32[2]" = torch.ops.aten.select.int(view_7, 0, 3)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:689 in forward, code: v = lambdas[0] * v
        select_27: "f32[]" = torch.ops.aten.select.int(select_24, 0, 0);  select_24 = None
        mul_65: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(select_27, getitem_44);  select_27 = getitem_44 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:693 in forward, code: v.transpose(1, 2),
        permute_37: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(mul_65, [0, 2, 1, 3]);  mul_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:2041 in flex_attention, code: out, lse, max_scores = flex_attention_hop(
        flex_attention_3 = torch.ops.higher_order.flex_attention(permute_35, permute_36, permute_37, sdpa_score3, (6144, 6144, clamp_max_2, unsqueeze_9, clamp_max_3, unsqueeze_13, convert_element_type_6, clone_10, convert_element_type_8, clone_13, 128, 128, sdpa_mask3), 0.12, {'BACKEND': 'AUTO', 'PRESCALE_QK': False, 'ROWS_GUARANTEED_SAFE': False, 'BLOCKS_ARE_CONTIGUOUS': False, 'WRITE_DQ': True, 'OUTPUT_LOGSUMEXP': False, 'OUTPUT_MAX': False}, (), (cumsum,));  permute_35 = permute_36 = permute_37 = sdpa_score3 = sdpa_mask3 = None
        getitem_49: "bf16[1, 6, 6144, 128]" = flex_attention_3[0];  flex_attention_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_29: "f32[2]" = torch.ops.aten.select.int(view_6, 0, 4)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        select_31: "f32[]" = torch.ops.aten.select.int(select_29, 0, 0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:696 in forward, code: ).transpose(1, 2)
        permute_38: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_49, [0, 2, 1, 3]);  getitem_49 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        slice_35: "bf16[1, 6144, 12]" = torch.ops.aten.slice.Tensor(convert_element_type_94, 2, 0, 12);  convert_element_type_94 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:618 in forward, code: return F.linear(x, self.weight.type_as(x))
        expand_26: "bf16[1, 6144, 12]" = torch.ops.aten.expand.default(slice_35, [1, 6144, 12]);  slice_35 = None
        squeeze_dim_14: "bf16[6144, 12]" = torch.ops.aten.squeeze.dim(expand_26, 0);  expand_26 = None
        convert_element_type_106: "bf16[6, 12]" = torch.ops.prims.convert_element_type.default(arg28_1, torch.bfloat16);  arg28_1 = None
        permute_39: "bf16[12, 6]" = torch.ops.aten.permute.default(convert_element_type_106, [1, 0]);  convert_element_type_106 = None
        expand_27: "bf16[1, 12, 6]" = torch.ops.aten.expand.default(permute_39, [1, 12, 6]);  permute_39 = None
        squeeze_dim_15: "bf16[12, 6]" = torch.ops.aten.squeeze.dim(expand_27, 0);  expand_27 = None
        mm_default_7: "bf16[6144, 6]" = torch.ops.aten.mm.default(squeeze_dim_14, squeeze_dim_15);  squeeze_dim_14 = squeeze_dim_15 = None
        unsqueeze_default_7: "bf16[1, 6144, 6]" = torch.ops.aten.unsqueeze.default(mm_default_7, 0);  mm_default_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        sigmoid_3: "bf16[1, 6144, 6]" = torch.ops.aten.sigmoid.default(unsqueeze_default_7);  unsqueeze_default_7 = None
        view_67: "bf16[1, 6144, 6, 1]" = torch.ops.aten.reshape.default(sigmoid_3, [1, 6144, 6, 1]);  sigmoid_3 = None
        mul_66: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_38, view_67);  permute_38 = view_67 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:701 in forward, code: y = y.contiguous().view(
        view_68: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mul_66, [1, 6144, 768]);  mul_66 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:704 in forward, code: y = F.linear(y, self.qkvo_w[3].type_as(y))
        view_69: "bf16[6144, 768]" = torch.ops.aten.reshape.default(view_68, [6144, 768]);  view_68 = None
        select_28: "f32[768, 768]" = torch.ops.aten.select.int(arg25_1, 0, 3);  arg25_1 = None
        convert_element_type_109: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(select_28, torch.bfloat16);  select_28 = None
        permute_40: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_109, [1, 0]);  convert_element_type_109 = None
        mm_13: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_69, permute_40);  view_69 = permute_40 = None
        view_70: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_13, [1, 6144, 768]);  mm_13 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:751 in forward, code: x = x + self.attn(norm(x), ve, sa_lambdas, block_mask)
        add_45: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_37, view_70);  add_37 = view_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_112: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_45, torch.float32)
        pow_20: "f32[1, 6144, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_112, 2)
        mean_16: "f32[1, 6144, 1]" = torch.ops.aten.mean.dim(pow_20, [2], True);  pow_20 = None
        add_46: "f32[1, 6144, 1]" = torch.ops.aten.add.Scalar(mean_16, 1.1920928955078125e-07);  mean_16 = None
        rsqrt_16: "f32[1, 6144, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        mul_67: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_112, rsqrt_16);  convert_element_type_112 = rsqrt_16 = None
        convert_element_type_113: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_67, torch.bfloat16);  mul_67 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:723 in forward, code: x = F.linear(x, self.c_fc.T.type_as(x))
        view_71: "bf16[6144, 768]" = torch.ops.aten.reshape.default(convert_element_type_113, [6144, 768]);  convert_element_type_113 = None
        permute_41: "f32[3072, 768]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        convert_element_type_114: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(permute_41, torch.bfloat16);  permute_41 = None
        permute_42: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_114, [1, 0]);  convert_element_type_114 = None
        mm_14: "bf16[6144, 3072]" = torch.ops.aten.mm.default(view_71, permute_42);  view_71 = permute_42 = None
        view_72: "bf16[1, 6144, 3072]" = torch.ops.aten.reshape.default(mm_14, [1, 6144, 3072]);  mm_14 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:724 in forward, code: x = F.relu(
        relu_3: "bf16[1, 6144, 3072]" = torch.ops.aten.relu.default(view_72);  view_72 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:726 in forward, code: ).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        pow_21: "bf16[1, 6144, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu_3, 2);  relu_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:727 in forward, code: x = F.linear(x, self.c_proj.type_as(x))
        view_73: "bf16[6144, 3072]" = torch.ops.aten.reshape.default(pow_21, [6144, 3072]);  pow_21 = None
        convert_element_type_117: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg30_1, torch.bfloat16);  arg30_1 = None
        permute_43: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_117, [1, 0]);  convert_element_type_117 = None
        mm_15: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_73, permute_43);  view_73 = permute_43 = None
        view_74: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_15, [1, 6144, 768]);  mm_15 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:752 in forward, code: x = x + self.mlp(norm(x))
        add_47: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_45, view_74);  add_45 = view_74 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        mul_68: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(select_31, add_47);  select_31 = None
        select_32: "f32[]" = torch.ops.aten.select.int(select_29, 0, 1);  select_29 = None
        mul_69: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(select_32, convert_element_type_11);  select_32 = None
        add_48: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(mul_68, mul_69);  mul_68 = mul_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_120: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_48, torch.float32)
        pow_22: "f32[1, 6144, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_120, 2)
        mean_17: "f32[1, 6144, 1]" = torch.ops.aten.mean.dim(pow_22, [2], True);  pow_22 = None
        add_49: "f32[1, 6144, 1]" = torch.ops.aten.add.Scalar(mean_17, 1.1920928955078125e-07);  mean_17 = None
        rsqrt_17: "f32[1, 6144, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        mul_70: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_120, rsqrt_17);  convert_element_type_120 = rsqrt_17 = None
        convert_element_type_121: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_70, torch.bfloat16);  mul_70 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        view_76: "bf16[6144, 768]" = torch.ops.aten.reshape.default(convert_element_type_121, [6144, 768])
        slice_36: "f32[3, 768, 768]" = torch.ops.aten.slice.Tensor(arg31_1, 0, 0, 3)
        view_75: "f32[2304, 768]" = torch.ops.aten.reshape.default(slice_36, [2304, 768]);  slice_36 = None
        convert_element_type_122: "bf16[2304, 768]" = torch.ops.prims.convert_element_type.default(view_75, torch.bfloat16);  view_75 = None
        permute_44: "bf16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_122, [1, 0]);  convert_element_type_122 = None
        mm_16: "bf16[6144, 2304]" = torch.ops.aten.mm.default(view_76, permute_44);  view_76 = permute_44 = None
        view_77: "bf16[1, 6144, 2304]" = torch.ops.aten.reshape.default(mm_16, [1, 6144, 2304]);  mm_16 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:679 in forward, code: .view(B, T, 3 * self.num_heads, self.head_dim)
        view_78: "bf16[1, 6144, 18, 128]" = torch.ops.aten.reshape.default(view_77, [1, 6144, 18, 128]);  view_77 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:680 in forward, code: .chunk(3, dim=-2)
        split_12 = torch.ops.aten.split.Tensor(view_78, 6, -2);  view_78 = None
        getitem_52: "bf16[1, 6144, 6, 128]" = split_12[0]
        getitem_53: "bf16[1, 6144, 6, 128]" = split_12[1]
        getitem_54: "bf16[1, 6144, 6, 128]" = split_12[2];  split_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_125: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_52, torch.float32);  getitem_52 = None
        pow_23: "f32[1, 6144, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_125, 2)
        mean_18: "f32[1, 6144, 6, 1]" = torch.ops.aten.mean.dim(pow_23, [3], True);  pow_23 = None
        add_50: "f32[1, 6144, 6, 1]" = torch.ops.aten.add.Scalar(mean_18, 1.1920928955078125e-07);  mean_18 = None
        rsqrt_18: "f32[1, 6144, 6, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        mul_71: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_125, rsqrt_18);  convert_element_type_125 = rsqrt_18 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        convert_element_type_default_13: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_71, torch.float32);  mul_71 = None
        split_13 = torch.ops.aten.split.Tensor(convert_element_type_default_13, 64, -1);  convert_element_type_default_13 = None
        getitem_55: "f32[1, 6144, 6, 64]" = split_13[0]
        getitem_56: "f32[1, 6144, 6, 64]" = split_13[1];  split_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_127: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_53, torch.float32);  getitem_53 = None
        pow_24: "f32[1, 6144, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_127, 2)
        mean_19: "f32[1, 6144, 6, 1]" = torch.ops.aten.mean.dim(pow_24, [3], True);  pow_24 = None
        add_51: "f32[1, 6144, 6, 1]" = torch.ops.aten.add.Scalar(mean_19, 1.1920928955078125e-07);  mean_19 = None
        rsqrt_19: "f32[1, 6144, 6, 1]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
        mul_72: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_127, rsqrt_19);  convert_element_type_127 = rsqrt_19 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        convert_element_type_default_12: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_72, torch.float32);  mul_72 = None
        split_14 = torch.ops.aten.split.Tensor(convert_element_type_default_12, 64, -1);  convert_element_type_default_12 = None
        getitem_57: "f32[1, 6144, 6, 64]" = split_14[0]
        getitem_58: "f32[1, 6144, 6, 64]" = split_14[1];  split_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:2041 in flex_attention, code: out, lse, max_scores = flex_attention_hop(
        sdpa_score4 = self.sdpa_score4
        sdpa_mask4 = self.sdpa_mask4

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:637 in forward, code: self.cos[None, : x_BTHD.size(-3), None, :],
        unsqueeze_75: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg32_1, 0)
        slice_37: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_75, 1, 0, 6144);  unsqueeze_75 = None
        unsqueeze_76: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_37, 2);  slice_37 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_73: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_55, unsqueeze_76)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:638 in forward, code: self.sin[None, : x_BTHD.size(-3), None, :],
        unsqueeze_77: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg33_1, 0)
        slice_38: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_77, 1, 0, 6144);  unsqueeze_77 = None
        unsqueeze_78: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_38, 2);  slice_38 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_74: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_56, unsqueeze_78)
        add_52: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_73, mul_74);  mul_73 = mul_74 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        neg_8: "f32[1, 6144, 1, 64]" = torch.ops.aten.neg.default(unsqueeze_78);  unsqueeze_78 = None
        mul_75: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_55, neg_8);  getitem_55 = neg_8 = None
        mul_76: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_56, unsqueeze_76);  getitem_56 = unsqueeze_76 = None
        add_53: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_75, mul_76);  mul_75 = mul_76 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        cat_8: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_52, add_53], 3);  add_52 = add_53 = None
        convert_element_type_130: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_8, torch.bfloat16);  cat_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:691 in forward, code: q.transpose(1, 2),
        permute_45: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(convert_element_type_130, [0, 2, 1, 3]);  convert_element_type_130 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:637 in forward, code: self.cos[None, : x_BTHD.size(-3), None, :],
        unsqueeze_79: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg32_1, 0);  arg32_1 = None
        slice_39: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_79, 1, 0, 6144);  unsqueeze_79 = None
        unsqueeze_80: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_39, 2);  slice_39 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_77: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_57, unsqueeze_80)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:638 in forward, code: self.sin[None, : x_BTHD.size(-3), None, :],
        unsqueeze_81: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg33_1, 0);  arg33_1 = None
        slice_40: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_81, 1, 0, 6144);  unsqueeze_81 = None
        unsqueeze_82: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_40, 2);  slice_40 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_78: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_58, unsqueeze_82)
        add_54: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_77, mul_78);  mul_77 = mul_78 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        neg_9: "f32[1, 6144, 1, 64]" = torch.ops.aten.neg.default(unsqueeze_82);  unsqueeze_82 = None
        mul_79: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_57, neg_9);  getitem_57 = neg_9 = None
        mul_80: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_58, unsqueeze_80);  getitem_58 = unsqueeze_80 = None
        add_55: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_79, mul_80);  mul_79 = mul_80 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        cat_9: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_54, add_55], 3);  add_54 = add_55 = None
        convert_element_type_132: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_9, torch.bfloat16);  cat_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:692 in forward, code: k.transpose(1, 2),
        permute_46: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(convert_element_type_132, [0, 2, 1, 3]);  convert_element_type_132 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_30: "f32[2]" = torch.ops.aten.select.int(view_7, 0, 4)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:689 in forward, code: v = lambdas[0] * v
        select_33: "f32[]" = torch.ops.aten.select.int(select_30, 0, 0);  select_30 = None
        mul_81: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(select_33, getitem_54);  select_33 = getitem_54 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:693 in forward, code: v.transpose(1, 2),
        permute_47: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(mul_81, [0, 2, 1, 3]);  mul_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:2041 in flex_attention, code: out, lse, max_scores = flex_attention_hop(
        flex_attention_4 = torch.ops.higher_order.flex_attention(permute_45, permute_46, permute_47, sdpa_score4, (6144, 6144, clamp_max, unsqueeze_9, clamp_max_1, unsqueeze_13, convert_element_type_2, clone_4, convert_element_type_4, clone_7, 128, 128, sdpa_mask4), 0.12, {'BACKEND': 'AUTO', 'PRESCALE_QK': False, 'ROWS_GUARANTEED_SAFE': False, 'BLOCKS_ARE_CONTIGUOUS': False, 'WRITE_DQ': True, 'OUTPUT_LOGSUMEXP': False, 'OUTPUT_MAX': False}, (), (cumsum,));  permute_45 = permute_46 = permute_47 = sdpa_score4 = sdpa_mask4 = None
        getitem_59: "bf16[1, 6, 6144, 128]" = flex_attention_4[0];  flex_attention_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_35: "f32[2]" = torch.ops.aten.select.int(view_6, 0, 5)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        select_37: "f32[]" = torch.ops.aten.select.int(select_35, 0, 0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:696 in forward, code: ).transpose(1, 2)
        permute_48: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_59, [0, 2, 1, 3]);  getitem_59 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        slice_41: "bf16[1, 6144, 12]" = torch.ops.aten.slice.Tensor(convert_element_type_121, 2, 0, 12);  convert_element_type_121 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:618 in forward, code: return F.linear(x, self.weight.type_as(x))
        expand_28: "bf16[1, 6144, 12]" = torch.ops.aten.expand.default(slice_41, [1, 6144, 12]);  slice_41 = None
        squeeze_dim_12: "bf16[6144, 12]" = torch.ops.aten.squeeze.dim(expand_28, 0);  expand_28 = None
        convert_element_type_133: "bf16[6, 12]" = torch.ops.prims.convert_element_type.default(arg34_1, torch.bfloat16);  arg34_1 = None
        permute_49: "bf16[12, 6]" = torch.ops.aten.permute.default(convert_element_type_133, [1, 0]);  convert_element_type_133 = None
        expand_29: "bf16[1, 12, 6]" = torch.ops.aten.expand.default(permute_49, [1, 12, 6]);  permute_49 = None
        squeeze_dim_13: "bf16[12, 6]" = torch.ops.aten.squeeze.dim(expand_29, 0);  expand_29 = None
        mm_default_6: "bf16[6144, 6]" = torch.ops.aten.mm.default(squeeze_dim_12, squeeze_dim_13);  squeeze_dim_12 = squeeze_dim_13 = None
        unsqueeze_default_6: "bf16[1, 6144, 6]" = torch.ops.aten.unsqueeze.default(mm_default_6, 0);  mm_default_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        sigmoid_4: "bf16[1, 6144, 6]" = torch.ops.aten.sigmoid.default(unsqueeze_default_6);  unsqueeze_default_6 = None
        view_83: "bf16[1, 6144, 6, 1]" = torch.ops.aten.reshape.default(sigmoid_4, [1, 6144, 6, 1]);  sigmoid_4 = None
        mul_82: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_48, view_83);  permute_48 = view_83 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:701 in forward, code: y = y.contiguous().view(
        view_84: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mul_82, [1, 6144, 768]);  mul_82 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:704 in forward, code: y = F.linear(y, self.qkvo_w[3].type_as(y))
        view_85: "bf16[6144, 768]" = torch.ops.aten.reshape.default(view_84, [6144, 768]);  view_84 = None
        select_34: "f32[768, 768]" = torch.ops.aten.select.int(arg31_1, 0, 3);  arg31_1 = None
        convert_element_type_136: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(select_34, torch.bfloat16);  select_34 = None
        permute_50: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_136, [1, 0]);  convert_element_type_136 = None
        mm_17: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_85, permute_50);  view_85 = permute_50 = None
        view_86: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_17, [1, 6144, 768]);  mm_17 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:751 in forward, code: x = x + self.attn(norm(x), ve, sa_lambdas, block_mask)
        add_56: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_48, view_86);  add_48 = view_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_139: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_56, torch.float32)
        pow_25: "f32[1, 6144, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_139, 2)
        mean_20: "f32[1, 6144, 1]" = torch.ops.aten.mean.dim(pow_25, [2], True);  pow_25 = None
        add_57: "f32[1, 6144, 1]" = torch.ops.aten.add.Scalar(mean_20, 1.1920928955078125e-07);  mean_20 = None
        rsqrt_20: "f32[1, 6144, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
        mul_83: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_139, rsqrt_20);  convert_element_type_139 = rsqrt_20 = None
        convert_element_type_140: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_83, torch.bfloat16);  mul_83 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:723 in forward, code: x = F.linear(x, self.c_fc.T.type_as(x))
        view_87: "bf16[6144, 768]" = torch.ops.aten.reshape.default(convert_element_type_140, [6144, 768]);  convert_element_type_140 = None
        permute_51: "f32[3072, 768]" = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
        convert_element_type_141: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(permute_51, torch.bfloat16);  permute_51 = None
        permute_52: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_141, [1, 0]);  convert_element_type_141 = None
        mm_18: "bf16[6144, 3072]" = torch.ops.aten.mm.default(view_87, permute_52);  view_87 = permute_52 = None
        view_88: "bf16[1, 6144, 3072]" = torch.ops.aten.reshape.default(mm_18, [1, 6144, 3072]);  mm_18 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:724 in forward, code: x = F.relu(
        relu_4: "bf16[1, 6144, 3072]" = torch.ops.aten.relu.default(view_88);  view_88 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:726 in forward, code: ).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        pow_26: "bf16[1, 6144, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu_4, 2);  relu_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:727 in forward, code: x = F.linear(x, self.c_proj.type_as(x))
        view_89: "bf16[6144, 3072]" = torch.ops.aten.reshape.default(pow_26, [6144, 3072]);  pow_26 = None
        convert_element_type_144: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg36_1, torch.bfloat16);  arg36_1 = None
        permute_53: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_144, [1, 0]);  convert_element_type_144 = None
        mm_19: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_89, permute_53);  view_89 = permute_53 = None
        view_90: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_19, [1, 6144, 768]);  mm_19 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:752 in forward, code: x = x + self.mlp(norm(x))
        add_58: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_56, view_90);  add_56 = view_90 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        mul_84: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(select_37, add_58);  select_37 = None
        select_38: "f32[]" = torch.ops.aten.select.int(select_35, 0, 1);  select_35 = None
        mul_85: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(select_38, convert_element_type_11);  select_38 = None
        add_59: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(mul_84, mul_85);  mul_84 = mul_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_147: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_59, torch.float32)
        pow_27: "f32[1, 6144, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_147, 2)
        mean_21: "f32[1, 6144, 1]" = torch.ops.aten.mean.dim(pow_27, [2], True);  pow_27 = None
        add_60: "f32[1, 6144, 1]" = torch.ops.aten.add.Scalar(mean_21, 1.1920928955078125e-07);  mean_21 = None
        rsqrt_21: "f32[1, 6144, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        mul_86: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_147, rsqrt_21);  convert_element_type_147 = rsqrt_21 = None
        convert_element_type_148: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_86, torch.bfloat16);  mul_86 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        view_92: "bf16[6144, 768]" = torch.ops.aten.reshape.default(convert_element_type_148, [6144, 768])
        slice_42: "f32[3, 768, 768]" = torch.ops.aten.slice.Tensor(arg37_1, 0, 0, 3)
        view_91: "f32[2304, 768]" = torch.ops.aten.reshape.default(slice_42, [2304, 768]);  slice_42 = None
        convert_element_type_149: "bf16[2304, 768]" = torch.ops.prims.convert_element_type.default(view_91, torch.bfloat16);  view_91 = None
        permute_54: "bf16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_149, [1, 0]);  convert_element_type_149 = None
        mm_20: "bf16[6144, 2304]" = torch.ops.aten.mm.default(view_92, permute_54);  view_92 = permute_54 = None
        view_93: "bf16[1, 6144, 2304]" = torch.ops.aten.reshape.default(mm_20, [1, 6144, 2304]);  mm_20 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:679 in forward, code: .view(B, T, 3 * self.num_heads, self.head_dim)
        view_94: "bf16[1, 6144, 18, 128]" = torch.ops.aten.reshape.default(view_93, [1, 6144, 18, 128]);  view_93 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:680 in forward, code: .chunk(3, dim=-2)
        split_15 = torch.ops.aten.split.Tensor(view_94, 6, -2);  view_94 = None
        getitem_62: "bf16[1, 6144, 6, 128]" = split_15[0]
        getitem_63: "bf16[1, 6144, 6, 128]" = split_15[1]
        getitem_64: "bf16[1, 6144, 6, 128]" = split_15[2];  split_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_152: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_62, torch.float32);  getitem_62 = None
        pow_28: "f32[1, 6144, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_152, 2)
        mean_22: "f32[1, 6144, 6, 1]" = torch.ops.aten.mean.dim(pow_28, [3], True);  pow_28 = None
        add_61: "f32[1, 6144, 6, 1]" = torch.ops.aten.add.Scalar(mean_22, 1.1920928955078125e-07);  mean_22 = None
        rsqrt_22: "f32[1, 6144, 6, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
        mul_87: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_152, rsqrt_22);  convert_element_type_152 = rsqrt_22 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        convert_element_type_default_11: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_87, torch.float32);  mul_87 = None
        split_16 = torch.ops.aten.split.Tensor(convert_element_type_default_11, 64, -1);  convert_element_type_default_11 = None
        getitem_65: "f32[1, 6144, 6, 64]" = split_16[0]
        getitem_66: "f32[1, 6144, 6, 64]" = split_16[1];  split_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_154: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_63, torch.float32);  getitem_63 = None
        pow_29: "f32[1, 6144, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_154, 2)
        mean_23: "f32[1, 6144, 6, 1]" = torch.ops.aten.mean.dim(pow_29, [3], True);  pow_29 = None
        add_62: "f32[1, 6144, 6, 1]" = torch.ops.aten.add.Scalar(mean_23, 1.1920928955078125e-07);  mean_23 = None
        rsqrt_23: "f32[1, 6144, 6, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        mul_88: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_154, rsqrt_23);  convert_element_type_154 = rsqrt_23 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        convert_element_type_default_10: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_88, torch.float32);  mul_88 = None
        split_17 = torch.ops.aten.split.Tensor(convert_element_type_default_10, 64, -1);  convert_element_type_default_10 = None
        getitem_67: "f32[1, 6144, 6, 64]" = split_17[0]
        getitem_68: "f32[1, 6144, 6, 64]" = split_17[1];  split_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:2041 in flex_attention, code: out, lse, max_scores = flex_attention_hop(
        sdpa_score5 = self.sdpa_score5
        sdpa_mask5 = self.sdpa_mask5

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:637 in forward, code: self.cos[None, : x_BTHD.size(-3), None, :],
        unsqueeze_83: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg38_1, 0)
        slice_43: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_83, 1, 0, 6144);  unsqueeze_83 = None
        unsqueeze_84: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_43, 2);  slice_43 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_89: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_65, unsqueeze_84)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:638 in forward, code: self.sin[None, : x_BTHD.size(-3), None, :],
        unsqueeze_85: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg39_1, 0)
        slice_44: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_85, 1, 0, 6144);  unsqueeze_85 = None
        unsqueeze_86: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_44, 2);  slice_44 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_90: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_66, unsqueeze_86)
        add_63: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_89, mul_90);  mul_89 = mul_90 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        neg_10: "f32[1, 6144, 1, 64]" = torch.ops.aten.neg.default(unsqueeze_86);  unsqueeze_86 = None
        mul_91: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_65, neg_10);  getitem_65 = neg_10 = None
        mul_92: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_66, unsqueeze_84);  getitem_66 = unsqueeze_84 = None
        add_64: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_91, mul_92);  mul_91 = mul_92 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        cat_10: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_63, add_64], 3);  add_63 = add_64 = None
        convert_element_type_157: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_10, torch.bfloat16);  cat_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:691 in forward, code: q.transpose(1, 2),
        permute_55: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(convert_element_type_157, [0, 2, 1, 3]);  convert_element_type_157 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:637 in forward, code: self.cos[None, : x_BTHD.size(-3), None, :],
        unsqueeze_87: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg38_1, 0);  arg38_1 = None
        slice_45: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_87, 1, 0, 6144);  unsqueeze_87 = None
        unsqueeze_88: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_45, 2);  slice_45 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_93: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_67, unsqueeze_88)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:638 in forward, code: self.sin[None, : x_BTHD.size(-3), None, :],
        unsqueeze_89: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg39_1, 0);  arg39_1 = None
        slice_46: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_89, 1, 0, 6144);  unsqueeze_89 = None
        unsqueeze_90: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_46, 2);  slice_46 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_94: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_68, unsqueeze_90)
        add_65: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_93, mul_94);  mul_93 = mul_94 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        neg_11: "f32[1, 6144, 1, 64]" = torch.ops.aten.neg.default(unsqueeze_90);  unsqueeze_90 = None
        mul_95: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_67, neg_11);  getitem_67 = neg_11 = None
        mul_96: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_68, unsqueeze_88);  getitem_68 = unsqueeze_88 = None
        add_66: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_95, mul_96);  mul_95 = mul_96 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        cat_11: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_65, add_66], 3);  add_65 = add_66 = None
        convert_element_type_159: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_11, torch.bfloat16);  cat_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:692 in forward, code: k.transpose(1, 2),
        permute_56: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(convert_element_type_159, [0, 2, 1, 3]);  convert_element_type_159 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_36: "f32[2]" = torch.ops.aten.select.int(view_7, 0, 5)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:689 in forward, code: v = lambdas[0] * v
        select_39: "f32[]" = torch.ops.aten.select.int(select_36, 0, 0);  select_36 = None
        mul_97: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(select_39, getitem_64);  select_39 = getitem_64 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:693 in forward, code: v.transpose(1, 2),
        permute_57: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(mul_97, [0, 2, 1, 3]);  mul_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:2041 in flex_attention, code: out, lse, max_scores = flex_attention_hop(
        flex_attention_5 = torch.ops.higher_order.flex_attention(permute_55, permute_56, permute_57, sdpa_score5, (6144, 6144, clamp_max_2, unsqueeze_9, clamp_max_3, unsqueeze_13, convert_element_type_6, clone_10, convert_element_type_8, clone_13, 128, 128, sdpa_mask5), 0.12, {'BACKEND': 'AUTO', 'PRESCALE_QK': False, 'ROWS_GUARANTEED_SAFE': False, 'BLOCKS_ARE_CONTIGUOUS': False, 'WRITE_DQ': True, 'OUTPUT_LOGSUMEXP': False, 'OUTPUT_MAX': False}, (), (cumsum,));  permute_55 = permute_56 = permute_57 = sdpa_score5 = sdpa_mask5 = None
        getitem_69: "bf16[1, 6, 6144, 128]" = flex_attention_5[0];  flex_attention_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_42: "f32[2]" = torch.ops.aten.select.int(view_6, 0, 6)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        select_44: "f32[]" = torch.ops.aten.select.int(select_42, 0, 0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:913 in forward, code: skip_weights = self.scalars[: (len(self.blocks) // 2)]
        slice_9: "f32[6]" = torch.ops.aten.slice.Tensor(arg6_1, 0, 0, 6);  arg6_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:923 in forward, code: x = x + skip_weights[i - n] * skip_connections.pop()
        select_41: "f32[]" = torch.ops.aten.select.int(slice_9, 0, 0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:696 in forward, code: ).transpose(1, 2)
        permute_58: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_69, [0, 2, 1, 3]);  getitem_69 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        slice_47: "bf16[1, 6144, 12]" = torch.ops.aten.slice.Tensor(convert_element_type_148, 2, 0, 12);  convert_element_type_148 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:618 in forward, code: return F.linear(x, self.weight.type_as(x))
        expand_30: "bf16[1, 6144, 12]" = torch.ops.aten.expand.default(slice_47, [1, 6144, 12]);  slice_47 = None
        squeeze_dim_10: "bf16[6144, 12]" = torch.ops.aten.squeeze.dim(expand_30, 0);  expand_30 = None
        convert_element_type_160: "bf16[6, 12]" = torch.ops.prims.convert_element_type.default(arg40_1, torch.bfloat16);  arg40_1 = None
        permute_59: "bf16[12, 6]" = torch.ops.aten.permute.default(convert_element_type_160, [1, 0]);  convert_element_type_160 = None
        expand_31: "bf16[1, 12, 6]" = torch.ops.aten.expand.default(permute_59, [1, 12, 6]);  permute_59 = None
        squeeze_dim_11: "bf16[12, 6]" = torch.ops.aten.squeeze.dim(expand_31, 0);  expand_31 = None
        mm_default_5: "bf16[6144, 6]" = torch.ops.aten.mm.default(squeeze_dim_10, squeeze_dim_11);  squeeze_dim_10 = squeeze_dim_11 = None
        unsqueeze_default_5: "bf16[1, 6144, 6]" = torch.ops.aten.unsqueeze.default(mm_default_5, 0);  mm_default_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        sigmoid_5: "bf16[1, 6144, 6]" = torch.ops.aten.sigmoid.default(unsqueeze_default_5);  unsqueeze_default_5 = None
        view_99: "bf16[1, 6144, 6, 1]" = torch.ops.aten.reshape.default(sigmoid_5, [1, 6144, 6, 1]);  sigmoid_5 = None
        mul_98: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_58, view_99);  permute_58 = view_99 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:701 in forward, code: y = y.contiguous().view(
        view_100: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mul_98, [1, 6144, 768]);  mul_98 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:704 in forward, code: y = F.linear(y, self.qkvo_w[3].type_as(y))
        view_101: "bf16[6144, 768]" = torch.ops.aten.reshape.default(view_100, [6144, 768]);  view_100 = None
        select_40: "f32[768, 768]" = torch.ops.aten.select.int(arg37_1, 0, 3);  arg37_1 = None
        convert_element_type_163: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(select_40, torch.bfloat16);  select_40 = None
        permute_60: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_163, [1, 0]);  convert_element_type_163 = None
        mm_21: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_101, permute_60);  view_101 = permute_60 = None
        view_102: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_21, [1, 6144, 768]);  mm_21 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:751 in forward, code: x = x + self.attn(norm(x), ve, sa_lambdas, block_mask)
        add_67: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_59, view_102);  add_59 = view_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_166: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_67, torch.float32)
        pow_30: "f32[1, 6144, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_166, 2)
        mean_24: "f32[1, 6144, 1]" = torch.ops.aten.mean.dim(pow_30, [2], True);  pow_30 = None
        add_68: "f32[1, 6144, 1]" = torch.ops.aten.add.Scalar(mean_24, 1.1920928955078125e-07);  mean_24 = None
        rsqrt_24: "f32[1, 6144, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        mul_99: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_166, rsqrt_24);  convert_element_type_166 = rsqrt_24 = None
        convert_element_type_167: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_99, torch.bfloat16);  mul_99 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:723 in forward, code: x = F.linear(x, self.c_fc.T.type_as(x))
        view_103: "bf16[6144, 768]" = torch.ops.aten.reshape.default(convert_element_type_167, [6144, 768]);  convert_element_type_167 = None
        permute_61: "f32[3072, 768]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        convert_element_type_168: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(permute_61, torch.bfloat16);  permute_61 = None
        permute_62: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_168, [1, 0]);  convert_element_type_168 = None
        mm_22: "bf16[6144, 3072]" = torch.ops.aten.mm.default(view_103, permute_62);  view_103 = permute_62 = None
        view_104: "bf16[1, 6144, 3072]" = torch.ops.aten.reshape.default(mm_22, [1, 6144, 3072]);  mm_22 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:724 in forward, code: x = F.relu(
        relu_5: "bf16[1, 6144, 3072]" = torch.ops.aten.relu.default(view_104);  view_104 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:726 in forward, code: ).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        pow_31: "bf16[1, 6144, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu_5, 2);  relu_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:727 in forward, code: x = F.linear(x, self.c_proj.type_as(x))
        view_105: "bf16[6144, 3072]" = torch.ops.aten.reshape.default(pow_31, [6144, 3072]);  pow_31 = None
        convert_element_type_171: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg42_1, torch.bfloat16);  arg42_1 = None
        permute_63: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_171, [1, 0]);  convert_element_type_171 = None
        mm_23: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_105, permute_63);  view_105 = permute_63 = None
        view_106: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_23, [1, 6144, 768]);  mm_23 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:752 in forward, code: x = x + self.mlp(norm(x))
        add_69: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_67, view_106);  add_67 = view_106 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:923 in forward, code: x = x + skip_weights[i - n] * skip_connections.pop()
        mul_100: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(select_41, add_69);  select_41 = None
        add_70: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_69, mul_100);  add_69 = mul_100 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        mul_101: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(select_44, add_70);  select_44 = add_70 = None
        select_45: "f32[]" = torch.ops.aten.select.int(select_42, 0, 1);  select_42 = None
        mul_102: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(select_45, convert_element_type_11);  select_45 = None
        add_71: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(mul_101, mul_102);  mul_101 = mul_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_174: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_71, torch.float32)
        pow_32: "f32[1, 6144, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_174, 2)
        mean_25: "f32[1, 6144, 1]" = torch.ops.aten.mean.dim(pow_32, [2], True);  pow_32 = None
        add_72: "f32[1, 6144, 1]" = torch.ops.aten.add.Scalar(mean_25, 1.1920928955078125e-07);  mean_25 = None
        rsqrt_25: "f32[1, 6144, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        mul_103: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_174, rsqrt_25);  convert_element_type_174 = rsqrt_25 = None
        convert_element_type_175: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_103, torch.bfloat16);  mul_103 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        view_108: "bf16[6144, 768]" = torch.ops.aten.reshape.default(convert_element_type_175, [6144, 768])
        slice_48: "f32[3, 768, 768]" = torch.ops.aten.slice.Tensor(arg43_1, 0, 0, 3)
        view_107: "f32[2304, 768]" = torch.ops.aten.reshape.default(slice_48, [2304, 768]);  slice_48 = None
        convert_element_type_176: "bf16[2304, 768]" = torch.ops.prims.convert_element_type.default(view_107, torch.bfloat16);  view_107 = None
        permute_64: "bf16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_176, [1, 0]);  convert_element_type_176 = None
        mm_24: "bf16[6144, 2304]" = torch.ops.aten.mm.default(view_108, permute_64);  view_108 = permute_64 = None
        view_109: "bf16[1, 6144, 2304]" = torch.ops.aten.reshape.default(mm_24, [1, 6144, 2304]);  mm_24 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:679 in forward, code: .view(B, T, 3 * self.num_heads, self.head_dim)
        view_110: "bf16[1, 6144, 18, 128]" = torch.ops.aten.reshape.default(view_109, [1, 6144, 18, 128]);  view_109 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:680 in forward, code: .chunk(3, dim=-2)
        split_18 = torch.ops.aten.split.Tensor(view_110, 6, -2);  view_110 = None
        getitem_72: "bf16[1, 6144, 6, 128]" = split_18[0]
        getitem_73: "bf16[1, 6144, 6, 128]" = split_18[1]
        getitem_74: "bf16[1, 6144, 6, 128]" = split_18[2];  split_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_179: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_72, torch.float32);  getitem_72 = None
        pow_33: "f32[1, 6144, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_179, 2)
        mean_26: "f32[1, 6144, 6, 1]" = torch.ops.aten.mean.dim(pow_33, [3], True);  pow_33 = None
        add_73: "f32[1, 6144, 6, 1]" = torch.ops.aten.add.Scalar(mean_26, 1.1920928955078125e-07);  mean_26 = None
        rsqrt_26: "f32[1, 6144, 6, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
        mul_104: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_179, rsqrt_26);  convert_element_type_179 = rsqrt_26 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        convert_element_type_default_9: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_104, torch.float32);  mul_104 = None
        split_19 = torch.ops.aten.split.Tensor(convert_element_type_default_9, 64, -1);  convert_element_type_default_9 = None
        getitem_75: "f32[1, 6144, 6, 64]" = split_19[0]
        getitem_76: "f32[1, 6144, 6, 64]" = split_19[1];  split_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_181: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_73, torch.float32);  getitem_73 = None
        pow_34: "f32[1, 6144, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_181, 2)
        mean_27: "f32[1, 6144, 6, 1]" = torch.ops.aten.mean.dim(pow_34, [3], True);  pow_34 = None
        add_74: "f32[1, 6144, 6, 1]" = torch.ops.aten.add.Scalar(mean_27, 1.1920928955078125e-07);  mean_27 = None
        rsqrt_27: "f32[1, 6144, 6, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        mul_105: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_181, rsqrt_27);  convert_element_type_181 = rsqrt_27 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        convert_element_type_default_8: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_105, torch.float32);  mul_105 = None
        split_20 = torch.ops.aten.split.Tensor(convert_element_type_default_8, 64, -1);  convert_element_type_default_8 = None
        getitem_77: "f32[1, 6144, 6, 64]" = split_20[0]
        getitem_78: "f32[1, 6144, 6, 64]" = split_20[1];  split_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:2041 in flex_attention, code: out, lse, max_scores = flex_attention_hop(
        sdpa_score6 = self.sdpa_score6
        sdpa_mask6 = self.sdpa_mask6

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:637 in forward, code: self.cos[None, : x_BTHD.size(-3), None, :],
        unsqueeze_91: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg44_1, 0)
        slice_49: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_91, 1, 0, 6144);  unsqueeze_91 = None
        unsqueeze_92: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_49, 2);  slice_49 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_106: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_75, unsqueeze_92)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:638 in forward, code: self.sin[None, : x_BTHD.size(-3), None, :],
        unsqueeze_93: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg45_1, 0)
        slice_50: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_93, 1, 0, 6144);  unsqueeze_93 = None
        unsqueeze_94: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_50, 2);  slice_50 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_107: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_76, unsqueeze_94)
        add_75: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_106, mul_107);  mul_106 = mul_107 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        neg_12: "f32[1, 6144, 1, 64]" = torch.ops.aten.neg.default(unsqueeze_94);  unsqueeze_94 = None
        mul_108: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_75, neg_12);  getitem_75 = neg_12 = None
        mul_109: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_76, unsqueeze_92);  getitem_76 = unsqueeze_92 = None
        add_76: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_108, mul_109);  mul_108 = mul_109 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        cat_12: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_75, add_76], 3);  add_75 = add_76 = None
        convert_element_type_184: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_12, torch.bfloat16);  cat_12 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:691 in forward, code: q.transpose(1, 2),
        permute_65: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(convert_element_type_184, [0, 2, 1, 3]);  convert_element_type_184 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:637 in forward, code: self.cos[None, : x_BTHD.size(-3), None, :],
        unsqueeze_95: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg44_1, 0);  arg44_1 = None
        slice_51: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_95, 1, 0, 6144);  unsqueeze_95 = None
        unsqueeze_96: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_51, 2);  slice_51 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_110: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_77, unsqueeze_96)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:638 in forward, code: self.sin[None, : x_BTHD.size(-3), None, :],
        unsqueeze_97: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg45_1, 0);  arg45_1 = None
        slice_52: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_97, 1, 0, 6144);  unsqueeze_97 = None
        unsqueeze_98: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_52, 2);  slice_52 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_111: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_78, unsqueeze_98)
        add_77: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_110, mul_111);  mul_110 = mul_111 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        neg_13: "f32[1, 6144, 1, 64]" = torch.ops.aten.neg.default(unsqueeze_98);  unsqueeze_98 = None
        mul_112: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_77, neg_13);  getitem_77 = neg_13 = None
        mul_113: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_78, unsqueeze_96);  getitem_78 = unsqueeze_96 = None
        add_78: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_112, mul_113);  mul_112 = mul_113 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        cat_13: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_77, add_78], 3);  add_77 = add_78 = None
        convert_element_type_186: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_13, torch.bfloat16);  cat_13 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:692 in forward, code: k.transpose(1, 2),
        permute_66: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(convert_element_type_186, [0, 2, 1, 3]);  convert_element_type_186 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_43: "f32[2]" = torch.ops.aten.select.int(view_7, 0, 6)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:689 in forward, code: v = lambdas[0] * v
        select_46: "f32[]" = torch.ops.aten.select.int(select_43, 0, 0);  select_43 = None
        mul_114: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(select_46, getitem_74);  select_46 = getitem_74 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:693 in forward, code: v.transpose(1, 2),
        permute_67: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(mul_114, [0, 2, 1, 3]);  mul_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:2041 in flex_attention, code: out, lse, max_scores = flex_attention_hop(
        flex_attention_6 = torch.ops.higher_order.flex_attention(permute_65, permute_66, permute_67, sdpa_score6, (6144, 6144, clamp_max_2, unsqueeze_9, clamp_max_3, unsqueeze_13, convert_element_type_6, clone_10, convert_element_type_8, clone_13, 128, 128, sdpa_mask6), 0.12, {'BACKEND': 'AUTO', 'PRESCALE_QK': False, 'ROWS_GUARANTEED_SAFE': False, 'BLOCKS_ARE_CONTIGUOUS': False, 'WRITE_DQ': True, 'OUTPUT_LOGSUMEXP': False, 'OUTPUT_MAX': False}, (), (cumsum,));  permute_65 = permute_66 = permute_67 = sdpa_score6 = sdpa_mask6 = None
        getitem_79: "bf16[1, 6, 6144, 128]" = flex_attention_6[0];  flex_attention_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_53: "f32[2]" = torch.ops.aten.select.int(view_6, 0, 8)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        select_55: "f32[]" = torch.ops.aten.select.int(select_53, 0, 0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_49: "f32[2]" = torch.ops.aten.select.int(view_6, 0, 7)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        select_50: "f32[]" = torch.ops.aten.select.int(select_49, 0, 0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:696 in forward, code: ).transpose(1, 2)
        permute_68: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_79, [0, 2, 1, 3]);  getitem_79 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        slice_53: "bf16[1, 6144, 12]" = torch.ops.aten.slice.Tensor(convert_element_type_175, 2, 0, 12);  convert_element_type_175 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:618 in forward, code: return F.linear(x, self.weight.type_as(x))
        expand_32: "bf16[1, 6144, 12]" = torch.ops.aten.expand.default(slice_53, [1, 6144, 12]);  slice_53 = None
        squeeze_dim_8: "bf16[6144, 12]" = torch.ops.aten.squeeze.dim(expand_32, 0);  expand_32 = None
        convert_element_type_187: "bf16[6, 12]" = torch.ops.prims.convert_element_type.default(arg46_1, torch.bfloat16);  arg46_1 = None
        permute_69: "bf16[12, 6]" = torch.ops.aten.permute.default(convert_element_type_187, [1, 0]);  convert_element_type_187 = None
        expand_33: "bf16[1, 12, 6]" = torch.ops.aten.expand.default(permute_69, [1, 12, 6]);  permute_69 = None
        squeeze_dim_9: "bf16[12, 6]" = torch.ops.aten.squeeze.dim(expand_33, 0);  expand_33 = None
        mm_default_4: "bf16[6144, 6]" = torch.ops.aten.mm.default(squeeze_dim_8, squeeze_dim_9);  squeeze_dim_8 = squeeze_dim_9 = None
        unsqueeze_default_4: "bf16[1, 6144, 6]" = torch.ops.aten.unsqueeze.default(mm_default_4, 0);  mm_default_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        sigmoid_6: "bf16[1, 6144, 6]" = torch.ops.aten.sigmoid.default(unsqueeze_default_4);  unsqueeze_default_4 = None
        view_115: "bf16[1, 6144, 6, 1]" = torch.ops.aten.reshape.default(sigmoid_6, [1, 6144, 6, 1]);  sigmoid_6 = None
        mul_115: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_68, view_115);  permute_68 = view_115 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:701 in forward, code: y = y.contiguous().view(
        view_116: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mul_115, [1, 6144, 768]);  mul_115 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:704 in forward, code: y = F.linear(y, self.qkvo_w[3].type_as(y))
        view_117: "bf16[6144, 768]" = torch.ops.aten.reshape.default(view_116, [6144, 768]);  view_116 = None
        select_47: "f32[768, 768]" = torch.ops.aten.select.int(arg43_1, 0, 3);  arg43_1 = None
        convert_element_type_190: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(select_47, torch.bfloat16);  select_47 = None
        permute_70: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_190, [1, 0]);  convert_element_type_190 = None
        mm_25: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_117, permute_70);  view_117 = permute_70 = None
        view_118: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_25, [1, 6144, 768]);  mm_25 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:751 in forward, code: x = x + self.attn(norm(x), ve, sa_lambdas, block_mask)
        add_79: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_71, view_118);  add_71 = view_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_193: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_79, torch.float32)
        pow_35: "f32[1, 6144, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_193, 2)
        mean_28: "f32[1, 6144, 1]" = torch.ops.aten.mean.dim(pow_35, [2], True);  pow_35 = None
        add_80: "f32[1, 6144, 1]" = torch.ops.aten.add.Scalar(mean_28, 1.1920928955078125e-07);  mean_28 = None
        rsqrt_28: "f32[1, 6144, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        mul_116: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_193, rsqrt_28);  convert_element_type_193 = rsqrt_28 = None
        convert_element_type_194: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_116, torch.bfloat16);  mul_116 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:723 in forward, code: x = F.linear(x, self.c_fc.T.type_as(x))
        view_119: "bf16[6144, 768]" = torch.ops.aten.reshape.default(convert_element_type_194, [6144, 768]);  convert_element_type_194 = None
        permute_71: "f32[3072, 768]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        convert_element_type_195: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(permute_71, torch.bfloat16);  permute_71 = None
        permute_72: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_195, [1, 0]);  convert_element_type_195 = None
        mm_26: "bf16[6144, 3072]" = torch.ops.aten.mm.default(view_119, permute_72);  view_119 = permute_72 = None
        view_120: "bf16[1, 6144, 3072]" = torch.ops.aten.reshape.default(mm_26, [1, 6144, 3072]);  mm_26 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:724 in forward, code: x = F.relu(
        relu_6: "bf16[1, 6144, 3072]" = torch.ops.aten.relu.default(view_120);  view_120 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:726 in forward, code: ).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        pow_36: "bf16[1, 6144, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu_6, 2);  relu_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:727 in forward, code: x = F.linear(x, self.c_proj.type_as(x))
        view_121: "bf16[6144, 3072]" = torch.ops.aten.reshape.default(pow_36, [6144, 3072]);  pow_36 = None
        convert_element_type_198: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg48_1, torch.bfloat16);  arg48_1 = None
        permute_73: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_198, [1, 0]);  convert_element_type_198 = None
        mm_27: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_121, permute_73);  view_121 = permute_73 = None
        view_122: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_27, [1, 6144, 768]);  mm_27 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:752 in forward, code: x = x + self.mlp(norm(x))
        add_81: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_79, view_122);  add_79 = view_122 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:923 in forward, code: x = x + skip_weights[i - n] * skip_connections.pop()
        select_48: "f32[]" = torch.ops.aten.select.int(slice_9, 0, 1)
        mul_117: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(select_48, add_58);  select_48 = add_58 = None
        add_82: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_81, mul_117);  add_81 = mul_117 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        mul_118: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(select_50, add_82);  select_50 = add_82 = None
        select_51: "f32[]" = torch.ops.aten.select.int(select_49, 0, 1);  select_49 = None
        mul_119: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(select_51, convert_element_type_11);  select_51 = None
        add_83: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(mul_118, mul_119);  mul_118 = mul_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_201: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_83, torch.float32)
        pow_37: "f32[1, 6144, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_201, 2)
        mean_29: "f32[1, 6144, 1]" = torch.ops.aten.mean.dim(pow_37, [2], True);  pow_37 = None
        add_84: "f32[1, 6144, 1]" = torch.ops.aten.add.Scalar(mean_29, 1.1920928955078125e-07);  mean_29 = None
        rsqrt_29: "f32[1, 6144, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        mul_120: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_201, rsqrt_29);  convert_element_type_201 = rsqrt_29 = None
        convert_element_type_202: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_120, torch.bfloat16);  mul_120 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:723 in forward, code: x = F.linear(x, self.c_fc.T.type_as(x))
        view_123: "bf16[6144, 768]" = torch.ops.aten.reshape.default(convert_element_type_202, [6144, 768]);  convert_element_type_202 = None
        permute_74: "f32[3072, 768]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        convert_element_type_203: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(permute_74, torch.bfloat16);  permute_74 = None
        permute_75: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_203, [1, 0]);  convert_element_type_203 = None
        mm_28: "bf16[6144, 3072]" = torch.ops.aten.mm.default(view_123, permute_75);  view_123 = permute_75 = None
        view_124: "bf16[1, 6144, 3072]" = torch.ops.aten.reshape.default(mm_28, [1, 6144, 3072]);  mm_28 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:724 in forward, code: x = F.relu(
        relu_7: "bf16[1, 6144, 3072]" = torch.ops.aten.relu.default(view_124);  view_124 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:726 in forward, code: ).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        pow_38: "bf16[1, 6144, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu_7, 2);  relu_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:727 in forward, code: x = F.linear(x, self.c_proj.type_as(x))
        view_125: "bf16[6144, 3072]" = torch.ops.aten.reshape.default(pow_38, [6144, 3072]);  pow_38 = None
        convert_element_type_206: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg50_1, torch.bfloat16);  arg50_1 = None
        permute_76: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_206, [1, 0]);  convert_element_type_206 = None
        mm_29: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_125, permute_76);  view_125 = permute_76 = None
        view_126: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_29, [1, 6144, 768]);  mm_29 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:752 in forward, code: x = x + self.mlp(norm(x))
        add_85: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_83, view_126);  add_83 = view_126 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:923 in forward, code: x = x + skip_weights[i - n] * skip_connections.pop()
        select_52: "f32[]" = torch.ops.aten.select.int(slice_9, 0, 2)
        mul_121: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(select_52, add_47);  select_52 = add_47 = None
        add_86: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_85, mul_121);  add_85 = mul_121 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        mul_122: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(select_55, add_86);  select_55 = add_86 = None
        select_56: "f32[]" = torch.ops.aten.select.int(select_53, 0, 1);  select_53 = None
        mul_123: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(select_56, convert_element_type_11);  select_56 = None
        add_87: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(mul_122, mul_123);  mul_122 = mul_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_209: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_87, torch.float32)
        pow_39: "f32[1, 6144, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_209, 2)
        mean_30: "f32[1, 6144, 1]" = torch.ops.aten.mean.dim(pow_39, [2], True);  pow_39 = None
        add_88: "f32[1, 6144, 1]" = torch.ops.aten.add.Scalar(mean_30, 1.1920928955078125e-07);  mean_30 = None
        rsqrt_30: "f32[1, 6144, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
        mul_124: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_209, rsqrt_30);  convert_element_type_209 = rsqrt_30 = None
        convert_element_type_210: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_124, torch.bfloat16);  mul_124 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        view_128: "bf16[6144, 768]" = torch.ops.aten.reshape.default(convert_element_type_210, [6144, 768])
        slice_54: "f32[3, 768, 768]" = torch.ops.aten.slice.Tensor(arg51_1, 0, 0, 3)
        view_127: "f32[2304, 768]" = torch.ops.aten.reshape.default(slice_54, [2304, 768]);  slice_54 = None
        convert_element_type_211: "bf16[2304, 768]" = torch.ops.prims.convert_element_type.default(view_127, torch.bfloat16);  view_127 = None
        permute_77: "bf16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_211, [1, 0]);  convert_element_type_211 = None
        mm_30: "bf16[6144, 2304]" = torch.ops.aten.mm.default(view_128, permute_77);  view_128 = permute_77 = None
        view_129: "bf16[1, 6144, 2304]" = torch.ops.aten.reshape.default(mm_30, [1, 6144, 2304]);  mm_30 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:679 in forward, code: .view(B, T, 3 * self.num_heads, self.head_dim)
        view_130: "bf16[1, 6144, 18, 128]" = torch.ops.aten.reshape.default(view_129, [1, 6144, 18, 128]);  view_129 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:680 in forward, code: .chunk(3, dim=-2)
        split_21 = torch.ops.aten.split.Tensor(view_130, 6, -2);  view_130 = None
        getitem_82: "bf16[1, 6144, 6, 128]" = split_21[0]
        getitem_83: "bf16[1, 6144, 6, 128]" = split_21[1]
        getitem_84: "bf16[1, 6144, 6, 128]" = split_21[2];  split_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_214: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_82, torch.float32);  getitem_82 = None
        pow_40: "f32[1, 6144, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_214, 2)
        mean_31: "f32[1, 6144, 6, 1]" = torch.ops.aten.mean.dim(pow_40, [3], True);  pow_40 = None
        add_89: "f32[1, 6144, 6, 1]" = torch.ops.aten.add.Scalar(mean_31, 1.1920928955078125e-07);  mean_31 = None
        rsqrt_31: "f32[1, 6144, 6, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
        mul_125: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_214, rsqrt_31);  convert_element_type_214 = rsqrt_31 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        convert_element_type_default_7: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_125, torch.float32);  mul_125 = None
        split_22 = torch.ops.aten.split.Tensor(convert_element_type_default_7, 64, -1);  convert_element_type_default_7 = None
        getitem_85: "f32[1, 6144, 6, 64]" = split_22[0]
        getitem_86: "f32[1, 6144, 6, 64]" = split_22[1];  split_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_216: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_83, torch.float32);  getitem_83 = None
        pow_41: "f32[1, 6144, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_216, 2)
        mean_32: "f32[1, 6144, 6, 1]" = torch.ops.aten.mean.dim(pow_41, [3], True);  pow_41 = None
        add_90: "f32[1, 6144, 6, 1]" = torch.ops.aten.add.Scalar(mean_32, 1.1920928955078125e-07);  mean_32 = None
        rsqrt_32: "f32[1, 6144, 6, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        mul_126: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_216, rsqrt_32);  convert_element_type_216 = rsqrt_32 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        convert_element_type_default_6: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_126, torch.float32);  mul_126 = None
        split_23 = torch.ops.aten.split.Tensor(convert_element_type_default_6, 64, -1);  convert_element_type_default_6 = None
        getitem_87: "f32[1, 6144, 6, 64]" = split_23[0]
        getitem_88: "f32[1, 6144, 6, 64]" = split_23[1];  split_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:2041 in flex_attention, code: out, lse, max_scores = flex_attention_hop(
        sdpa_score7 = self.sdpa_score7
        sdpa_mask7 = self.sdpa_mask7

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:637 in forward, code: self.cos[None, : x_BTHD.size(-3), None, :],
        unsqueeze_99: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg52_1, 0)
        slice_55: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_99, 1, 0, 6144);  unsqueeze_99 = None
        unsqueeze_100: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_55, 2);  slice_55 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_127: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_85, unsqueeze_100)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:638 in forward, code: self.sin[None, : x_BTHD.size(-3), None, :],
        unsqueeze_101: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg53_1, 0)
        slice_56: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_101, 1, 0, 6144);  unsqueeze_101 = None
        unsqueeze_102: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_56, 2);  slice_56 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_128: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_86, unsqueeze_102)
        add_91: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_127, mul_128);  mul_127 = mul_128 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        neg_14: "f32[1, 6144, 1, 64]" = torch.ops.aten.neg.default(unsqueeze_102);  unsqueeze_102 = None
        mul_129: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_85, neg_14);  getitem_85 = neg_14 = None
        mul_130: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_86, unsqueeze_100);  getitem_86 = unsqueeze_100 = None
        add_92: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_129, mul_130);  mul_129 = mul_130 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        cat_14: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_91, add_92], 3);  add_91 = add_92 = None
        convert_element_type_219: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_14, torch.bfloat16);  cat_14 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:691 in forward, code: q.transpose(1, 2),
        permute_78: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(convert_element_type_219, [0, 2, 1, 3]);  convert_element_type_219 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:637 in forward, code: self.cos[None, : x_BTHD.size(-3), None, :],
        unsqueeze_103: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg52_1, 0);  arg52_1 = None
        slice_57: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_103, 1, 0, 6144);  unsqueeze_103 = None
        unsqueeze_104: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_57, 2);  slice_57 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_131: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_87, unsqueeze_104)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:638 in forward, code: self.sin[None, : x_BTHD.size(-3), None, :],
        unsqueeze_105: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg53_1, 0);  arg53_1 = None
        slice_58: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_105, 1, 0, 6144);  unsqueeze_105 = None
        unsqueeze_106: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_58, 2);  slice_58 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_132: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_88, unsqueeze_106)
        add_93: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_131, mul_132);  mul_131 = mul_132 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        neg_15: "f32[1, 6144, 1, 64]" = torch.ops.aten.neg.default(unsqueeze_106);  unsqueeze_106 = None
        mul_133: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_87, neg_15);  getitem_87 = neg_15 = None
        mul_134: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_88, unsqueeze_104);  getitem_88 = unsqueeze_104 = None
        add_94: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_133, mul_134);  mul_133 = mul_134 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        cat_15: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_93, add_94], 3);  add_93 = add_94 = None
        convert_element_type_221: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_15, torch.bfloat16);  cat_15 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:692 in forward, code: k.transpose(1, 2),
        permute_79: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(convert_element_type_221, [0, 2, 1, 3]);  convert_element_type_221 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_54: "f32[2]" = torch.ops.aten.select.int(view_7, 0, 8)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:689 in forward, code: v = lambdas[0] * v
        select_57: "f32[]" = torch.ops.aten.select.int(select_54, 0, 0);  select_54 = None
        mul_135: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(select_57, getitem_84);  select_57 = getitem_84 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:693 in forward, code: v.transpose(1, 2),
        permute_80: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(mul_135, [0, 2, 1, 3]);  mul_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:2041 in flex_attention, code: out, lse, max_scores = flex_attention_hop(
        flex_attention_7 = torch.ops.higher_order.flex_attention(permute_78, permute_79, permute_80, sdpa_score7, (6144, 6144, clamp_max_2, unsqueeze_9, clamp_max_3, unsqueeze_13, convert_element_type_6, clone_10, convert_element_type_8, clone_13, 128, 128, sdpa_mask7), 0.12, {'BACKEND': 'AUTO', 'PRESCALE_QK': False, 'ROWS_GUARANTEED_SAFE': False, 'BLOCKS_ARE_CONTIGUOUS': False, 'WRITE_DQ': True, 'OUTPUT_LOGSUMEXP': False, 'OUTPUT_MAX': False}, (), (cumsum,));  permute_78 = permute_79 = permute_80 = sdpa_score7 = sdpa_mask7 = None
        getitem_89: "bf16[1, 6, 6144, 128]" = flex_attention_7[0];  flex_attention_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_60: "f32[2]" = torch.ops.aten.select.int(view_6, 0, 9)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        select_62: "f32[]" = torch.ops.aten.select.int(select_60, 0, 0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:696 in forward, code: ).transpose(1, 2)
        permute_81: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_89, [0, 2, 1, 3]);  getitem_89 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        slice_59: "bf16[1, 6144, 12]" = torch.ops.aten.slice.Tensor(convert_element_type_210, 2, 0, 12);  convert_element_type_210 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:618 in forward, code: return F.linear(x, self.weight.type_as(x))
        expand_34: "bf16[1, 6144, 12]" = torch.ops.aten.expand.default(slice_59, [1, 6144, 12]);  slice_59 = None
        squeeze_dim_6: "bf16[6144, 12]" = torch.ops.aten.squeeze.dim(expand_34, 0);  expand_34 = None
        convert_element_type_222: "bf16[6, 12]" = torch.ops.prims.convert_element_type.default(arg54_1, torch.bfloat16);  arg54_1 = None
        permute_82: "bf16[12, 6]" = torch.ops.aten.permute.default(convert_element_type_222, [1, 0]);  convert_element_type_222 = None
        expand_35: "bf16[1, 12, 6]" = torch.ops.aten.expand.default(permute_82, [1, 12, 6]);  permute_82 = None
        squeeze_dim_7: "bf16[12, 6]" = torch.ops.aten.squeeze.dim(expand_35, 0);  expand_35 = None
        mm_default_3: "bf16[6144, 6]" = torch.ops.aten.mm.default(squeeze_dim_6, squeeze_dim_7);  squeeze_dim_6 = squeeze_dim_7 = None
        unsqueeze_default_3: "bf16[1, 6144, 6]" = torch.ops.aten.unsqueeze.default(mm_default_3, 0);  mm_default_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        sigmoid_7: "bf16[1, 6144, 6]" = torch.ops.aten.sigmoid.default(unsqueeze_default_3);  unsqueeze_default_3 = None
        view_135: "bf16[1, 6144, 6, 1]" = torch.ops.aten.reshape.default(sigmoid_7, [1, 6144, 6, 1]);  sigmoid_7 = None
        mul_136: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_81, view_135);  permute_81 = view_135 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:701 in forward, code: y = y.contiguous().view(
        view_136: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mul_136, [1, 6144, 768]);  mul_136 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:704 in forward, code: y = F.linear(y, self.qkvo_w[3].type_as(y))
        view_137: "bf16[6144, 768]" = torch.ops.aten.reshape.default(view_136, [6144, 768]);  view_136 = None
        select_58: "f32[768, 768]" = torch.ops.aten.select.int(arg51_1, 0, 3);  arg51_1 = None
        convert_element_type_225: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(select_58, torch.bfloat16);  select_58 = None
        permute_83: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_225, [1, 0]);  convert_element_type_225 = None
        mm_31: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_137, permute_83);  view_137 = permute_83 = None
        view_138: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_31, [1, 6144, 768]);  mm_31 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:751 in forward, code: x = x + self.attn(norm(x), ve, sa_lambdas, block_mask)
        add_95: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_87, view_138);  add_87 = view_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_228: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_95, torch.float32)
        pow_42: "f32[1, 6144, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_228, 2)
        mean_33: "f32[1, 6144, 1]" = torch.ops.aten.mean.dim(pow_42, [2], True);  pow_42 = None
        add_96: "f32[1, 6144, 1]" = torch.ops.aten.add.Scalar(mean_33, 1.1920928955078125e-07);  mean_33 = None
        rsqrt_33: "f32[1, 6144, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        mul_137: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_228, rsqrt_33);  convert_element_type_228 = rsqrt_33 = None
        convert_element_type_229: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_137, torch.bfloat16);  mul_137 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:723 in forward, code: x = F.linear(x, self.c_fc.T.type_as(x))
        view_139: "bf16[6144, 768]" = torch.ops.aten.reshape.default(convert_element_type_229, [6144, 768]);  convert_element_type_229 = None
        permute_84: "f32[3072, 768]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        convert_element_type_230: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(permute_84, torch.bfloat16);  permute_84 = None
        permute_85: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_230, [1, 0]);  convert_element_type_230 = None
        mm_32: "bf16[6144, 3072]" = torch.ops.aten.mm.default(view_139, permute_85);  view_139 = permute_85 = None
        view_140: "bf16[1, 6144, 3072]" = torch.ops.aten.reshape.default(mm_32, [1, 6144, 3072]);  mm_32 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:724 in forward, code: x = F.relu(
        relu_8: "bf16[1, 6144, 3072]" = torch.ops.aten.relu.default(view_140);  view_140 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:726 in forward, code: ).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        pow_43: "bf16[1, 6144, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu_8, 2);  relu_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:727 in forward, code: x = F.linear(x, self.c_proj.type_as(x))
        view_141: "bf16[6144, 3072]" = torch.ops.aten.reshape.default(pow_43, [6144, 3072]);  pow_43 = None
        convert_element_type_233: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg56_1, torch.bfloat16);  arg56_1 = None
        permute_86: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_233, [1, 0]);  convert_element_type_233 = None
        mm_33: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_141, permute_86);  view_141 = permute_86 = None
        view_142: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_33, [1, 6144, 768]);  mm_33 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:752 in forward, code: x = x + self.mlp(norm(x))
        add_97: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_95, view_142);  add_95 = view_142 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:923 in forward, code: x = x + skip_weights[i - n] * skip_connections.pop()
        select_59: "f32[]" = torch.ops.aten.select.int(slice_9, 0, 3)
        mul_138: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(select_59, add_36);  select_59 = add_36 = None
        add_98: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_97, mul_138);  add_97 = mul_138 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        mul_139: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(select_62, add_98);  select_62 = add_98 = None
        select_63: "f32[]" = torch.ops.aten.select.int(select_60, 0, 1);  select_60 = None
        mul_140: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(select_63, convert_element_type_11);  select_63 = None
        add_99: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(mul_139, mul_140);  mul_139 = mul_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_236: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_99, torch.float32)
        pow_44: "f32[1, 6144, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_236, 2)
        mean_34: "f32[1, 6144, 1]" = torch.ops.aten.mean.dim(pow_44, [2], True);  pow_44 = None
        add_100: "f32[1, 6144, 1]" = torch.ops.aten.add.Scalar(mean_34, 1.1920928955078125e-07);  mean_34 = None
        rsqrt_34: "f32[1, 6144, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
        mul_141: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_236, rsqrt_34);  convert_element_type_236 = rsqrt_34 = None
        convert_element_type_237: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_141, torch.bfloat16);  mul_141 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        view_144: "bf16[6144, 768]" = torch.ops.aten.reshape.default(convert_element_type_237, [6144, 768])
        slice_60: "f32[3, 768, 768]" = torch.ops.aten.slice.Tensor(arg57_1, 0, 0, 3)
        view_143: "f32[2304, 768]" = torch.ops.aten.reshape.default(slice_60, [2304, 768]);  slice_60 = None
        convert_element_type_238: "bf16[2304, 768]" = torch.ops.prims.convert_element_type.default(view_143, torch.bfloat16);  view_143 = None
        permute_87: "bf16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_238, [1, 0]);  convert_element_type_238 = None
        mm_34: "bf16[6144, 2304]" = torch.ops.aten.mm.default(view_144, permute_87);  view_144 = permute_87 = None
        view_145: "bf16[1, 6144, 2304]" = torch.ops.aten.reshape.default(mm_34, [1, 6144, 2304]);  mm_34 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:679 in forward, code: .view(B, T, 3 * self.num_heads, self.head_dim)
        view_146: "bf16[1, 6144, 18, 128]" = torch.ops.aten.reshape.default(view_145, [1, 6144, 18, 128]);  view_145 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:680 in forward, code: .chunk(3, dim=-2)
        split_24 = torch.ops.aten.split.Tensor(view_146, 6, -2);  view_146 = None
        getitem_92: "bf16[1, 6144, 6, 128]" = split_24[0]
        getitem_93: "bf16[1, 6144, 6, 128]" = split_24[1]
        getitem_94: "bf16[1, 6144, 6, 128]" = split_24[2];  split_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_241: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_92, torch.float32);  getitem_92 = None
        pow_45: "f32[1, 6144, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_241, 2)
        mean_35: "f32[1, 6144, 6, 1]" = torch.ops.aten.mean.dim(pow_45, [3], True);  pow_45 = None
        add_101: "f32[1, 6144, 6, 1]" = torch.ops.aten.add.Scalar(mean_35, 1.1920928955078125e-07);  mean_35 = None
        rsqrt_35: "f32[1, 6144, 6, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
        mul_142: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_241, rsqrt_35);  convert_element_type_241 = rsqrt_35 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        convert_element_type_default_5: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_142, torch.float32);  mul_142 = None
        split_25 = torch.ops.aten.split.Tensor(convert_element_type_default_5, 64, -1);  convert_element_type_default_5 = None
        getitem_95: "f32[1, 6144, 6, 64]" = split_25[0]
        getitem_96: "f32[1, 6144, 6, 64]" = split_25[1];  split_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_243: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_93, torch.float32);  getitem_93 = None
        pow_46: "f32[1, 6144, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_243, 2)
        mean_36: "f32[1, 6144, 6, 1]" = torch.ops.aten.mean.dim(pow_46, [3], True);  pow_46 = None
        add_102: "f32[1, 6144, 6, 1]" = torch.ops.aten.add.Scalar(mean_36, 1.1920928955078125e-07);  mean_36 = None
        rsqrt_36: "f32[1, 6144, 6, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
        mul_143: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_243, rsqrt_36);  convert_element_type_243 = rsqrt_36 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        convert_element_type_default_4: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_143, torch.float32);  mul_143 = None
        split_26 = torch.ops.aten.split.Tensor(convert_element_type_default_4, 64, -1);  convert_element_type_default_4 = None
        getitem_97: "f32[1, 6144, 6, 64]" = split_26[0]
        getitem_98: "f32[1, 6144, 6, 64]" = split_26[1];  split_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:2041 in flex_attention, code: out, lse, max_scores = flex_attention_hop(
        sdpa_score8 = self.sdpa_score8
        sdpa_mask8 = self.sdpa_mask8

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:637 in forward, code: self.cos[None, : x_BTHD.size(-3), None, :],
        unsqueeze_107: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg58_1, 0)
        slice_61: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_107, 1, 0, 6144);  unsqueeze_107 = None
        unsqueeze_108: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_61, 2);  slice_61 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_144: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_95, unsqueeze_108)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:638 in forward, code: self.sin[None, : x_BTHD.size(-3), None, :],
        unsqueeze_109: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg59_1, 0)
        slice_62: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_109, 1, 0, 6144);  unsqueeze_109 = None
        unsqueeze_110: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_62, 2);  slice_62 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_145: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_96, unsqueeze_110)
        add_103: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        neg_16: "f32[1, 6144, 1, 64]" = torch.ops.aten.neg.default(unsqueeze_110);  unsqueeze_110 = None
        mul_146: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_95, neg_16);  getitem_95 = neg_16 = None
        mul_147: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_96, unsqueeze_108);  getitem_96 = unsqueeze_108 = None
        add_104: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_146, mul_147);  mul_146 = mul_147 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        cat_16: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_103, add_104], 3);  add_103 = add_104 = None
        convert_element_type_246: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_16, torch.bfloat16);  cat_16 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:691 in forward, code: q.transpose(1, 2),
        permute_88: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(convert_element_type_246, [0, 2, 1, 3]);  convert_element_type_246 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:637 in forward, code: self.cos[None, : x_BTHD.size(-3), None, :],
        unsqueeze_111: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg58_1, 0);  arg58_1 = None
        slice_63: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_111, 1, 0, 6144);  unsqueeze_111 = None
        unsqueeze_112: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_63, 2);  slice_63 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_148: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_97, unsqueeze_112)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:638 in forward, code: self.sin[None, : x_BTHD.size(-3), None, :],
        unsqueeze_113: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg59_1, 0);  arg59_1 = None
        slice_64: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_113, 1, 0, 6144);  unsqueeze_113 = None
        unsqueeze_114: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_64, 2);  slice_64 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_149: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_98, unsqueeze_114)
        add_105: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_148, mul_149);  mul_148 = mul_149 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        neg_17: "f32[1, 6144, 1, 64]" = torch.ops.aten.neg.default(unsqueeze_114);  unsqueeze_114 = None
        mul_150: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_97, neg_17);  getitem_97 = neg_17 = None
        mul_151: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_98, unsqueeze_112);  getitem_98 = unsqueeze_112 = None
        add_106: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_150, mul_151);  mul_150 = mul_151 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        cat_17: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_105, add_106], 3);  add_105 = add_106 = None
        convert_element_type_248: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_17, torch.bfloat16);  cat_17 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:692 in forward, code: k.transpose(1, 2),
        permute_89: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(convert_element_type_248, [0, 2, 1, 3]);  convert_element_type_248 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_61: "f32[2]" = torch.ops.aten.select.int(view_7, 0, 9)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:685 in forward, code: v = lambdas[0] * v + lambdas[1] * ve.view_as(
        select_64: "f32[]" = torch.ops.aten.select.int(select_61, 0, 0)
        mul_152: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(select_64, getitem_94);  select_64 = getitem_94 = None
        select_65: "f32[]" = torch.ops.aten.select.int(select_61, 0, 1);  select_61 = None
        view_147: "bf16[1, 6144, 6, 128]" = torch.ops.aten.reshape.default(embedding, [1, 6144, 6, 128]);  embedding = None
        mul_153: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(select_65, view_147);  select_65 = view_147 = None
        add_107: "bf16[1, 6144, 6, 128]" = torch.ops.aten.add.Tensor(mul_152, mul_153);  mul_152 = mul_153 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:693 in forward, code: v.transpose(1, 2),
        permute_90: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(add_107, [0, 2, 1, 3]);  add_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:2041 in flex_attention, code: out, lse, max_scores = flex_attention_hop(
        flex_attention_8 = torch.ops.higher_order.flex_attention(permute_88, permute_89, permute_90, sdpa_score8, (6144, 6144, clamp_max_2, unsqueeze_9, clamp_max_3, unsqueeze_13, convert_element_type_6, clone_10, convert_element_type_8, clone_13, 128, 128, sdpa_mask8), 0.12, {'BACKEND': 'AUTO', 'PRESCALE_QK': False, 'ROWS_GUARANTEED_SAFE': False, 'BLOCKS_ARE_CONTIGUOUS': False, 'WRITE_DQ': True, 'OUTPUT_LOGSUMEXP': False, 'OUTPUT_MAX': False}, (), (cumsum,));  permute_88 = permute_89 = permute_90 = sdpa_score8 = sdpa_mask8 = None
        getitem_99: "bf16[1, 6, 6144, 128]" = flex_attention_8[0];  flex_attention_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_68: "f32[2]" = torch.ops.aten.select.int(view_6, 0, 10)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        select_70: "f32[]" = torch.ops.aten.select.int(select_68, 0, 0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:696 in forward, code: ).transpose(1, 2)
        permute_91: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_99, [0, 2, 1, 3]);  getitem_99 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        slice_65: "bf16[1, 6144, 12]" = torch.ops.aten.slice.Tensor(convert_element_type_237, 2, 0, 12);  convert_element_type_237 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:618 in forward, code: return F.linear(x, self.weight.type_as(x))
        expand_36: "bf16[1, 6144, 12]" = torch.ops.aten.expand.default(slice_65, [1, 6144, 12]);  slice_65 = None
        squeeze_dim_4: "bf16[6144, 12]" = torch.ops.aten.squeeze.dim(expand_36, 0);  expand_36 = None
        convert_element_type_249: "bf16[6, 12]" = torch.ops.prims.convert_element_type.default(arg60_1, torch.bfloat16);  arg60_1 = None
        permute_92: "bf16[12, 6]" = torch.ops.aten.permute.default(convert_element_type_249, [1, 0]);  convert_element_type_249 = None
        expand_37: "bf16[1, 12, 6]" = torch.ops.aten.expand.default(permute_92, [1, 12, 6]);  permute_92 = None
        squeeze_dim_5: "bf16[12, 6]" = torch.ops.aten.squeeze.dim(expand_37, 0);  expand_37 = None
        mm_default_2: "bf16[6144, 6]" = torch.ops.aten.mm.default(squeeze_dim_4, squeeze_dim_5);  squeeze_dim_4 = squeeze_dim_5 = None
        unsqueeze_default_2: "bf16[1, 6144, 6]" = torch.ops.aten.unsqueeze.default(mm_default_2, 0);  mm_default_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        sigmoid_8: "bf16[1, 6144, 6]" = torch.ops.aten.sigmoid.default(unsqueeze_default_2);  unsqueeze_default_2 = None
        view_152: "bf16[1, 6144, 6, 1]" = torch.ops.aten.reshape.default(sigmoid_8, [1, 6144, 6, 1]);  sigmoid_8 = None
        mul_154: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_91, view_152);  permute_91 = view_152 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:701 in forward, code: y = y.contiguous().view(
        view_153: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mul_154, [1, 6144, 768]);  mul_154 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:704 in forward, code: y = F.linear(y, self.qkvo_w[3].type_as(y))
        view_154: "bf16[6144, 768]" = torch.ops.aten.reshape.default(view_153, [6144, 768]);  view_153 = None
        select_66: "f32[768, 768]" = torch.ops.aten.select.int(arg57_1, 0, 3);  arg57_1 = None
        convert_element_type_252: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(select_66, torch.bfloat16);  select_66 = None
        permute_93: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_252, [1, 0]);  convert_element_type_252 = None
        mm_35: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_154, permute_93);  view_154 = permute_93 = None
        view_155: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_35, [1, 6144, 768]);  mm_35 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:751 in forward, code: x = x + self.attn(norm(x), ve, sa_lambdas, block_mask)
        add_108: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_99, view_155);  add_99 = view_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_255: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_108, torch.float32)
        pow_47: "f32[1, 6144, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_255, 2)
        mean_37: "f32[1, 6144, 1]" = torch.ops.aten.mean.dim(pow_47, [2], True);  pow_47 = None
        add_109: "f32[1, 6144, 1]" = torch.ops.aten.add.Scalar(mean_37, 1.1920928955078125e-07);  mean_37 = None
        rsqrt_37: "f32[1, 6144, 1]" = torch.ops.aten.rsqrt.default(add_109);  add_109 = None
        mul_155: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_255, rsqrt_37);  convert_element_type_255 = rsqrt_37 = None
        convert_element_type_256: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_155, torch.bfloat16);  mul_155 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:723 in forward, code: x = F.linear(x, self.c_fc.T.type_as(x))
        view_156: "bf16[6144, 768]" = torch.ops.aten.reshape.default(convert_element_type_256, [6144, 768]);  convert_element_type_256 = None
        permute_94: "f32[3072, 768]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        convert_element_type_257: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(permute_94, torch.bfloat16);  permute_94 = None
        permute_95: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_257, [1, 0]);  convert_element_type_257 = None
        mm_36: "bf16[6144, 3072]" = torch.ops.aten.mm.default(view_156, permute_95);  view_156 = permute_95 = None
        view_157: "bf16[1, 6144, 3072]" = torch.ops.aten.reshape.default(mm_36, [1, 6144, 3072]);  mm_36 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:724 in forward, code: x = F.relu(
        relu_9: "bf16[1, 6144, 3072]" = torch.ops.aten.relu.default(view_157);  view_157 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:726 in forward, code: ).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        pow_48: "bf16[1, 6144, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu_9, 2);  relu_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:727 in forward, code: x = F.linear(x, self.c_proj.type_as(x))
        view_158: "bf16[6144, 3072]" = torch.ops.aten.reshape.default(pow_48, [6144, 3072]);  pow_48 = None
        convert_element_type_260: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg62_1, torch.bfloat16);  arg62_1 = None
        permute_96: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_260, [1, 0]);  convert_element_type_260 = None
        mm_37: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_158, permute_96);  view_158 = permute_96 = None
        view_159: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_37, [1, 6144, 768]);  mm_37 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:752 in forward, code: x = x + self.mlp(norm(x))
        add_110: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_108, view_159);  add_108 = view_159 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:923 in forward, code: x = x + skip_weights[i - n] * skip_connections.pop()
        select_67: "f32[]" = torch.ops.aten.select.int(slice_9, 0, 4)
        mul_156: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(select_67, add_24);  select_67 = add_24 = None
        add_111: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_110, mul_156);  add_110 = mul_156 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        mul_157: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(select_70, add_111);  select_70 = add_111 = None
        select_71: "f32[]" = torch.ops.aten.select.int(select_68, 0, 1);  select_68 = None
        mul_158: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(select_71, convert_element_type_11);  select_71 = None
        add_112: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(mul_157, mul_158);  mul_157 = mul_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_263: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_112, torch.float32)
        pow_49: "f32[1, 6144, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_263, 2)
        mean_38: "f32[1, 6144, 1]" = torch.ops.aten.mean.dim(pow_49, [2], True);  pow_49 = None
        add_113: "f32[1, 6144, 1]" = torch.ops.aten.add.Scalar(mean_38, 1.1920928955078125e-07);  mean_38 = None
        rsqrt_38: "f32[1, 6144, 1]" = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
        mul_159: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_263, rsqrt_38);  convert_element_type_263 = rsqrt_38 = None
        convert_element_type_264: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_159, torch.bfloat16);  mul_159 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        view_161: "bf16[6144, 768]" = torch.ops.aten.reshape.default(convert_element_type_264, [6144, 768])
        slice_66: "f32[3, 768, 768]" = torch.ops.aten.slice.Tensor(arg63_1, 0, 0, 3)
        view_160: "f32[2304, 768]" = torch.ops.aten.reshape.default(slice_66, [2304, 768]);  slice_66 = None
        convert_element_type_265: "bf16[2304, 768]" = torch.ops.prims.convert_element_type.default(view_160, torch.bfloat16);  view_160 = None
        permute_97: "bf16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_265, [1, 0]);  convert_element_type_265 = None
        mm_38: "bf16[6144, 2304]" = torch.ops.aten.mm.default(view_161, permute_97);  view_161 = permute_97 = None
        view_162: "bf16[1, 6144, 2304]" = torch.ops.aten.reshape.default(mm_38, [1, 6144, 2304]);  mm_38 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:679 in forward, code: .view(B, T, 3 * self.num_heads, self.head_dim)
        view_163: "bf16[1, 6144, 18, 128]" = torch.ops.aten.reshape.default(view_162, [1, 6144, 18, 128]);  view_162 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:680 in forward, code: .chunk(3, dim=-2)
        split_27 = torch.ops.aten.split.Tensor(view_163, 6, -2);  view_163 = None
        getitem_102: "bf16[1, 6144, 6, 128]" = split_27[0]
        getitem_103: "bf16[1, 6144, 6, 128]" = split_27[1]
        getitem_104: "bf16[1, 6144, 6, 128]" = split_27[2];  split_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_268: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_102, torch.float32);  getitem_102 = None
        pow_50: "f32[1, 6144, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_268, 2)
        mean_39: "f32[1, 6144, 6, 1]" = torch.ops.aten.mean.dim(pow_50, [3], True);  pow_50 = None
        add_114: "f32[1, 6144, 6, 1]" = torch.ops.aten.add.Scalar(mean_39, 1.1920928955078125e-07);  mean_39 = None
        rsqrt_39: "f32[1, 6144, 6, 1]" = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
        mul_160: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_268, rsqrt_39);  convert_element_type_268 = rsqrt_39 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        convert_element_type_default_3: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_160, torch.float32);  mul_160 = None
        split_28 = torch.ops.aten.split.Tensor(convert_element_type_default_3, 64, -1);  convert_element_type_default_3 = None
        getitem_105: "f32[1, 6144, 6, 64]" = split_28[0]
        getitem_106: "f32[1, 6144, 6, 64]" = split_28[1];  split_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_270: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_103, torch.float32);  getitem_103 = None
        pow_51: "f32[1, 6144, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_270, 2)
        mean_40: "f32[1, 6144, 6, 1]" = torch.ops.aten.mean.dim(pow_51, [3], True);  pow_51 = None
        add_115: "f32[1, 6144, 6, 1]" = torch.ops.aten.add.Scalar(mean_40, 1.1920928955078125e-07);  mean_40 = None
        rsqrt_40: "f32[1, 6144, 6, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
        mul_161: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_270, rsqrt_40);  convert_element_type_270 = rsqrt_40 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        convert_element_type_default_2: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_161, torch.float32);  mul_161 = None
        split_29 = torch.ops.aten.split.Tensor(convert_element_type_default_2, 64, -1);  convert_element_type_default_2 = None
        getitem_107: "f32[1, 6144, 6, 64]" = split_29[0]
        getitem_108: "f32[1, 6144, 6, 64]" = split_29[1];  split_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:2041 in flex_attention, code: out, lse, max_scores = flex_attention_hop(
        sdpa_score9 = self.sdpa_score9
        sdpa_mask9 = self.sdpa_mask9

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:637 in forward, code: self.cos[None, : x_BTHD.size(-3), None, :],
        unsqueeze_115: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg64_1, 0)
        slice_67: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_115, 1, 0, 6144);  unsqueeze_115 = None
        unsqueeze_116: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_67, 2);  slice_67 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_162: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_105, unsqueeze_116)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:638 in forward, code: self.sin[None, : x_BTHD.size(-3), None, :],
        unsqueeze_117: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg65_1, 0)
        slice_68: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_117, 1, 0, 6144);  unsqueeze_117 = None
        unsqueeze_118: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_68, 2);  slice_68 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_163: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_106, unsqueeze_118)
        add_116: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_162, mul_163);  mul_162 = mul_163 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        neg_18: "f32[1, 6144, 1, 64]" = torch.ops.aten.neg.default(unsqueeze_118);  unsqueeze_118 = None
        mul_164: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_105, neg_18);  getitem_105 = neg_18 = None
        mul_165: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_106, unsqueeze_116);  getitem_106 = unsqueeze_116 = None
        add_117: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_164, mul_165);  mul_164 = mul_165 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        cat_18: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_116, add_117], 3);  add_116 = add_117 = None
        convert_element_type_273: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_18, torch.bfloat16);  cat_18 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:691 in forward, code: q.transpose(1, 2),
        permute_98: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(convert_element_type_273, [0, 2, 1, 3]);  convert_element_type_273 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:637 in forward, code: self.cos[None, : x_BTHD.size(-3), None, :],
        unsqueeze_119: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg64_1, 0);  arg64_1 = None
        slice_69: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_119, 1, 0, 6144);  unsqueeze_119 = None
        unsqueeze_120: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_69, 2);  slice_69 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_166: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_107, unsqueeze_120)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:638 in forward, code: self.sin[None, : x_BTHD.size(-3), None, :],
        unsqueeze_121: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg65_1, 0);  arg65_1 = None
        slice_70: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_121, 1, 0, 6144);  unsqueeze_121 = None
        unsqueeze_122: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_70, 2);  slice_70 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_167: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_108, unsqueeze_122)
        add_118: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_166, mul_167);  mul_166 = mul_167 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        neg_19: "f32[1, 6144, 1, 64]" = torch.ops.aten.neg.default(unsqueeze_122);  unsqueeze_122 = None
        mul_168: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_107, neg_19);  getitem_107 = neg_19 = None
        mul_169: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_108, unsqueeze_120);  getitem_108 = unsqueeze_120 = None
        add_119: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_168, mul_169);  mul_168 = mul_169 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        cat_19: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_118, add_119], 3);  add_118 = add_119 = None
        convert_element_type_275: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_19, torch.bfloat16);  cat_19 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:692 in forward, code: k.transpose(1, 2),
        permute_99: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(convert_element_type_275, [0, 2, 1, 3]);  convert_element_type_275 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_69: "f32[2]" = torch.ops.aten.select.int(view_7, 0, 10)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:685 in forward, code: v = lambdas[0] * v + lambdas[1] * ve.view_as(
        select_72: "f32[]" = torch.ops.aten.select.int(select_69, 0, 0)
        mul_170: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(select_72, getitem_104);  select_72 = getitem_104 = None
        select_73: "f32[]" = torch.ops.aten.select.int(select_69, 0, 1);  select_69 = None
        view_164: "bf16[1, 6144, 6, 128]" = torch.ops.aten.reshape.default(embedding_1, [1, 6144, 6, 128]);  embedding_1 = None
        mul_171: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(select_73, view_164);  select_73 = view_164 = None
        add_120: "bf16[1, 6144, 6, 128]" = torch.ops.aten.add.Tensor(mul_170, mul_171);  mul_170 = mul_171 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:693 in forward, code: v.transpose(1, 2),
        permute_100: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(add_120, [0, 2, 1, 3]);  add_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:2041 in flex_attention, code: out, lse, max_scores = flex_attention_hop(
        flex_attention_9 = torch.ops.higher_order.flex_attention(permute_98, permute_99, permute_100, sdpa_score9, (6144, 6144, clamp_max_2, unsqueeze_9, clamp_max_3, unsqueeze_13, convert_element_type_6, clone_10, convert_element_type_8, clone_13, 128, 128, sdpa_mask9), 0.12, {'BACKEND': 'AUTO', 'PRESCALE_QK': False, 'ROWS_GUARANTEED_SAFE': False, 'BLOCKS_ARE_CONTIGUOUS': False, 'WRITE_DQ': True, 'OUTPUT_LOGSUMEXP': False, 'OUTPUT_MAX': False}, (), (cumsum,));  permute_98 = permute_99 = permute_100 = sdpa_score9 = clamp_max_2 = clamp_max_3 = convert_element_type_6 = clone_10 = convert_element_type_8 = clone_13 = sdpa_mask9 = None
        getitem_109: "bf16[1, 6, 6144, 128]" = flex_attention_9[0];  flex_attention_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_76: "f32[2]" = torch.ops.aten.select.int(view_6, 0, 11);  view_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        select_78: "f32[]" = torch.ops.aten.select.int(select_76, 0, 0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:696 in forward, code: ).transpose(1, 2)
        permute_101: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_109, [0, 2, 1, 3]);  getitem_109 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        slice_71: "bf16[1, 6144, 12]" = torch.ops.aten.slice.Tensor(convert_element_type_264, 2, 0, 12);  convert_element_type_264 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:618 in forward, code: return F.linear(x, self.weight.type_as(x))
        expand_38: "bf16[1, 6144, 12]" = torch.ops.aten.expand.default(slice_71, [1, 6144, 12]);  slice_71 = None
        squeeze_dim_2: "bf16[6144, 12]" = torch.ops.aten.squeeze.dim(expand_38, 0);  expand_38 = None
        convert_element_type_276: "bf16[6, 12]" = torch.ops.prims.convert_element_type.default(arg66_1, torch.bfloat16);  arg66_1 = None
        permute_102: "bf16[12, 6]" = torch.ops.aten.permute.default(convert_element_type_276, [1, 0]);  convert_element_type_276 = None
        expand_39: "bf16[1, 12, 6]" = torch.ops.aten.expand.default(permute_102, [1, 12, 6]);  permute_102 = None
        squeeze_dim_3: "bf16[12, 6]" = torch.ops.aten.squeeze.dim(expand_39, 0);  expand_39 = None
        mm_default_1: "bf16[6144, 6]" = torch.ops.aten.mm.default(squeeze_dim_2, squeeze_dim_3);  squeeze_dim_2 = squeeze_dim_3 = None
        unsqueeze_default_1: "bf16[1, 6144, 6]" = torch.ops.aten.unsqueeze.default(mm_default_1, 0);  mm_default_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        sigmoid_9: "bf16[1, 6144, 6]" = torch.ops.aten.sigmoid.default(unsqueeze_default_1);  unsqueeze_default_1 = None
        view_169: "bf16[1, 6144, 6, 1]" = torch.ops.aten.reshape.default(sigmoid_9, [1, 6144, 6, 1]);  sigmoid_9 = None
        mul_172: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_101, view_169);  permute_101 = view_169 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:701 in forward, code: y = y.contiguous().view(
        view_170: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mul_172, [1, 6144, 768]);  mul_172 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:704 in forward, code: y = F.linear(y, self.qkvo_w[3].type_as(y))
        view_171: "bf16[6144, 768]" = torch.ops.aten.reshape.default(view_170, [6144, 768]);  view_170 = None
        select_74: "f32[768, 768]" = torch.ops.aten.select.int(arg63_1, 0, 3);  arg63_1 = None
        convert_element_type_279: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(select_74, torch.bfloat16);  select_74 = None
        permute_103: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_279, [1, 0]);  convert_element_type_279 = None
        mm_39: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_171, permute_103);  view_171 = permute_103 = None
        view_172: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_39, [1, 6144, 768]);  mm_39 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:751 in forward, code: x = x + self.attn(norm(x), ve, sa_lambdas, block_mask)
        add_121: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_112, view_172);  add_112 = view_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_282: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_121, torch.float32)
        pow_52: "f32[1, 6144, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_282, 2)
        mean_41: "f32[1, 6144, 1]" = torch.ops.aten.mean.dim(pow_52, [2], True);  pow_52 = None
        add_122: "f32[1, 6144, 1]" = torch.ops.aten.add.Scalar(mean_41, 1.1920928955078125e-07);  mean_41 = None
        rsqrt_41: "f32[1, 6144, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
        mul_173: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_282, rsqrt_41);  convert_element_type_282 = rsqrt_41 = None
        convert_element_type_283: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_173, torch.bfloat16);  mul_173 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:723 in forward, code: x = F.linear(x, self.c_fc.T.type_as(x))
        view_173: "bf16[6144, 768]" = torch.ops.aten.reshape.default(convert_element_type_283, [6144, 768]);  convert_element_type_283 = None
        permute_104: "f32[3072, 768]" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        convert_element_type_284: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(permute_104, torch.bfloat16);  permute_104 = None
        permute_105: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_284, [1, 0]);  convert_element_type_284 = None
        mm_40: "bf16[6144, 3072]" = torch.ops.aten.mm.default(view_173, permute_105);  view_173 = permute_105 = None
        view_174: "bf16[1, 6144, 3072]" = torch.ops.aten.reshape.default(mm_40, [1, 6144, 3072]);  mm_40 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:724 in forward, code: x = F.relu(
        relu_10: "bf16[1, 6144, 3072]" = torch.ops.aten.relu.default(view_174);  view_174 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:726 in forward, code: ).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        pow_53: "bf16[1, 6144, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu_10, 2);  relu_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:727 in forward, code: x = F.linear(x, self.c_proj.type_as(x))
        view_175: "bf16[6144, 3072]" = torch.ops.aten.reshape.default(pow_53, [6144, 3072]);  pow_53 = None
        convert_element_type_287: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg68_1, torch.bfloat16);  arg68_1 = None
        permute_106: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_287, [1, 0]);  convert_element_type_287 = None
        mm_41: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_175, permute_106);  view_175 = permute_106 = None
        view_176: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_41, [1, 6144, 768]);  mm_41 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:752 in forward, code: x = x + self.mlp(norm(x))
        add_123: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_121, view_176);  add_121 = view_176 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:923 in forward, code: x = x + skip_weights[i - n] * skip_connections.pop()
        select_75: "f32[]" = torch.ops.aten.select.int(slice_9, 0, 5);  slice_9 = None
        mul_174: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(select_75, add_12);  select_75 = add_12 = None
        add_124: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_123, mul_174);  add_123 = mul_174 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        mul_175: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(select_78, add_124);  select_78 = add_124 = None
        select_79: "f32[]" = torch.ops.aten.select.int(select_76, 0, 1);  select_76 = None
        mul_176: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(select_79, convert_element_type_11);  select_79 = convert_element_type_11 = None
        add_125: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(mul_175, mul_176);  mul_175 = mul_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_290: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_125, torch.float32)
        pow_54: "f32[1, 6144, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_290, 2)
        mean_42: "f32[1, 6144, 1]" = torch.ops.aten.mean.dim(pow_54, [2], True);  pow_54 = None
        add_126: "f32[1, 6144, 1]" = torch.ops.aten.add.Scalar(mean_42, 1.1920928955078125e-07);  mean_42 = None
        rsqrt_42: "f32[1, 6144, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
        mul_177: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_290, rsqrt_42);  convert_element_type_290 = rsqrt_42 = None
        convert_element_type_291: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_177, torch.bfloat16);  mul_177 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        view_178: "bf16[6144, 768]" = torch.ops.aten.reshape.default(convert_element_type_291, [6144, 768])
        slice_72: "f32[3, 768, 768]" = torch.ops.aten.slice.Tensor(arg69_1, 0, 0, 3)
        view_177: "f32[2304, 768]" = torch.ops.aten.reshape.default(slice_72, [2304, 768]);  slice_72 = None
        convert_element_type_292: "bf16[2304, 768]" = torch.ops.prims.convert_element_type.default(view_177, torch.bfloat16);  view_177 = None
        permute_107: "bf16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_292, [1, 0]);  convert_element_type_292 = None
        mm_42: "bf16[6144, 2304]" = torch.ops.aten.mm.default(view_178, permute_107);  view_178 = permute_107 = None
        view_179: "bf16[1, 6144, 2304]" = torch.ops.aten.reshape.default(mm_42, [1, 6144, 2304]);  mm_42 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:679 in forward, code: .view(B, T, 3 * self.num_heads, self.head_dim)
        view_180: "bf16[1, 6144, 18, 128]" = torch.ops.aten.reshape.default(view_179, [1, 6144, 18, 128]);  view_179 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:680 in forward, code: .chunk(3, dim=-2)
        split_30 = torch.ops.aten.split.Tensor(view_180, 6, -2);  view_180 = None
        getitem_112: "bf16[1, 6144, 6, 128]" = split_30[0]
        getitem_113: "bf16[1, 6144, 6, 128]" = split_30[1]
        getitem_114: "bf16[1, 6144, 6, 128]" = split_30[2];  split_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_295: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_112, torch.float32);  getitem_112 = None
        pow_55: "f32[1, 6144, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_295, 2)
        mean_43: "f32[1, 6144, 6, 1]" = torch.ops.aten.mean.dim(pow_55, [3], True);  pow_55 = None
        add_127: "f32[1, 6144, 6, 1]" = torch.ops.aten.add.Scalar(mean_43, 1.1920928955078125e-07);  mean_43 = None
        rsqrt_43: "f32[1, 6144, 6, 1]" = torch.ops.aten.rsqrt.default(add_127);  add_127 = None
        mul_178: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_295, rsqrt_43);  convert_element_type_295 = rsqrt_43 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        convert_element_type_default_1: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_178, torch.float32);  mul_178 = None
        split_31 = torch.ops.aten.split.Tensor(convert_element_type_default_1, 64, -1);  convert_element_type_default_1 = None
        getitem_115: "f32[1, 6144, 6, 64]" = split_31[0]
        getitem_116: "f32[1, 6144, 6, 64]" = split_31[1];  split_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_297: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_113, torch.float32);  getitem_113 = None
        pow_56: "f32[1, 6144, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_297, 2)
        mean_44: "f32[1, 6144, 6, 1]" = torch.ops.aten.mean.dim(pow_56, [3], True);  pow_56 = None
        add_128: "f32[1, 6144, 6, 1]" = torch.ops.aten.add.Scalar(mean_44, 1.1920928955078125e-07);  mean_44 = None
        rsqrt_44: "f32[1, 6144, 6, 1]" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
        mul_179: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_297, rsqrt_44);  convert_element_type_297 = rsqrt_44 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        convert_element_type_default: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_179, torch.float32);  mul_179 = None
        split_32 = torch.ops.aten.split.Tensor(convert_element_type_default, 64, -1);  convert_element_type_default = None
        getitem_117: "f32[1, 6144, 6, 64]" = split_32[0]
        getitem_118: "f32[1, 6144, 6, 64]" = split_32[1];  split_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:2041 in flex_attention, code: out, lse, max_scores = flex_attention_hop(
        sdpa_score10 = self.sdpa_score10
        sdpa_mask10 = self.sdpa_mask10

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:637 in forward, code: self.cos[None, : x_BTHD.size(-3), None, :],
        unsqueeze_123: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg70_1, 0)
        slice_73: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_123, 1, 0, 6144);  unsqueeze_123 = None
        unsqueeze_124: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_73, 2);  slice_73 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_180: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_115, unsqueeze_124)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:638 in forward, code: self.sin[None, : x_BTHD.size(-3), None, :],
        unsqueeze_125: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg71_1, 0)
        slice_74: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_125, 1, 0, 6144);  unsqueeze_125 = None
        unsqueeze_126: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_74, 2);  slice_74 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_181: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_116, unsqueeze_126)
        add_129: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_180, mul_181);  mul_180 = mul_181 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        neg_20: "f32[1, 6144, 1, 64]" = torch.ops.aten.neg.default(unsqueeze_126);  unsqueeze_126 = None
        mul_182: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_115, neg_20);  getitem_115 = neg_20 = None
        mul_183: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_116, unsqueeze_124);  getitem_116 = unsqueeze_124 = None
        add_130: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_182, mul_183);  mul_182 = mul_183 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        cat_20: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_129, add_130], 3);  add_129 = add_130 = None
        convert_element_type_300: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_20, torch.bfloat16);  cat_20 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:691 in forward, code: q.transpose(1, 2),
        permute_108: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(convert_element_type_300, [0, 2, 1, 3]);  convert_element_type_300 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:637 in forward, code: self.cos[None, : x_BTHD.size(-3), None, :],
        unsqueeze_127: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg70_1, 0);  arg70_1 = None
        slice_75: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_127, 1, 0, 6144);  unsqueeze_127 = None
        unsqueeze_128: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_75, 2);  slice_75 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_184: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_117, unsqueeze_128)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:638 in forward, code: self.sin[None, : x_BTHD.size(-3), None, :],
        unsqueeze_129: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg71_1, 0);  arg71_1 = None
        slice_76: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_129, 1, 0, 6144);  unsqueeze_129 = None
        unsqueeze_130: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_76, 2);  slice_76 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_185: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_118, unsqueeze_130)
        add_131: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_184, mul_185);  mul_184 = mul_185 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        neg_21: "f32[1, 6144, 1, 64]" = torch.ops.aten.neg.default(unsqueeze_130);  unsqueeze_130 = None
        mul_186: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_117, neg_21);  getitem_117 = neg_21 = None
        mul_187: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_118, unsqueeze_128);  getitem_118 = unsqueeze_128 = None
        add_132: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_186, mul_187);  mul_186 = mul_187 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        cat_21: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_131, add_132], 3);  add_131 = add_132 = None
        convert_element_type_302: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_21, torch.bfloat16);  cat_21 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:692 in forward, code: k.transpose(1, 2),
        permute_109: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(convert_element_type_302, [0, 2, 1, 3]);  convert_element_type_302 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_77: "f32[2]" = torch.ops.aten.select.int(view_7, 0, 11);  view_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:685 in forward, code: v = lambdas[0] * v + lambdas[1] * ve.view_as(
        select_80: "f32[]" = torch.ops.aten.select.int(select_77, 0, 0)
        mul_188: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(select_80, getitem_114);  select_80 = getitem_114 = None
        select_81: "f32[]" = torch.ops.aten.select.int(select_77, 0, 1);  select_77 = None
        view_181: "bf16[1, 6144, 6, 128]" = torch.ops.aten.reshape.default(embedding_2, [1, 6144, 6, 128]);  embedding_2 = None
        mul_189: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(select_81, view_181);  select_81 = view_181 = None
        add_133: "bf16[1, 6144, 6, 128]" = torch.ops.aten.add.Tensor(mul_188, mul_189);  mul_188 = mul_189 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:693 in forward, code: v.transpose(1, 2),
        permute_110: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(add_133, [0, 2, 1, 3]);  add_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:2041 in flex_attention, code: out, lse, max_scores = flex_attention_hop(
        flex_attention_10 = torch.ops.higher_order.flex_attention(permute_108, permute_109, permute_110, sdpa_score10, (6144, 6144, clamp_max, unsqueeze_9, clamp_max_1, unsqueeze_13, convert_element_type_2, clone_4, convert_element_type_4, clone_7, 128, 128, sdpa_mask10), 0.12, {'BACKEND': 'AUTO', 'PRESCALE_QK': False, 'ROWS_GUARANTEED_SAFE': False, 'BLOCKS_ARE_CONTIGUOUS': False, 'WRITE_DQ': True, 'OUTPUT_LOGSUMEXP': False, 'OUTPUT_MAX': False}, (), (cumsum,));  permute_108 = permute_109 = permute_110 = sdpa_score10 = clamp_max = unsqueeze_9 = clamp_max_1 = unsqueeze_13 = convert_element_type_2 = clone_4 = convert_element_type_4 = clone_7 = sdpa_mask10 = cumsum = None
        getitem_119: "bf16[1, 6, 6144, 128]" = flex_attention_10[0];  flex_attention_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:932 in forward, code: loss = F.cross_entropy(
        ne_1: "b8[6144]" = torch.ops.aten.ne.Scalar(arg76_1, -100)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:696 in forward, code: ).transpose(1, 2)
        permute_111: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_119, [0, 2, 1, 3]);  getitem_119 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        slice_77: "bf16[1, 6144, 12]" = torch.ops.aten.slice.Tensor(convert_element_type_291, 2, 0, 12);  convert_element_type_291 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:618 in forward, code: return F.linear(x, self.weight.type_as(x))
        expand_40: "bf16[1, 6144, 12]" = torch.ops.aten.expand.default(slice_77, [1, 6144, 12]);  slice_77 = None
        squeeze_dim: "bf16[6144, 12]" = torch.ops.aten.squeeze.dim(expand_40, 0);  expand_40 = None
        convert_element_type_303: "bf16[6, 12]" = torch.ops.prims.convert_element_type.default(arg72_1, torch.bfloat16);  arg72_1 = None
        permute_112: "bf16[12, 6]" = torch.ops.aten.permute.default(convert_element_type_303, [1, 0]);  convert_element_type_303 = None
        expand_41: "bf16[1, 12, 6]" = torch.ops.aten.expand.default(permute_112, [1, 12, 6]);  permute_112 = None
        squeeze_dim_1: "bf16[12, 6]" = torch.ops.aten.squeeze.dim(expand_41, 0);  expand_41 = None
        mm_default: "bf16[6144, 6]" = torch.ops.aten.mm.default(squeeze_dim, squeeze_dim_1);  squeeze_dim = squeeze_dim_1 = None
        unsqueeze_default: "bf16[1, 6144, 6]" = torch.ops.aten.unsqueeze.default(mm_default, 0);  mm_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        sigmoid_10: "bf16[1, 6144, 6]" = torch.ops.aten.sigmoid.default(unsqueeze_default);  unsqueeze_default = None
        view_186: "bf16[1, 6144, 6, 1]" = torch.ops.aten.reshape.default(sigmoid_10, [1, 6144, 6, 1]);  sigmoid_10 = None
        mul_190: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_111, view_186);  permute_111 = view_186 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:701 in forward, code: y = y.contiguous().view(
        view_187: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mul_190, [1, 6144, 768]);  mul_190 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:704 in forward, code: y = F.linear(y, self.qkvo_w[3].type_as(y))
        view_188: "bf16[6144, 768]" = torch.ops.aten.reshape.default(view_187, [6144, 768]);  view_187 = None
        select_82: "f32[768, 768]" = torch.ops.aten.select.int(arg69_1, 0, 3);  arg69_1 = None
        convert_element_type_306: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(select_82, torch.bfloat16);  select_82 = None
        permute_113: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_306, [1, 0]);  convert_element_type_306 = None
        mm_43: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_188, permute_113);  view_188 = permute_113 = None
        view_189: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_43, [1, 6144, 768]);  mm_43 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:751 in forward, code: x = x + self.attn(norm(x), ve, sa_lambdas, block_mask)
        add_134: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_125, view_189);  add_125 = view_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_309: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_134, torch.float32)
        pow_57: "f32[1, 6144, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_309, 2)
        mean_45: "f32[1, 6144, 1]" = torch.ops.aten.mean.dim(pow_57, [2], True);  pow_57 = None
        add_135: "f32[1, 6144, 1]" = torch.ops.aten.add.Scalar(mean_45, 1.1920928955078125e-07);  mean_45 = None
        rsqrt_45: "f32[1, 6144, 1]" = torch.ops.aten.rsqrt.default(add_135);  add_135 = None
        mul_191: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_309, rsqrt_45);  convert_element_type_309 = rsqrt_45 = None
        convert_element_type_310: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_191, torch.bfloat16);  mul_191 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:723 in forward, code: x = F.linear(x, self.c_fc.T.type_as(x))
        view_190: "bf16[6144, 768]" = torch.ops.aten.reshape.default(convert_element_type_310, [6144, 768]);  convert_element_type_310 = None
        permute_114: "f32[3072, 768]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        convert_element_type_311: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(permute_114, torch.bfloat16);  permute_114 = None
        permute_115: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_311, [1, 0]);  convert_element_type_311 = None
        mm_44: "bf16[6144, 3072]" = torch.ops.aten.mm.default(view_190, permute_115);  view_190 = permute_115 = None
        view_191: "bf16[1, 6144, 3072]" = torch.ops.aten.reshape.default(mm_44, [1, 6144, 3072]);  mm_44 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:724 in forward, code: x = F.relu(
        relu_11: "bf16[1, 6144, 3072]" = torch.ops.aten.relu.default(view_191);  view_191 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:726 in forward, code: ).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        pow_58: "bf16[1, 6144, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu_11, 2);  relu_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:727 in forward, code: x = F.linear(x, self.c_proj.type_as(x))
        view_192: "bf16[6144, 3072]" = torch.ops.aten.reshape.default(pow_58, [6144, 3072]);  pow_58 = None
        convert_element_type_314: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg74_1, torch.bfloat16);  arg74_1 = None
        permute_116: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_314, [1, 0]);  convert_element_type_314 = None
        mm_45: "bf16[6144, 768]" = torch.ops.aten.mm.default(view_192, permute_116);  view_192 = permute_116 = None
        view_193: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_45, [1, 6144, 768]);  mm_45 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:752 in forward, code: x = x + self.mlp(norm(x))
        add_136: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_134, view_193);  add_134 = view_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_317: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_136, torch.float32);  add_136 = None
        pow_59: "f32[1, 6144, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_317, 2)
        mean_46: "f32[1, 6144, 1]" = torch.ops.aten.mean.dim(pow_59, [2], True);  pow_59 = None
        add_137: "f32[1, 6144, 1]" = torch.ops.aten.add.Scalar(mean_46, 1.1920928955078125e-07);  mean_46 = None
        rsqrt_46: "f32[1, 6144, 1]" = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
        mul_192: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_317, rsqrt_46);  convert_element_type_317 = rsqrt_46 = None
        convert_element_type_318: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_192, torch.bfloat16);  mul_192 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:618 in forward, code: return F.linear(x, self.weight.type_as(x))
        view_194: "bf16[6144, 768]" = torch.ops.aten.reshape.default(convert_element_type_318, [6144, 768]);  convert_element_type_318 = None
        convert_element_type_319: "bf16[50304, 768]" = torch.ops.prims.convert_element_type.default(arg75_1, torch.bfloat16);  arg75_1 = None
        permute_117: "bf16[768, 50304]" = torch.ops.aten.permute.default(convert_element_type_319, [1, 0]);  convert_element_type_319 = None
        mm_46: "bf16[6144, 50304]" = torch.ops.aten.mm.default(view_194, permute_117);  view_194 = permute_117 = None
        view_195: "bf16[1, 6144, 50304]" = torch.ops.aten.reshape.default(mm_46, [1, 6144, 50304]);  mm_46 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:929 in forward, code: logits = self.lm_head(x).float()
        convert_element_type_322: "f32[1, 6144, 50304]" = torch.ops.prims.convert_element_type.default(view_195, torch.float32);  view_195 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:931 in forward, code: logits = 30 * torch.sigmoid(logits / 7.5)
        div_1: "f32[1, 6144, 50304]" = torch.ops.aten.div.Tensor(convert_element_type_322, 7.5);  convert_element_type_322 = None
        sigmoid_11: "f32[1, 6144, 50304]" = torch.ops.aten.sigmoid.default(div_1);  div_1 = None
        mul_193: "f32[1, 6144, 50304]" = torch.ops.aten.mul.Tensor(sigmoid_11, 30);  sigmoid_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:933 in forward, code: logits.view(-1, logits.size(-1)),
        view_196: "f32[6144, 50304]" = torch.ops.aten.reshape.default(mul_193, [-1, 50304]);  mul_193 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:932 in forward, code: loss = F.cross_entropy(
        amax: "f32[6144, 1]" = torch.ops.aten.amax.default(view_196, [1], True)
        sub_4: "f32[6144, 50304]" = torch.ops.aten.sub.Tensor(view_196, amax);  view_196 = amax = None
        exp: "f32[6144, 50304]" = torch.ops.aten.exp.default(sub_4)
        sum_7: "f32[6144, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[6144, 1]" = torch.ops.aten.log.default(sum_7);  sum_7 = None
        sub_5: "f32[6144, 50304]" = torch.ops.aten.sub.Tensor(sub_4, log);  sub_4 = log = None
        ne: "b8[6144]" = torch.ops.aten.ne.Scalar(arg76_1, -100)
        full_default_12: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4: "i64[6144]" = torch.ops.aten.where.self(ne, arg76_1, full_default_12);  ne = full_default_12 = None
        unsqueeze_131: "i64[6144, 1]" = torch.ops.aten.unsqueeze.default(where_4, 1);  where_4 = None
        gather: "f32[6144, 1]" = torch.ops.aten.gather.default(sub_5, 1, unsqueeze_131);  sub_5 = unsqueeze_131 = None
        squeeze: "f32[6144]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg_22: "f32[6144]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_13: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_5: "f32[6144]" = torch.ops.aten.where.self(ne_1, neg_22, full_default_13);  ne_1 = neg_22 = full_default_13 = None
        sum_9: "f32[]" = torch.ops.aten.sum.default(where_5);  where_5 = None
        ne_2: "b8[6144]" = torch.ops.aten.ne.Scalar(arg76_1, -100);  arg76_1 = None
        sum_8: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_323: "f32[]" = torch.ops.prims.convert_element_type.default(sum_8, torch.float32);  sum_8 = None
        div_2: "f32[]" = torch.ops.aten.div.Tensor(sum_9, convert_element_type_323);  sum_9 = convert_element_type_323 = None
        return (div_2,)

    class sdpa_score0(torch.nn.Module):
        def forward(self, arg0_1: "bf16[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]"):
            return arg0_1

    class sdpa_mask0(torch.nn.Module):
        def forward(self, arg0_1: "i32[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i64[6144]"):
            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:826 in document_causal, code: causal_mask = q_idx >= kv_idx
            ge: "b8[]" = torch.ops.aten.ge.Tensor(arg2_1, arg3_1)

            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:827 in document_causal, code: document_mask = docs[q_idx] == docs[kv_idx]
            index: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg2_1]);  arg2_1 = None
            index_1: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg3_1]);  arg4_1 = arg3_1 = None
            eq: "b8[]" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None

            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:828 in document_causal, code: return causal_mask & document_mask
            bitwise_and: "b8[]" = torch.ops.aten.bitwise_and.Tensor(ge, eq);  ge = eq = None
            return bitwise_and

    class sdpa_score1(torch.nn.Module):
        def forward(self, arg0_1: "bf16[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]"):
            return arg0_1

    class sdpa_mask1(torch.nn.Module):
        def forward(self, arg0_1: "i32[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i64[6144]"):
            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:826 in document_causal, code: causal_mask = q_idx >= kv_idx
            ge: "b8[]" = torch.ops.aten.ge.Tensor(arg2_1, arg3_1)

            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:827 in document_causal, code: document_mask = docs[q_idx] == docs[kv_idx]
            index: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg2_1]);  arg2_1 = None
            index_1: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg3_1]);  arg4_1 = arg3_1 = None
            eq: "b8[]" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None

            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:828 in document_causal, code: return causal_mask & document_mask
            bitwise_and: "b8[]" = torch.ops.aten.bitwise_and.Tensor(ge, eq);  ge = eq = None
            return bitwise_and

    class sdpa_score2(torch.nn.Module):
        def forward(self, arg0_1: "bf16[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]"):
            return arg0_1

    class sdpa_mask2(torch.nn.Module):
        def forward(self, arg0_1: "i32[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i64[6144]"):
            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:826 in document_causal, code: causal_mask = q_idx >= kv_idx
            ge: "b8[]" = torch.ops.aten.ge.Tensor(arg2_1, arg3_1)

            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:827 in document_causal, code: document_mask = docs[q_idx] == docs[kv_idx]
            index: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg2_1]);  arg2_1 = None
            index_1: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg3_1]);  arg4_1 = arg3_1 = None
            eq: "b8[]" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None

            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:828 in document_causal, code: return causal_mask & document_mask
            bitwise_and: "b8[]" = torch.ops.aten.bitwise_and.Tensor(ge, eq);  ge = eq = None
            return bitwise_and

    class sdpa_score3(torch.nn.Module):
        def forward(self, arg0_1: "bf16[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]"):
            return arg0_1

    class sdpa_mask3(torch.nn.Module):
        def forward(self, arg0_1: "i32[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i64[6144]"):
            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:826 in document_causal, code: causal_mask = q_idx >= kv_idx
            ge: "b8[]" = torch.ops.aten.ge.Tensor(arg2_1, arg3_1)

            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:827 in document_causal, code: document_mask = docs[q_idx] == docs[kv_idx]
            index: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg2_1]);  arg2_1 = None
            index_1: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg3_1]);  arg4_1 = arg3_1 = None
            eq: "b8[]" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None

            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:828 in document_causal, code: return causal_mask & document_mask
            bitwise_and: "b8[]" = torch.ops.aten.bitwise_and.Tensor(ge, eq);  ge = eq = None
            return bitwise_and

    class sdpa_score4(torch.nn.Module):
        def forward(self, arg0_1: "bf16[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]"):
            return arg0_1

    class sdpa_mask4(torch.nn.Module):
        def forward(self, arg0_1: "i32[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i64[6144]"):
            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:826 in document_causal, code: causal_mask = q_idx >= kv_idx
            ge: "b8[]" = torch.ops.aten.ge.Tensor(arg2_1, arg3_1)

            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:827 in document_causal, code: document_mask = docs[q_idx] == docs[kv_idx]
            index: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg2_1]);  arg2_1 = None
            index_1: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg3_1]);  arg4_1 = arg3_1 = None
            eq: "b8[]" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None

            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:828 in document_causal, code: return causal_mask & document_mask
            bitwise_and: "b8[]" = torch.ops.aten.bitwise_and.Tensor(ge, eq);  ge = eq = None
            return bitwise_and

    class sdpa_score5(torch.nn.Module):
        def forward(self, arg0_1: "bf16[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]"):
            return arg0_1

    class sdpa_mask5(torch.nn.Module):
        def forward(self, arg0_1: "i32[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i64[6144]"):
            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:826 in document_causal, code: causal_mask = q_idx >= kv_idx
            ge: "b8[]" = torch.ops.aten.ge.Tensor(arg2_1, arg3_1)

            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:827 in document_causal, code: document_mask = docs[q_idx] == docs[kv_idx]
            index: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg2_1]);  arg2_1 = None
            index_1: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg3_1]);  arg4_1 = arg3_1 = None
            eq: "b8[]" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None

            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:828 in document_causal, code: return causal_mask & document_mask
            bitwise_and: "b8[]" = torch.ops.aten.bitwise_and.Tensor(ge, eq);  ge = eq = None
            return bitwise_and

    class sdpa_score6(torch.nn.Module):
        def forward(self, arg0_1: "bf16[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]"):
            return arg0_1

    class sdpa_mask6(torch.nn.Module):
        def forward(self, arg0_1: "i32[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i64[6144]"):
            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:826 in document_causal, code: causal_mask = q_idx >= kv_idx
            ge: "b8[]" = torch.ops.aten.ge.Tensor(arg2_1, arg3_1)

            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:827 in document_causal, code: document_mask = docs[q_idx] == docs[kv_idx]
            index: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg2_1]);  arg2_1 = None
            index_1: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg3_1]);  arg4_1 = arg3_1 = None
            eq: "b8[]" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None

            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:828 in document_causal, code: return causal_mask & document_mask
            bitwise_and: "b8[]" = torch.ops.aten.bitwise_and.Tensor(ge, eq);  ge = eq = None
            return bitwise_and

    class sdpa_score7(torch.nn.Module):
        def forward(self, arg0_1: "bf16[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]"):
            return arg0_1

    class sdpa_mask7(torch.nn.Module):
        def forward(self, arg0_1: "i32[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i64[6144]"):
            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:826 in document_causal, code: causal_mask = q_idx >= kv_idx
            ge: "b8[]" = torch.ops.aten.ge.Tensor(arg2_1, arg3_1)

            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:827 in document_causal, code: document_mask = docs[q_idx] == docs[kv_idx]
            index: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg2_1]);  arg2_1 = None
            index_1: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg3_1]);  arg4_1 = arg3_1 = None
            eq: "b8[]" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None

            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:828 in document_causal, code: return causal_mask & document_mask
            bitwise_and: "b8[]" = torch.ops.aten.bitwise_and.Tensor(ge, eq);  ge = eq = None
            return bitwise_and

    class sdpa_score8(torch.nn.Module):
        def forward(self, arg0_1: "bf16[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]"):
            return arg0_1

    class sdpa_mask8(torch.nn.Module):
        def forward(self, arg0_1: "i32[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i64[6144]"):
            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:826 in document_causal, code: causal_mask = q_idx >= kv_idx
            ge: "b8[]" = torch.ops.aten.ge.Tensor(arg2_1, arg3_1)

            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:827 in document_causal, code: document_mask = docs[q_idx] == docs[kv_idx]
            index: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg2_1]);  arg2_1 = None
            index_1: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg3_1]);  arg4_1 = arg3_1 = None
            eq: "b8[]" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None

            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:828 in document_causal, code: return causal_mask & document_mask
            bitwise_and: "b8[]" = torch.ops.aten.bitwise_and.Tensor(ge, eq);  ge = eq = None
            return bitwise_and

    class sdpa_score9(torch.nn.Module):
        def forward(self, arg0_1: "bf16[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]"):
            return arg0_1

    class sdpa_mask9(torch.nn.Module):
        def forward(self, arg0_1: "i32[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i64[6144]"):
            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:826 in document_causal, code: causal_mask = q_idx >= kv_idx
            ge: "b8[]" = torch.ops.aten.ge.Tensor(arg2_1, arg3_1)

            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:827 in document_causal, code: document_mask = docs[q_idx] == docs[kv_idx]
            index: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg2_1]);  arg2_1 = None
            index_1: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg3_1]);  arg4_1 = arg3_1 = None
            eq: "b8[]" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None

            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:828 in document_causal, code: return causal_mask & document_mask
            bitwise_and: "b8[]" = torch.ops.aten.bitwise_and.Tensor(ge, eq);  ge = eq = None
            return bitwise_and

    class sdpa_score10(torch.nn.Module):
        def forward(self, arg0_1: "bf16[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]"):
            return arg0_1

    class sdpa_mask10(torch.nn.Module):
        def forward(self, arg0_1: "i32[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i64[6144]"):
            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:826 in document_causal, code: causal_mask = q_idx >= kv_idx
            ge: "b8[]" = torch.ops.aten.ge.Tensor(arg2_1, arg3_1)

            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:827 in document_causal, code: document_mask = docs[q_idx] == docs[kv_idx]
            index: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg2_1]);  arg2_1 = None
            index_1: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg3_1]);  arg4_1 = arg3_1 = None
            eq: "b8[]" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None

            # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:828 in document_causal, code: return causal_mask & document_mask
            bitwise_and: "b8[]" = torch.ops.aten.bitwise_and.Tensor(ge, eq);  ge = eq = None
            return bitwise_and
