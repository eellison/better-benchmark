"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_T5_base_train
Pattern hash: 3bab072ccdb7
Shape hash: 31c8e7bf
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
_shapes_config = "(T([8, 1024], i64), T([32128, 768], f32), T([124], i64), T([768], f32), S([8192, 768]))"

class Repro(torch.nn.Module):
    def forward(self, primals_101: "i64[8, 1024]", primals_2: "f32[32128, 768]", inductor_seeds_default: "i64[124]", primals_102: "f32[768]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:592 in _shift_right, code: shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        full_default: "i64[8, 1024]" = torch.ops.aten.full.default([8, 1024], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:593 in _shift_right, code: shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        slice_tensor: "i64[8, 1023]" = torch.ops.aten.slice.Tensor(primals_101, 1, 0, -1);  primals_101 = None
        clone_default: "i64[8, 1023]" = torch.ops.aten.clone.default(slice_tensor);  slice_tensor = None
        slice_tensor_1: "i64[8, 1023]" = torch.ops.aten.slice.Tensor(full_default, 1, 1, 9223372036854775807)
        copy_default: "i64[8, 1023]" = torch.ops.aten.copy.default(slice_tensor_1, clone_default);  slice_tensor_1 = clone_default = None
        slice_scatter_default: "i64[8, 1024]" = torch.ops.aten.slice_scatter.default(full_default, copy_default, 1, 1, 9223372036854775807);  full_default = copy_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:594 in _shift_right, code: shifted_input_ids[..., 0] = decoder_start_token_id
        full_default_1: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        select_int: "i64[8]" = torch.ops.aten.select.int(slice_scatter_default, 1, 0)
        copy_default_1: "i64[8]" = torch.ops.aten.copy.default(select_int, full_default_1);  select_int = full_default_1 = None
        select_scatter_default: "i64[8, 1024]" = torch.ops.aten.select_scatter.default(slice_scatter_default, copy_default_1, 1, 0);  slice_scatter_default = copy_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:599 in _shift_right, code: shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        eq_scalar: "b8[8, 1024]" = torch.ops.aten.eq.Scalar(select_scatter_default, -100)
        full_default_2: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[8, 1024]" = torch.ops.aten.where.self(eq_scalar, full_default_2, select_scatter_default);  eq_scalar = full_default_2 = select_scatter_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:669 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        embedding_default: "f32[8, 1024, 768]" = torch.ops.aten.embedding.default(primals_2, where_self);  primals_2 = where_self = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:721 in forward, code: hidden_states = self.dropout(inputs_embeds)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 50);  inductor_seeds_default = None
        inductor_random_default: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_scalar, embedding_default);  gt_scalar = embedding_default = None
        mul_tensor_1: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_tensor_scalar: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(mul_tensor_1, 2)
        mean_dim: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_tensor: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor_2: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_1, rsqrt_default);  mul_tensor_1 = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_3: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_102, mul_tensor_2);  primals_102 = mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_tensor_3, _shape_param_0);  mul_tensor_3 = _shape_param_0 = None
        return reshape_default



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
