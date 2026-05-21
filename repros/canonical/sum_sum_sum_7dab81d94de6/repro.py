"""
Standalone repro captured via capture_hook.
Label: torchbench_nanogpt_train
Pattern hash: 7dab81d94de6
Shape hash: 4e28ca44
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
_shapes_config = "(T([1, 50304], f32), T([1, 768], f32), T([64, 768], f32), T([768], f32), T([1, 64, 768], f32), T([1, 64, 1], f32), T([1, 64, 768], f32), T([1, 64], i64, gen=Index(1024)), T([1, 64], i64, gen=Index(50304)), S([1, 64, 768]))"

class Repro(torch.nn.Module):
    def forward(self, view_146: "f32[1, 50304]", view_144: "f32[1, 768]", mm_96: "f32[64, 768]", primals_4: "f32[768]", mul: "f32[1, 64, 768]", div_24: "f32[1, 64, 1]", add_145: "f32[1, 64, 768]", unsqueeze: "i64[1, 64]", primals_1: "i64[1, 64]", _shape_param_0):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:239 in forward, code: logits = self.lm_head(
        permute_default: "f32[50304, 1]" = torch.ops.aten.permute.default(view_146, [1, 0]);  view_146 = None
        mul_tensor: "f32[50304, 768]" = torch.ops.aten.mul.Tensor(permute_default, view_144);  permute_default = view_144 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:83 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        reshape_default: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(mm_96, _shape_param_0);  mm_96 = _shape_param_0 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        mul_tensor_1: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(reshape_default, primals_4);  primals_4 = None
        mul_tensor_2: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_1, 768)
        sum_dim_int_list: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [2], True)
        mul_tensor_3: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_1, mul);  mul_tensor_1 = None
        sum_dim_int_list_1: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True);  mul_tensor_3 = None
        mul_tensor_4: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_2, sum_dim_int_list);  mul_tensor_2 = sum_dim_int_list = None
        sub_tensor_1: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_4);  sub_tensor = mul_tensor_4 = None
        mul_tensor_5: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(div_24, sub_tensor_1);  div_24 = sub_tensor_1 = None
        mul_tensor_6: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(reshape_default, mul);  mul = None
        sum_dim_int_list_2: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1]);  mul_tensor_6 = None
        sum_dim_int_list_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(reshape_default, [0, 1]);  reshape_default = None
        add_tensor: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_145, mul_tensor_5);  add_145 = mul_tensor_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:230 in forward, code: pos_emb = self.transformer.wpe(
        eq_scalar: "b8[1, 64]" = torch.ops.aten.eq.Scalar(unsqueeze, -1)
        unsqueeze_default: "b8[1, 64, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[1, 64, 768]" = torch.ops.aten.where.self(unsqueeze_default, full_default, add_tensor);  unsqueeze_default = None
        full_default_1: "f32[1024, 768]" = torch.ops.aten.full.default([1024, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[1024, 768]" = torch.ops.aten.index_put.default(full_default_1, [unsqueeze], where_self, True);  full_default_1 = unsqueeze = where_self = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:229 in forward, code: tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        eq_scalar_1: "b8[1, 64]" = torch.ops.aten.eq.Scalar(primals_1, -1)
        unsqueeze_default_1: "b8[1, 64, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_1: "f32[1, 64, 768]" = torch.ops.aten.where.self(unsqueeze_default_1, full_default, add_tensor);  unsqueeze_default_1 = full_default = add_tensor = None
        full_default_2: "f32[50304, 768]" = torch.ops.aten.full.default([50304, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_1: "f32[50304, 768]" = torch.ops.aten.index_put.default(full_default_2, [primals_1], where_self_1, True);  full_default_2 = primals_1 = where_self_1 = None
        add_tensor_1: "f32[50304, 768]" = torch.ops.aten.add.Tensor(mul_tensor, index_put_default_1);  mul_tensor = index_put_default_1 = None
        return (add_tensor_1, index_put_default, sum_dim_int_list_3, sum_dim_int_list_2)



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
