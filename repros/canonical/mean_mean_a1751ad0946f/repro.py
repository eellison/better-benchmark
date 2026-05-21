"""
Standalone repro captured via capture_hook.
Label: torchbench_modded_nanogpt_train
Pattern hash: a1751ad0946f
Shape hash: e89d4c03
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
_shapes_config = "(T([50304, 768], bf16), T([6144], i32, gen=Index(50304)), T([64], f32), S([-1, 2]), S([6144, 768]))"

class Repro(torch.nn.Module):
    def forward(self, primals_6: "bf16[50304, 768]", primals_1: "i32[6144]", primals_7: "f32[64]", _shape_param_0, _shape_param_1):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:909 in forward, code: x = x0 = norm(self.embed(input_seq)[None])  # use of norm here by @Grad62304977
        embedding_default: "bf16[6144, 768]" = torch.ops.aten.embedding.default(primals_6, primals_1);  primals_6 = primals_1 = None
        unsqueeze_default: "bf16[1, 6144, 768]" = torch.ops.aten.unsqueeze.default(embedding_default, 0);  embedding_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_default: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(unsqueeze_default, torch.float32);  unsqueeze_default = None
        pow_tensor_scalar: "f32[1, 6144, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 2)
        mean_dim: "f32[1, 6144, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [2], True);  pow_tensor_scalar = None
        add_scalar: "f32[1, 6144, 1]" = torch.ops.aten.add.Scalar(mean_dim, 1.1920928955078125e-07);  mean_dim = None
        rsqrt_default: "f32[1, 6144, 1]" = torch.ops.aten.rsqrt.default(add_scalar);  add_scalar = None
        mul_tensor: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, rsqrt_default);  convert_element_type_default = rsqrt_default = None
        convert_element_type_default_1: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bfloat16);  mul_tensor = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:914 in forward, code: lambdas = self.scalars[1 * len(self.blocks) : 3 * len(self.blocks)].view(-1, 2)
        slice_tensor: "f32[24]" = torch.ops.aten.slice.Tensor(primals_7, 0, 12, 36);  primals_7 = None
        reshape_default: "f32[12, 2]" = torch.ops.aten.reshape.default(slice_tensor, _shape_param_0);  slice_tensor = _shape_param_0 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_int: "f32[2]" = torch.ops.aten.select.int(reshape_default, 0, 0);  reshape_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        select_int_1: "f32[]" = torch.ops.aten.select.int(select_int, 0, 0)
        mul_tensor_1: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(select_int_1, convert_element_type_default_1);  select_int_1 = None
        select_int_2: "f32[]" = torch.ops.aten.select.int(select_int, 0, 1);  select_int = None
        mul_tensor_2: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(select_int_2, convert_element_type_default_1);  select_int_2 = convert_element_type_default_1 = None
        add_tensor: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_2);  mul_tensor_1 = mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_default_2: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32);  add_tensor = None
        pow_tensor_scalar_1: "f32[1, 6144, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_2, 2)
        mean_dim_1: "f32[1, 6144, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar_1, [2], True);  pow_tensor_scalar_1 = None
        add_scalar_1: "f32[1, 6144, 1]" = torch.ops.aten.add.Scalar(mean_dim_1, 1.1920928955078125e-07);  mean_dim_1 = None
        rsqrt_default_1: "f32[1, 6144, 1]" = torch.ops.aten.rsqrt.default(add_scalar_1);  add_scalar_1 = None
        mul_tensor_3: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, rsqrt_default_1);  convert_element_type_default_2 = rsqrt_default_1 = None
        convert_element_type_default_3: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_tensor_3, torch.bfloat16);  mul_tensor_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        reshape_default_1: "bf16[6144, 768]" = torch.ops.aten.reshape.default(convert_element_type_default_3, _shape_param_1);  convert_element_type_default_3 = _shape_param_1 = None
        return reshape_default_1



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
