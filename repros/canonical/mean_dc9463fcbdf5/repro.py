"""
Standalone repro captured via capture_hook.
Label: torchbench_modded_nanogpt_infer
Pattern hash: dc9463fcbdf5
Shape hash: aa07a35c
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
_shapes_config = "(T([12, 2], f32), T([64], f32), T([6144, 768], bf16), T([1, 6144, 768], bf16), T([1, 6144, 768], bf16), T([4, 768, 768], f32), S([1, 6144, 768]), S([6144, 768]), S([2304, 768]))"

class Repro(torch.nn.Module):
    def forward(self, view_6: "f32[12, 2]", arg6_1: "f32[64]", mm_23: "bf16[6144, 768]", add_67: "bf16[1, 6144, 768]", convert_element_type_11: "bf16[1, 6144, 768]", arg43_1: "f32[4, 768, 768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_int: "f32[2]" = torch.ops.aten.select.int(view_6, 0, 6);  view_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        select_int_1: "f32[]" = torch.ops.aten.select.int(select_int, 0, 0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:913 in forward, code: skip_weights = self.scalars[: (len(self.blocks) // 2)]
        slice_tensor: "f32[6]" = torch.ops.aten.slice.Tensor(arg6_1, 0, 0, 6);  arg6_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:923 in forward, code: x = x + skip_weights[i - n] * skip_connections.pop()
        select_int_2: "f32[]" = torch.ops.aten.select.int(slice_tensor, 0, 0);  slice_tensor = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:727 in forward, code: x = F.linear(x, self.c_proj.type_as(x))
        reshape_default: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_23, _shape_param_0);  mm_23 = _shape_param_0 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:752 in forward, code: x = x + self.mlp(norm(x))
        add_tensor: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_67, reshape_default);  add_67 = reshape_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:923 in forward, code: x = x + skip_weights[i - n] * skip_connections.pop()
        mul_tensor: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(select_int_2, add_tensor);  select_int_2 = None
        add_tensor_1: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_tensor, mul_tensor);  add_tensor = mul_tensor = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        mul_tensor_1: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(select_int_1, add_tensor_1);  select_int_1 = add_tensor_1 = None
        select_int_3: "f32[]" = torch.ops.aten.select.int(select_int, 0, 1);  select_int = None
        mul_tensor_2: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(select_int_3, convert_element_type_11);  select_int_3 = convert_element_type_11 = None
        add_tensor_2: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_2);  mul_tensor_1 = mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_default: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float32);  add_tensor_2 = None
        pow_tensor_scalar: "f32[1, 6144, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 2)
        mean_dim: "f32[1, 6144, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [2], True);  pow_tensor_scalar = None
        add_scalar: "f32[1, 6144, 1]" = torch.ops.aten.add.Scalar(mean_dim, 1.1920928955078125e-07);  mean_dim = None
        rsqrt_default: "f32[1, 6144, 1]" = torch.ops.aten.rsqrt.default(add_scalar);  add_scalar = None
        mul_tensor_3: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, rsqrt_default);  convert_element_type_default = rsqrt_default = None
        convert_element_type_default_1: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_tensor_3, torch.bfloat16);  mul_tensor_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        reshape_default_1: "bf16[6144, 768]" = torch.ops.aten.reshape.default(convert_element_type_default_1, _shape_param_1);  convert_element_type_default_1 = _shape_param_1 = None
        slice_tensor_1: "f32[3, 768, 768]" = torch.ops.aten.slice.Tensor(arg43_1, 0, 0, 3);  arg43_1 = None
        reshape_default_2: "f32[2304, 768]" = torch.ops.aten.reshape.default(slice_tensor_1, _shape_param_2);  slice_tensor_1 = _shape_param_2 = None
        convert_element_type_default_2: "bf16[2304, 768]" = torch.ops.prims.convert_element_type.default(reshape_default_2, torch.bfloat16);  reshape_default_2 = None
        permute_default: "bf16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_default_2, [1, 0]);  convert_element_type_default_2 = None
        return (reshape_default_1, permute_default)



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
