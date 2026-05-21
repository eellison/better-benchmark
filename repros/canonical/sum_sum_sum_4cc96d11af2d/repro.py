"""
Standalone repro captured via capture_hook.
Label: vllm_Qwen_Qwen3-30B-A3B_001
Pattern hash: 4cc96d11af2d
Shape hash: 8e4e8d5d
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
_shapes_config = "(T([16384, 2048], bf16), T([16384, 2048], bf16), T([16384], bf16), T([16384], i64, gen=Index(16384)), T([2048, 8], f32), T([2048, 1], f32), T([2048, 128], f32), T([2048, 8], i64, gen=Index(128)), T([2048, 128], bf16), T([2048, 1], f32), T([2048, 1], f32), S([2048, 8]), S([2048, 8]))"

class Repro(torch.nn.Module):
    def forward(self, where_9: "bf16[16384, 2048]", arg79_1: "bf16[16384, 2048]", full_4: "bf16[16384]", arg72_1: "i64[16384]", arg71_1: "f32[2048, 8]", arg70_1: "f32[2048, 1]", full_6: "f32[2048, 128]", arg69_1: "i64[2048, 8]", arg66_1: "bf16[2048, 128]", arg67_1: "f32[2048, 1]", arg68_1: "f32[2048, 1]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        mul_tensor: "bf16[16384, 2048]" = torch.ops.aten.mul.Tensor(where_9, arg79_1);  where_9 = arg79_1 = None
        sum_dim_int_list: "bf16[16384, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [1], True);  mul_tensor = None
        squeeze_dim: "bf16[16384]" = torch.ops.aten.squeeze.dim(sum_dim_int_list, -1);  sum_dim_int_list = None
        index_put_default: "bf16[16384]" = torch.ops.aten.index_put.default(full_4, [arg72_1], squeeze_dim, True);  full_4 = arg72_1 = squeeze_dim = None
        view_default: "bf16[2048, 8]" = torch.ops.aten.view.default(index_put_default, _shape_param_0);  index_put_default = _shape_param_0 = None
        convert_element_type_default: "f32[2048, 8]" = torch.ops.prims.convert_element_type.default(view_default, torch.float32);  view_default = None
        div_tensor: "f32[2048, 8]" = torch.ops.aten.div.Tensor(arg71_1, arg70_1);  arg71_1 = None
        neg_default: "f32[2048, 8]" = torch.ops.aten.neg.default(convert_element_type_default)
        mul_tensor_1: "f32[2048, 8]" = torch.ops.aten.mul.Tensor(neg_default, div_tensor);  neg_default = div_tensor = None
        div_tensor_1: "f32[2048, 8]" = torch.ops.aten.div.Tensor(convert_element_type_default, arg70_1);  convert_element_type_default = arg70_1 = None
        sum_dim_int_list_1: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [1], True);  mul_tensor_1 = None
        expand_default: "f32[2048, 8]" = torch.ops.aten.expand.default(sum_dim_int_list_1, _shape_param_1);  sum_dim_int_list_1 = _shape_param_1 = None
        add_tensor: "f32[2048, 8]" = torch.ops.aten.add.Tensor(div_tensor_1, expand_default);  div_tensor_1 = expand_default = None
        scatter_src: "f32[2048, 128]" = torch.ops.aten.scatter.src(full_6, -1, arg69_1, add_tensor);  full_6 = arg69_1 = add_tensor = None
        convert_element_type_default_1: "f32[2048, 128]" = torch.ops.prims.convert_element_type.default(arg66_1, torch.float32);  arg66_1 = None
        sub_tensor: "f32[2048, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_default_1, arg67_1);  convert_element_type_default_1 = arg67_1 = None
        exp_default: "f32[2048, 128]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        div_tensor_2: "f32[2048, 128]" = torch.ops.aten.div.Tensor(exp_default, arg68_1);  exp_default = arg68_1 = None
        mul_tensor_2: "f32[2048, 128]" = torch.ops.aten.mul.Tensor(scatter_src, div_tensor_2);  scatter_src = None
        sum_dim_int_list_2: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [-1], True)
        neg_default_1: "f32[2048, 128]" = torch.ops.aten.neg.default(div_tensor_2);  div_tensor_2 = None
        fma_default: "f32[2048, 128]" = torch.ops.prims.fma.default(neg_default_1, sum_dim_int_list_2, mul_tensor_2);  neg_default_1 = sum_dim_int_list_2 = mul_tensor_2 = None
        convert_element_type_default_2: "bf16[2048, 128]" = torch.ops.prims.convert_element_type.default(fma_default, torch.bfloat16);  fma_default = None
        return convert_element_type_default_2



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
