"""
Standalone repro captured via capture_hook.
Label: hf_MegatronBertForCausalLM_train_000
Pattern hash: 170cb6646cc0
Shape hash: fd361de1
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
_shapes_config = "(T([29056, 1024], f32), T([16, 512], i64, gen=Index(29056)), T([2, 1024], f32), T([512, 1024], f32), T([1, 512], i64, gen=Index(512)), T([1024], f32), T([1024], f32), S([8192, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, arg2_1: "f32[29056, 1024]", arg1_1: "i64[16, 512]", arg4_1: "f32[2, 1024]", arg5_1: "f32[512, 1024]", arg3_1: "i64[1, 512]", arg6_1: "f32[1024]", arg7_1: "f32[1024]", _shape_param_0):
        # No stacktrace found for following nodes
        full_default: "i64[16, 512]" = torch.ops.aten.full.default([16, 512], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        embedding_default: "f32[16, 512, 1024]" = torch.ops.aten.embedding.default(arg2_1, arg1_1, 0);  arg2_1 = arg1_1 = None
        embedding_default_1: "f32[16, 512, 1024]" = torch.ops.aten.embedding.default(arg4_1, full_default);  arg4_1 = full_default = None
        add_tensor: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(embedding_default, embedding_default_1);  embedding_default = embedding_default_1 = None
        embedding_default_2: "f32[1, 512, 1024]" = torch.ops.aten.embedding.default(arg5_1, arg3_1);  arg5_1 = arg3_1 = None
        add_tensor_1: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_tensor, embedding_default_2);  add_tensor = embedding_default_2 = None
        inductor_seeds_default: "i64[49]" = torch.ops.prims.inductor_seeds.default(49, device(type='cuda', index=0))
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_scalar, add_tensor_1);  gt_scalar = add_tensor_1 = None
        mul_tensor_1: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        var_mean_correction = torch.ops.aten.var_mean.correction(mul_tensor_1, [2], correction = 0, keepdim = True)
        getitem: "f32[16, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[16, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_2: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_tensor_1, getitem_1);  mul_tensor_1 = getitem_1 = None
        mul_tensor_2: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        mul_tensor_3: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_2, arg6_1);  mul_tensor_2 = arg6_1 = None
        add_tensor_3: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_3, arg7_1);  mul_tensor_3 = arg7_1 = None
        view_default: "f32[8192, 1024]" = torch.ops.aten.view.default(add_tensor_3, _shape_param_0);  add_tensor_3 = _shape_param_0 = None
        div_tensor: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_default, 1024);  rsqrt_default = None
        return (view_default, div_tensor)



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
