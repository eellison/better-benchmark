"""
Standalone repro captured via capture_hook.
Label: hf_ElectraForCausalLM_train_000
Pattern hash: 0a0eec8ae477
Shape hash: c6c13b6f
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([1, 512], i64, gen=Index(2)), T([1, 512], i64, gen=Index(512)), T([30522, 128], f32), T([64, 512], i64, gen=Index(30522)), T([2, 128], f32), T([512, 128], f32), T([128], f32), T([128], f32), S([64, 512]), S([32768, 128]))"

class Repro(torch.nn.Module):
    def forward(self, arg3_1: "i64[1, 512]", arg2_1: "i64[1, 512]", arg4_1: "f32[30522, 128]", arg1_1: "i64[64, 512]", arg5_1: "f32[2, 128]", arg6_1: "f32[512, 128]", arg7_1: "f32[128]", arg8_1: "f32[128]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        expand_default: "i64[1, 512]" = torch.ops.aten.expand.default(arg3_1, [1, -1]);  arg3_1 = None
        gather_default: "i64[1, 512]" = torch.ops.aten.gather.default(expand_default, 1, arg2_1);  expand_default = None
        expand_default_1: "i64[64, 512]" = torch.ops.aten.expand.default(gather_default, _shape_param_0);  gather_default = _shape_param_0 = None
        embedding_default: "f32[64, 512, 128]" = torch.ops.aten.embedding.default(arg4_1, arg1_1, 0);  arg4_1 = arg1_1 = None
        embedding_default_1: "f32[64, 512, 128]" = torch.ops.aten.embedding.default(arg5_1, expand_default_1);  arg5_1 = expand_default_1 = None
        add_tensor: "f32[64, 512, 128]" = torch.ops.aten.add.Tensor(embedding_default, embedding_default_1);  embedding_default = embedding_default_1 = None
        embedding_default_2: "f32[1, 512, 128]" = torch.ops.aten.embedding.default(arg6_1, arg2_1);  arg6_1 = arg2_1 = None
        add_tensor_1: "f32[64, 512, 128]" = torch.ops.aten.add.Tensor(add_tensor, embedding_default_2);  add_tensor = embedding_default_2 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_1, [2], correction = 0, keepdim = True)
        getitem: "f32[64, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[64, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_2: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor: "f32[64, 512, 128]" = torch.ops.aten.sub.Tensor(add_tensor_1, getitem_1);  add_tensor_1 = getitem_1 = None
        mul_tensor: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        mul_tensor_1: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor, arg7_1);  mul_tensor = arg7_1 = None
        add_tensor_3: "f32[64, 512, 128]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg8_1);  mul_tensor_1 = arg8_1 = None
        inductor_seeds_default: "i64[37]" = torch.ops.prims.inductor_seeds.default(37, device(type='cuda', index=0))
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[64, 512, 128]" = torch.ops.prims.inductor_random.default([64, 512, 128], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[64, 512, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor_2: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(gt_scalar, add_tensor_3);  gt_scalar = add_tensor_3 = None
        mul_tensor_3: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 1.1111111111111112);  mul_tensor_2 = None
        view_default: "f32[32768, 128]" = torch.ops.aten.view.default(mul_tensor_3, _shape_param_1);  mul_tensor_3 = _shape_param_1 = None
        div_tensor: "f32[64, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_default, 128);  rsqrt_default = None
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
