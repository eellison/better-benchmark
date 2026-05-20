"""
Standalone repro captured via capture_hook.
Label: hf_XLNetLMHeadModel_train
Pattern hash: aa8797d8a21a
Shape hash: b303d80f
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([8192, 1024], f32), T([99], i64, gen=Index(99)), T([512, 16, 1024], f32), T([1024], f32), T([1024], f32), T([1024, 16, 64], f32), T([1024, 16, 64], f32), T([1024, 16, 64], f32), S([512, 16, 1024]), S([1, 8192, 1024]), S([1, 1024, 1024]), S([1, 1024, 1024]), S([1, 1024, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_45: "f32[8192, 1024]", inductor_seeds_default: "i64[99]", add_248: "f32[512, 16, 1024]", primals_346: "f32[1024]", primals_347: "f32[1024]", primals_348: "f32[1024, 16, 64]", primals_349: "f32[1024, 16, 64]", primals_351: "f32[1024, 16, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        reshape_default: "f32[512, 16, 1024]" = torch.ops.aten.reshape.default(addmm_45, _shape_param_0);  addmm_45 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:303 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 93);  inductor_seeds_default = None
        inductor_random_default: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_scalar, reshape_default);  gt_scalar = reshape_default = None
        mul_tensor_1: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_tensor: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_1, add_248);  mul_tensor_1 = add_248 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[512, 16, 1]" = var_mean_correction[0]
        getitem_1: "f32[512, 16, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        mul_tensor_2: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_3: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_2, primals_346);  mul_tensor_2 = primals_346 = None
        add_tensor_2: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_3, primals_347);  mul_tensor_3 = primals_347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_default: "f32[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_tensor_2, 3);  add_tensor_2 = None
        unsqueeze_default_1: "f32[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 4);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_348, 3);  primals_348 = None
        unsqueeze_default_3: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, 4);  unsqueeze_default_2 = None
        reshape_default_1: "f32[1, 8192, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_1, _shape_param_1);  unsqueeze_default_1 = _shape_param_1 = None
        reshape_default_2: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_3, _shape_param_2);  unsqueeze_default_3 = _shape_param_2 = None
        squeeze_dim: "f32[8192, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_1, 0);  reshape_default_1 = None
        squeeze_dim_1: "f32[1024, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_2, 0);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_default_4: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_349, 3);  primals_349 = None
        unsqueeze_default_5: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 4);  unsqueeze_default_4 = None
        reshape_default_3: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_5, _shape_param_3);  unsqueeze_default_5 = _shape_param_3 = None
        squeeze_dim_2: "f32[1024, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_3, 0);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        unsqueeze_default_6: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_351, 3);  primals_351 = None
        unsqueeze_default_7: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 4);  unsqueeze_default_6 = None
        reshape_default_4: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_7, _shape_param_4);  unsqueeze_default_7 = _shape_param_4 = None
        squeeze_dim_3: "f32[1024, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_4, 0);  reshape_default_4 = None
        return (squeeze_dim, squeeze_dim_1, squeeze_dim_2, squeeze_dim_3)


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
