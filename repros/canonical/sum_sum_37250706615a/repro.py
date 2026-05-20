"""
Standalone repro captured via capture_hook.
Label: hf_XLNetLMHeadModel_train
Pattern hash: 37250706615a
Shape hash: c4971add
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([8192, 1024], f32), T([512, 16, 1024], f32), T([8192, 1024], f32), T([8192, 1024], f32), T([1024], f32), T([512, 16, 1024], f32), T([512, 16, 1], f32), T([512, 16, 1024], b8), T([1024, 4096], f32), S([512, 16, 1024, 1, 1]), S([512, 16, 1024, 1, 1]), S([512, 16, 1024, 1, 1]), S([8192, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, mm_default_13: "f32[8192, 1024]", mul_1101: "f32[512, 16, 1024]", mm_default_11: "f32[8192, 1024]", mm_default_9: "f32[8192, 1024]", primals_16: "f32[1024]", mul_20: "f32[512, 16, 1024]", div_73: "f32[512, 16, 1]", gt_5: "b8[512, 16, 1024]", primals_14: "f32[1024, 4096]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_default: "f32[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_13, 0);  mm_default_13 = None
        reshape_default: "f32[512, 16, 1024, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default, _shape_param_0);  unsqueeze_default = _shape_param_0 = None
        squeeze_dim: "f32[512, 16, 1024, 1]" = torch.ops.aten.squeeze.dim(reshape_default, 4);  reshape_default = None
        squeeze_dim_1: "f32[512, 16, 1024]" = torch.ops.aten.squeeze.dim(squeeze_dim, 3);  squeeze_dim = None
        add_tensor: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_1101, squeeze_dim_1);  mul_1101 = squeeze_dim_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_default_1: "f32[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_11, 0);  mm_default_11 = None
        reshape_default_1: "f32[512, 16, 1024, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_1, _shape_param_1);  unsqueeze_default_1 = _shape_param_1 = None
        squeeze_dim_2: "f32[512, 16, 1024, 1]" = torch.ops.aten.squeeze.dim(reshape_default_1, 4);  reshape_default_1 = None
        squeeze_dim_3: "f32[512, 16, 1024]" = torch.ops.aten.squeeze.dim(squeeze_dim_2, 3);  squeeze_dim_2 = None
        add_tensor_1: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(add_tensor, squeeze_dim_3);  add_tensor = squeeze_dim_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_default_2: "f32[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_9, 0);  mm_default_9 = None
        reshape_default_2: "f32[512, 16, 1024, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default_2, _shape_param_2);  unsqueeze_default_2 = _shape_param_2 = None
        squeeze_dim_4: "f32[512, 16, 1024, 1]" = torch.ops.aten.squeeze.dim(reshape_default_2, 4);  reshape_default_2 = None
        squeeze_dim_5: "f32[512, 16, 1024]" = torch.ops.aten.squeeze.dim(squeeze_dim_4, 3);  squeeze_dim_4 = None
        add_tensor_2: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(add_tensor_1, squeeze_dim_5);  add_tensor_1 = squeeze_dim_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        mul_tensor: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_tensor_2, primals_16);  add_tensor_2 = primals_16 = None
        mul_tensor_1: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, 1024)
        sum_dim_int_list: "f32[512, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_20);  mul_tensor = None
        sum_dim_int_list_1: "f32[512, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_20, sum_dim_int_list_1);  mul_20 = sum_dim_int_list_1 = None
        sub_tensor: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(div_73, sub_tensor_1);  div_73 = sub_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:303 in forward, code: output = self.dropout(output)
        convert_element_type_default: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(gt_5, torch.float32);  gt_5 = None
        mul_tensor_5: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_6: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        reshape_default_3: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_tensor_6, _shape_param_3);  mul_tensor_6 = _shape_param_3 = None
        permute_default: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_14, [1, 0]);  primals_14 = None
        permute_default_1: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_3, permute_default_1)


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
