"""
Standalone repro captured via capture_hook.
Label: hf_XLNetLMHeadModel_train
Pattern hash: 98a1a7e50644
Shape hash: 85616f43
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([8192, 4096], f32), T([512, 16, 4096], b8), T([8192, 4096], f32), T([4096, 1024], f32), S([512, 16, 4096]), S([512, 16, 4096]), S([8192, 4096]))"

class Repro(torch.nn.Module):
    def forward(self, mm_94: "f32[8192, 4096]", gt_4: "b8[512, 16, 4096]", addmm: "f32[8192, 4096]", primals_12: "f32[4096, 1024]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        reshape_default: "f32[512, 16, 4096]" = torch.ops.aten.reshape.default(mm_94, _shape_param_0);  mm_94 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:301 in forward, code: output = self.dropout(output)
        convert_element_type_default: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(gt_4, torch.float32);  gt_4 = None
        mul_tensor: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(reshape_default, mul_tensor);  reshape_default = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        reshape_default_1: "f32[512, 16, 4096]" = torch.ops.aten.reshape.default(addmm, _shape_param_1);  addmm = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_tensor_2: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(reshape_default_1, 0.7071067811865476)
        erf_default: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_tensor_2);  mul_tensor_2 = None
        add_tensor: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_3: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(add_tensor, 0.5);  add_tensor = None
        mul_tensor_4: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(reshape_default_1, reshape_default_1)
        mul_tensor_5: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_tensor_4, -0.5);  mul_tensor_4 = None
        exp_default: "f32[512, 16, 4096]" = torch.ops.aten.exp.default(mul_tensor_5);  mul_tensor_5 = None
        mul_tensor_6: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(exp_default, 0.3989422804014327);  exp_default = None
        mul_tensor_7: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(reshape_default_1, mul_tensor_6);  reshape_default_1 = mul_tensor_6 = None
        add_tensor_1: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(mul_tensor_3, mul_tensor_7);  mul_tensor_3 = mul_tensor_7 = None
        mul_tensor_8: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_tensor_1, add_tensor_1);  mul_tensor_1 = add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        reshape_default_2: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_tensor_8, _shape_param_2);  mul_tensor_8 = _shape_param_2 = None
        permute_default: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
        permute_default_1: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_2, permute_default_1)


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
