"""
Standalone repro captured via capture_hook.
Label: torchbench_BERT_pytorch_train
Pattern hash: 98a1a7e50644
Shape hash: 638520ea
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
_shapes_config = "(T([16384, 3072], f32), T([128, 128, 3072], b8), T([16384, 3072], f32), T([3072, 768], f32), S([128, 128, 3072]), S([128, 128, 3072]), S([16384, 3072]))"

class Repro(torch.nn.Module):
    def forward(self, mm_136: "f32[16384, 3072]", gt_4: "b8[128, 128, 3072]", addmm_4: "f32[16384, 3072]", primals_18: "f32[3072, 768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        reshape_default: "f32[128, 128, 3072]" = torch.ops.aten.reshape.default(mm_136, _shape_param_0);  mm_136 = _shape_param_0 = None
        convert_element_type_default: "f32[128, 128, 3072]" = torch.ops.prims.convert_element_type.default(gt_4, torch.float32);  gt_4 = None
        mul_tensor: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(reshape_default, mul_tensor);  reshape_default = mul_tensor = None
        reshape_default_1: "f32[128, 128, 3072]" = torch.ops.aten.reshape.default(addmm_4, _shape_param_1);  addmm_4 = _shape_param_1 = None
        mul_tensor_2: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(reshape_default_1, 0.7071067811865476)
        erf_default: "f32[128, 128, 3072]" = torch.ops.aten.erf.default(mul_tensor_2);  mul_tensor_2 = None
        add_tensor: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_3: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(add_tensor, 0.5);  add_tensor = None
        mul_tensor_4: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(reshape_default_1, reshape_default_1)
        mul_tensor_5: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_tensor_4, -0.5);  mul_tensor_4 = None
        exp_default: "f32[128, 128, 3072]" = torch.ops.aten.exp.default(mul_tensor_5);  mul_tensor_5 = None
        mul_tensor_6: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_default, 0.3989422804014327);  exp_default = None
        mul_tensor_7: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(reshape_default_1, mul_tensor_6);  reshape_default_1 = mul_tensor_6 = None
        add_tensor_1: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(mul_tensor_3, mul_tensor_7);  mul_tensor_3 = mul_tensor_7 = None
        mul_tensor_8: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_tensor_1, add_tensor_1);  mul_tensor_1 = add_tensor_1 = None
        reshape_default_2: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_tensor_8, _shape_param_2);  mul_tensor_8 = _shape_param_2 = None
        permute_default: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
        permute_default_1: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
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
