"""
Standalone repro captured via capture_hook.
Label: hf_XLNetLMHeadModel_training
Pattern hash: 62614da58896
Shape hash: 8c6be8d0
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm: "f32[4096, 1024]", _shape_param_0, gt_98: "b8[512, 8, 1024]", primals_361: "f32[1024]", mul_388: "f32[512, 8, 1024]", div_27: "f32[512, 8, 1]", gt_97: "b8[512, 8, 1024]", _shape_param_1, primals_359: "f32[1024, 4096]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1426 in forward, code: logits = self.lm_loss(hidden_states[:, slice_indices, :])
        reshape_default: "f32[8, 512, 1024]" = torch.ops.aten.reshape.default(mm, _shape_param_0);  mm = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1180 in forward, code: output = output.permute(1, 0, 2).contiguous()
        permute_default: "f32[512, 8, 1024]" = torch.ops.aten.permute.default(reshape_default, [1, 0, 2]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1177 in forward, code: output = self.dropout(output_g if output_g is not None else output_h)
        convert_element_type_default: "f32[512, 8, 1024]" = torch.ops.prims.convert_element_type.default(gt_98, torch.float32);  gt_98 = None
        mul_tensor: "f32[512, 8, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[512, 8, 1024]" = torch.ops.aten.mul.Tensor(permute_default, mul_tensor);  permute_default = mul_tensor = None
        clone_default: "f32[512, 8, 1024]" = torch.ops.aten.clone.default(mul_tensor_1, memory_format = torch.contiguous_format);  mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        mul_tensor_2: "f32[512, 8, 1024]" = torch.ops.aten.mul.Tensor(clone_default, primals_361);  clone_default = primals_361 = None
        mul_tensor_3: "f32[512, 8, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 1024)
        sum_dim_int_list: "f32[512, 8, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True)
        mul_tensor_4: "f32[512, 8, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_388);  mul_tensor_2 = None
        sum_dim_int_list_1: "f32[512, 8, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [2], True);  mul_tensor_4 = None
        mul_tensor_5: "f32[512, 8, 1024]" = torch.ops.aten.mul.Tensor(mul_388, sum_dim_int_list_1);  mul_388 = sum_dim_int_list_1 = None
        sub_tensor: "f32[512, 8, 1024]" = torch.ops.aten.sub.Tensor(mul_tensor_3, sum_dim_int_list);  mul_tensor_3 = sum_dim_int_list = None
        sub_tensor_1: "f32[512, 8, 1024]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_5);  sub_tensor = mul_tensor_5 = None
        mul_tensor_6: "f32[512, 8, 1024]" = torch.ops.aten.mul.Tensor(div_27, sub_tensor_1);  div_27 = sub_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:303 in forward, code: output = self.dropout(output)
        convert_element_type_default_1: "f32[512, 8, 1024]" = torch.ops.prims.convert_element_type.default(gt_97, torch.float32);  gt_97 = None
        mul_tensor_7: "f32[512, 8, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_8: "f32[512, 8, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_6, mul_tensor_7);  mul_tensor_6 = mul_tensor_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        reshape_default_1: "f32[4096, 1024]" = torch.ops.aten.reshape.default(mul_tensor_8, _shape_param_1);  mul_tensor_8 = _shape_param_1 = None
        permute_default_1: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_359, [1, 0]);  primals_359 = None
        permute_default_2: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_default_1, [1, 0]);  permute_default_1 = None
        return (reshape_default_1, permute_default_2)


def _default_make_inputs():
    return [
    torch.randn([4096, 1024], dtype=torch.float32, device='cuda'),
    [8, 512, 1024],  # _shape_param_0
    torch.randint(0, 2, [512, 8, 1024], dtype=torch.bool, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([512, 8, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([512, 8, 1], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [512, 8, 1024], dtype=torch.bool, device='cuda'),
    [4096, 1024],  # _shape_param_1
    torch.randn([1024, 4096], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
