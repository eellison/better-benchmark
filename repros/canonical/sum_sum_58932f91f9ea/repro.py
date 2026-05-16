"""
Standalone repro captured via capture_hook.
Label: hf_GoogleFnet_training
Pattern hash: 58932f91f9ea
Shape hash: ce0795b4
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, view_as_real_22: "f32[8, 512, 768, 2]", mul_421: "f32[8, 512, 768]", primals_17: "f32[768]", mul_12: "f32[8, 512, 768]", div_25: "f32[8, 512, 1]", gt_1: "b8[8, 512, 768]", _shape_param_0, primals_15: "f32[768, 3072]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:176 in forward, code: outputs = self.fourier_transform(hidden_states).real
        select_int: "f32[8, 512, 768]" = torch.ops.aten.select.int(view_as_real_22, 3, 0);  view_as_real_22 = None
        add_tensor: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(mul_421, select_int);  mul_421 = select_int = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(add_tensor, primals_17);  add_tensor = primals_17 = None
        mul_tensor_1: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_12);  mul_tensor = None
        sum_dim_int_list_1: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul_12, sum_dim_int_list_1);  mul_12 = sum_dim_int_list_1 = None
        sub_tensor: "f32[8, 512, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[8, 512, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(div_25, sub_tensor_1);  div_25 = sub_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:229 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_default: "f32[8, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_tensor_5: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_6: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:228 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default: "f32[4096, 768]" = torch.ops.aten.reshape.default(mul_tensor_6, _shape_param_0);  mul_tensor_6 = _shape_param_0 = None
        permute_default: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
        permute_default_1: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([8, 512, 768, 2], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [8, 512, 768], dtype=torch.bool, device='cuda'),
    [4096, 768],  # _shape_param_0
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
