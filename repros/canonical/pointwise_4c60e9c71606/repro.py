"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_training
Pattern hash: 4c60e9c71606
Shape hash: e43f1f24
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_347: "f32[1024, 128]", _shape_param_0, primals_1074: "f32[128]", primals_1075: "f32[128]", _shape_param_1, primals_1076: "f32[128, 128]", primals_1078: "f32[128, 128]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        reshape_default: "f32[8, 128, 128]" = torch.ops.aten.reshape.default(addmm_347, _shape_param_0);  addmm_347 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_tensor: "f32[8, 128, 128]" = torch.ops.aten.mul.Tensor(reshape_default, primals_1074);  reshape_default = primals_1074 = None
        add_tensor: "f32[8, 128, 128]" = torch.ops.aten.add.Tensor(mul_tensor, primals_1075);  mul_tensor = primals_1075 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        reshape_default_1: "f32[1024, 128]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_1);  add_tensor = _shape_param_1 = None
        permute_default: "f32[128, 128]" = torch.ops.aten.permute.default(primals_1076, [1, 0]);  primals_1076 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        permute_default_1: "f32[128, 128]" = torch.ops.aten.permute.default(primals_1078, [1, 0]);  primals_1078 = None
        return (reshape_default_1, permute_default, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([1024, 128], dtype=torch.float32, device='cuda'),
    [8, 128, 128],  # _shape_param_0
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    [1024, 128],  # _shape_param_1
    torch.randn([128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([128, 128], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
