"""
Standalone repro captured via capture_hook.
Label: falcon_rw_1b
Pattern hash: 8da395803e57
Shape hash: 686e0070
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
    def forward(self, mm_95: "f16[2048, 2048]", _shape_param_0, arg291_1: "f16[2048]", add_260: "f16[4, 512, 2048]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/falcon/modeling_falcon.py:62 in forward, code: hidden_states = input @ self.weight.T
        reshape_default: "f16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_95, _shape_param_0);  mm_95 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/falcon/modeling_falcon.py:65 in forward, code: return hidden_states + self.bias
        add_tensor: "f16[4, 512, 2048]" = torch.ops.aten.add.Tensor(reshape_default, arg291_1);  reshape_default = arg291_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/falcon/modeling_falcon.py:212 in dropout_add, code: out = residual + out
        add_tensor_1: "f16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_260, add_tensor);  add_260 = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/falcon/modeling_falcon.py:819 in forward, code: hidden_states = self.ln_f(hidden_states)
        convert_element_type_default: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float32);  add_tensor_1 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True);  convert_element_type_default = None
        getitem: "f32[4, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[4, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        _output_to_half_0: "f16[4, 512, 1]" = torch.ops.prims.convert_element_type.default(getitem, torch.float16);  getitem = None
        _output_to_half_1: "f16[4, 512, 1]" = torch.ops.prims.convert_element_type.default(getitem_1, torch.float16);  getitem_1 = None
        return (_output_to_half_0, _output_to_half_1)


def _default_make_inputs():
    return [
    torch.randn([2048, 2048], dtype=torch.float16, device='cuda'),
    [4, 512, 2048],  # _shape_param_0
    torch.randn([2048], dtype=torch.float16, device='cuda'),
    torch.randn([4, 512, 2048], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
