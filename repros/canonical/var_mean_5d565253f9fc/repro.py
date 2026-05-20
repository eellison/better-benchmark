"""
Standalone repro captured via capture_hook.
Label: hf_AlbertForMaskedLM_train
Pattern hash: 5d565253f9fc
Shape hash: 05d74714
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([4096, 4096], f32), T([8, 512, 4096], f32), T([4096], f32), T([4096], f32), S([8, 512, 4096]), S([4096, 4096]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_66: "f32[4096, 4096]", add_99: "f32[8, 512, 4096]", primals_25: "f32[4096]", primals_26: "f32[4096]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        reshape_default: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_66, _shape_param_0);  addmm_66 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        add_tensor: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(reshape_default, add_99);  reshape_default = add_99 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[8, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        mul_tensor: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_tensor, primals_25);  mul_tensor = primals_25 = None
        add_tensor_2: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_tensor_1, primals_26);  mul_tensor_1 = primals_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default_1: "f32[4096, 4096]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_1);  add_tensor_2 = _shape_param_1 = None
        return reshape_default_1


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
