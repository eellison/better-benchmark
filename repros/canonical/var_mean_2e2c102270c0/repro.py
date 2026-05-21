"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Reformer_infer
Pattern hash: 2e2c102270c0
Shape hash: c293e185
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
_shapes_config = "(T([4096, 256], f16), T([1, 4096, 256], f16), T([256], f16), T([256], f16), S([1, 4096, 256]), S([4096, 256]), S([4096, 256]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_3: "f16[4096, 256]", add_6: "f16[1, 4096, 256]", arg26_1: "f16[256]", arg27_1: "f16[256]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1450 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default: "f16[1, 4096, 256]" = torch.ops.aten.reshape.default(addmm_3, _shape_param_0);  addmm_3 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1567 in forward, code: hidden_states = hidden_states + self.feed_forward(attn_output)
        add_tensor: "f16[1, 4096, 256]" = torch.ops.aten.add.Tensor(add_6, reshape_default);  add_6 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1373 in forward, code: hidden_states = self.layer_norm(hidden_states)
        convert_element_type_default: "f32[1, 4096, 256]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32);  add_tensor = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 4096, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 4096, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[1, 4096, 256]" = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1);  convert_element_type_default = getitem_1 = None
        add_tensor_1: "f32[1, 4096, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[1, 4096, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[1, 4096, 256]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[1, 4096, 256]" = torch.ops.aten.mul.Tensor(mul_tensor, arg26_1);  mul_tensor = arg26_1 = None
        add_tensor_2: "f32[1, 4096, 256]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg27_1);  mul_tensor_1 = arg27_1 = None
        convert_element_type_default_1: "f16[1, 4096, 256]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float16);  add_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1162 in forward, code: query_vectors = self.query(hidden_states)
        reshape_default_1: "f16[4096, 256]" = torch.ops.aten.reshape.default(convert_element_type_default_1, _shape_param_1);  _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1163 in forward, code: key_vectors = self.key(hidden_states)
        reshape_default_2: "f16[4096, 256]" = torch.ops.aten.reshape.default(convert_element_type_default_1, _shape_param_2);  convert_element_type_default_1 = _shape_param_2 = None
        return (reshape_default_2, reshape_default_1)



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
