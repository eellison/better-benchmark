"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Whisper_train
Pattern hash: e1cc38ad9185
Shape hash: ad9f6940
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
_shapes_config = "(T([12000, 384], f32), T([8, 1500, 384], f32, stride=(576000, 1, 1500)), T([384], f32), T([384], f32), S([8, 1500, 384]), S([12000, 384]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_2: "f32[12000, 384]", primals_3: "f32[8, 1500, 384]", primals_11: "f32[384]", primals_12: "f32[384]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        reshape_default: "f32[8, 1500, 384]" = torch.ops.aten.reshape.default(addmm_2, _shape_param_0);  addmm_2 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:399 in forward, code: hidden_states = residual + hidden_states
        add_tensor: "f32[8, 1500, 384]" = torch.ops.aten.add.Tensor(primals_3, reshape_default);  primals_3 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:402 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        clone_default: "f32[8, 1500, 384]" = torch.ops.aten.clone.default(add_tensor, memory_format = torch.contiguous_format);  add_tensor = None
        var_mean_correction = torch.ops.aten.var_mean.correction(clone_default, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 1500, 1]" = var_mean_correction[0]
        getitem_1: "f32[8, 1500, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[8, 1500, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[8, 1500, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[8, 1500, 384]" = torch.ops.aten.sub.Tensor(clone_default, getitem_1);  clone_default = getitem_1 = None
        mul_tensor: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(mul_tensor, primals_11);  mul_tensor = primals_11 = None
        add_tensor_2: "f32[8, 1500, 384]" = torch.ops.aten.add.Tensor(mul_tensor_1, primals_12);  mul_tensor_1 = primals_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:403 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        reshape_default_1: "f32[12000, 384]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_1);  add_tensor_2 = _shape_param_1 = None
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
