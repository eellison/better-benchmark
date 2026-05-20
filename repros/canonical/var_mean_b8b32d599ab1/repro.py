"""
Standalone repro captured via capture_hook.
Label: hf_DistilBertForMaskedLM_infer
Pattern hash: b8b32d599ab1
Shape hash: ae139990
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([32768, 768], f32), T([30522], f32), T([768], f32), T([768], f32), T([30522, 768], f32), S([256, 128, 768]), S([32768, 768]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_36: "f32[32768, 768]", arg106_1: "f32[30522]", arg104_1: "f32[768]", arg105_1: "f32[768]", arg1_1: "f32[30522, 768]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:514 in forward, code: prediction_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
        reshape_default: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(addmm_36, _shape_param_0);  addmm_36 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_tensor: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(reshape_default, 0.5)
        mul_tensor_1: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(reshape_default, 0.7071067811865476);  reshape_default = None
        erf_default: "f32[256, 128, 768]" = torch.ops.aten.erf.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_2: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  mul_tensor = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:516 in forward, code: prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
        var_mean_correction = torch.ops.aten.var_mean.correction(mul_tensor_2, [2], correction = 0, keepdim = True)
        getitem: "f32[256, 128, 1]" = var_mean_correction[0]
        getitem_1: "f32[256, 128, 1]" = var_mean_correction[1];  var_mean_correction = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:517 in forward, code: prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)
        full_default: "f32[2]" = torch.ops.aten.full.default([2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat_default: "f32[30524]" = torch.ops.aten.cat.default([arg106_1, full_default]);  arg106_1 = full_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:516 in forward, code: prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
        sub_tensor: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_2, getitem_1);  mul_tensor_2 = getitem_1 = None
        add_tensor_1: "f32[256, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[256, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor_3: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_4: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_3, arg104_1);  mul_tensor_3 = arg104_1 = None
        add_tensor_2: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_tensor_4, arg105_1);  mul_tensor_4 = arg105_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:517 in forward, code: prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)
        reshape_default_1: "f32[32768, 768]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_1);  add_tensor_2 = _shape_param_1 = None
        permute_default: "f32[768, 30522]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        constant_pad_nd_default: "f32[768, 30524]" = torch.ops.aten.constant_pad_nd.default(permute_default, [0, 2, 0, 0]);  permute_default = None
        return (cat_default, reshape_default_1, constant_pad_nd_default)


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
