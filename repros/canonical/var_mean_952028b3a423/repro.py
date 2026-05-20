"""
Standalone repro captured via capture_hook.
Label: hf_DistilBertForMaskedLM_train
Pattern hash: 952028b3a423
Shape hash: a58d9cc5
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([32768, 768], f32), T([13], i64, max=13), T([256, 128, 768], f32), T([768], f32), T([768], f32), T([768, 768], f32), S([256, 128, 768]), S([32768, 768]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_35: "f32[32768, 768]", inductor_seeds_default: "i64[13]", add_48: "f32[256, 128, 768]", primals_101: "f32[768]", primals_102: "f32[768]", primals_103: "f32[768, 768]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:226 in ff_chunk, code: x = self.lin2(x)
        reshape_default: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(addmm_35, _shape_param_0);  addmm_35 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:227 in ff_chunk, code: x = self.dropout(x)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 12);  inductor_seeds_default = None
        inductor_random_default: "f32[256, 128, 768]" = torch.ops.prims.inductor_random.default([256, 128, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[256, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(gt_scalar, reshape_default);  gt_scalar = reshape_default = None
        mul_tensor_1: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:261 in forward, code: ffn_output = self.output_layer_norm(ffn_output + attention_output)
        add_tensor: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, add_48);  mul_tensor_1 = add_48 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[256, 128, 1]" = var_mean_correction[0]
        getitem_1: "f32[256, 128, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[256, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[256, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        mul_tensor_2: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_3: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, primals_101);  mul_tensor_2 = primals_101 = None
        add_tensor_2: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_tensor_3, primals_102);  mul_tensor_3 = primals_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:514 in forward, code: prediction_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
        reshape_default_1: "f32[32768, 768]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_1);  add_tensor_2 = _shape_param_1 = None
        permute_default: "f32[768, 768]" = torch.ops.aten.permute.default(primals_103, [1, 0]);  primals_103 = None
        return (reshape_default_1, permute_default)


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
