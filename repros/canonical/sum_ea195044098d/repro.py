"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_train
Pattern hash: ea195044098d
Shape hash: 816937af
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([], f32), T([], f32), T([256, 128], i64, max=256), T([32768, 30522], f32, stride=(30528, 1)), T([256, 128, 30522], f32), T([30522, 512], f32, stride=(1, 30528)), S([1, 30522]), S([32768, 30522]), S([256, 128, 30522]), S([32768, 30522]))"

class Repro(torch.nn.Module):
    def forward(self, tangents_1: "f32[]", convert_element_type: "f32[]", primals_1120: "i64[256, 128]", exp_25: "f32[32768, 30522]", tangents_2: "f32[256, 128, 30522]", permute_484: "f32[30522, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:825 in forward, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        div_tensor: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type);  tangents_1 = convert_element_type = None
        reshape_default: "i64[32768]" = torch.ops.aten.reshape.default(primals_1120, [-1]);  primals_1120 = None
        unsqueeze_default: "i64[32768, 1]" = torch.ops.aten.unsqueeze.default(reshape_default, 1);  reshape_default = None
        ne_scalar: "b8[32768, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_default, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[32768, 1]" = torch.ops.aten.where.self(ne_scalar, unsqueeze_default, full_default);  unsqueeze_default = full_default = None

        # No stacktrace found for following nodes
        iota_default: "i64[30522]" = torch.ops.prims.iota.default(30522, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        reshape_default_1: "i64[1, 30522]" = torch.ops.aten.reshape.default(iota_default, _shape_param_0);  iota_default = _shape_param_0 = None
        expand_default: "i64[32768, 30522]" = torch.ops.aten.expand.default(where_self, _shape_param_1);  where_self = _shape_param_1 = None
        eq_tensor: "b8[32768, 30522]" = torch.ops.aten.eq.Tensor(expand_default, reshape_default_1);  expand_default = reshape_default_1 = None
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:825 in forward, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        where_self_1: "f32[32768, 30522]" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:825 in forward, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        where_self_2: "f32[32768, 1]" = torch.ops.aten.where.self(ne_scalar, div_tensor, full_default_1);  ne_scalar = div_tensor = full_default_1 = None
        mul_tensor: "f32[32768, 30522]" = torch.ops.aten.mul.Tensor(where_self_1, where_self_2);  where_self_1 = where_self_2 = None
        sum_dim_int_list: "f32[32768, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [1], True)
        mul_tensor_1: "f32[32768, 30522]" = torch.ops.aten.mul.Tensor(exp_25, sum_dim_int_list);  exp_25 = sum_dim_int_list = None
        sub_tensor: "f32[32768, 30522]" = torch.ops.aten.sub.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None
        reshape_default_2: "f32[256, 128, 30522]" = torch.ops.aten.reshape.default(sub_tensor, _shape_param_2);  sub_tensor = _shape_param_2 = None
        add_tensor: "f32[256, 128, 30522]" = torch.ops.aten.add.Tensor(tangents_2, reshape_default_2);  tangents_2 = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:507 in forward, code: hidden_states = hidden_states.matmul(torch.cat([self.decoder.weight.t(), self.dense.weight], dim=0))
        reshape_default_3: "f32[32768, 30522]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_3);  add_tensor = _shape_param_3 = None
        constant_pad_nd_default: "f32[32768, 30524]" = torch.ops.aten.constant_pad_nd.default(reshape_default_3, [0, 2, 0, 0]);  reshape_default_3 = None
        constant_pad_nd_default_1: "f32[30524, 512]" = torch.ops.aten.constant_pad_nd.default(permute_484, [0, 0, 0, 2]);  permute_484 = None
        return (constant_pad_nd_default, constant_pad_nd_default_1)


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
