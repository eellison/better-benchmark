"""
Standalone repro captured via capture_hook.
Label: hf_LayoutLMForMaskedLM_infer
Pattern hash: 379df05ba796
Shape hash: ef250ee2
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([32, 512], i64, gen=Index(512)), T([16384, 30524], f32), S([32, 512, 30522]), S([-1, 30522]))"

class Repro(torch.nn.Module):
    def forward(self, arg210_1: "i64[32, 512]", addmm_default: "f32[16384, 30524]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:496 in forward, code: attention_mask = torch.ones(input_shape, device=device)
        full_default: "f32[32, 512]" = torch.ops.aten.full.default([32, 512], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:503 in forward, code: extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        unsqueeze_default: "f32[32, 1, 512]" = torch.ops.aten.unsqueeze.default(full_default, 1);  full_default = None
        unsqueeze_default_1: "f32[32, 1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:506 in forward, code: extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min
        sub_tensor: "f32[32, 1, 1, 512]" = torch.ops.aten.sub.Tensor(1.0, unsqueeze_default_1);  unsqueeze_default_1 = None
        full_default_1: "f32[32, 1, 1, 512]" = torch.ops.aten.full.default([32, 1, 1, 512], -0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:634 in forward, code: labels.view(-1),
        reshape_default: "i64[16384]" = torch.ops.aten.reshape.default(arg210_1, [-1]);  arg210_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:632 in forward, code: masked_lm_loss = loss_fct(
        ne_scalar: "b8[16384]" = torch.ops.aten.ne.Scalar(reshape_default, -100)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:378 in forward, code: hidden_states = self.decoder(hidden_states)
        slice_tensor: "f32[16384, 30522]" = torch.ops.aten.slice.Tensor(addmm_default, 1, 0, -2);  addmm_default = None
        reshape_default_1: "f32[32, 512, 30522]" = torch.ops.aten.reshape.default(slice_tensor, _shape_param_0);  slice_tensor = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:633 in forward, code: prediction_scores.view(-1, self.config.vocab_size),
        reshape_default_2: "f32[16384, 30522]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_1);  reshape_default_1 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:632 in forward, code: masked_lm_loss = loss_fct(
        amax_default: "f32[16384, 1]" = torch.ops.aten.amax.default(reshape_default_2, [1], True)
        sub_tensor_1: "f32[16384, 30522]" = torch.ops.aten.sub.Tensor(reshape_default_2, amax_default);  reshape_default_2 = amax_default = None
        exp_default: "f32[16384, 30522]" = torch.ops.aten.exp.default(sub_tensor_1)
        sum_dim_int_list: "f32[16384, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True);  exp_default = None
        log_default: "f32[16384, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_2: "f32[16384, 30522]" = torch.ops.aten.sub.Tensor(sub_tensor_1, log_default);  sub_tensor_1 = log_default = None
        ne_scalar_1: "b8[16384]" = torch.ops.aten.ne.Scalar(reshape_default, -100)
        full_default_2: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[16384]" = torch.ops.aten.where.self(ne_scalar_1, reshape_default, full_default_2);  ne_scalar_1 = full_default_2 = None
        unsqueeze_default_2: "i64[16384, 1]" = torch.ops.aten.unsqueeze.default(where_self, 1);  where_self = None
        gather_default: "f32[16384, 1]" = torch.ops.aten.gather.default(sub_tensor_2, 1, unsqueeze_default_2);  sub_tensor_2 = unsqueeze_default_2 = None
        squeeze_dim: "f32[16384]" = torch.ops.aten.squeeze.dim(gather_default, 1);  gather_default = None
        neg_default: "f32[16384]" = torch.ops.aten.neg.default(squeeze_dim);  squeeze_dim = None
        full_default_3: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f32[16384]" = torch.ops.aten.where.self(ne_scalar, neg_default, full_default_3);  ne_scalar = neg_default = full_default_3 = None
        sum_default: "f32[]" = torch.ops.aten.sum.default(where_self_1);  where_self_1 = None
        ne_scalar_2: "b8[16384]" = torch.ops.aten.ne.Scalar(reshape_default, -100);  reshape_default = None
        sum_default_1: "i64[]" = torch.ops.aten.sum.default(ne_scalar_2);  ne_scalar_2 = None
        convert_element_type_default: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_1, torch.float32);  sum_default_1 = None
        div_tensor: "f32[]" = torch.ops.aten.div.Tensor(sum_default, convert_element_type_default);  sum_default = convert_element_type_default = None
        return (sub_tensor, full_default_1, div_tensor)


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
