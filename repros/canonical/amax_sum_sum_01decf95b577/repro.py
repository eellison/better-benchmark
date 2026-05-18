"""
Standalone repro captured via capture_hook.
Label: hf_LayoutLMForMaskedLM_train
Pattern hash: 01decf95b577
Shape hash: 9dfde6cc
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_default: "f32[16384, 30524]", _shape_param_0, _shape_param_1, primals_211: "i64[32, 512]", rsqrt_24: "f32[32, 512, 1]", rsqrt_23: "f32[32, 512, 1]", rsqrt_22: "f32[32, 512, 1]", rsqrt_21: "f32[32, 512, 1]", rsqrt_20: "f32[32, 512, 1]", rsqrt_19: "f32[32, 512, 1]", rsqrt_18: "f32[32, 512, 1]", rsqrt_17: "f32[32, 512, 1]", rsqrt_16: "f32[32, 512, 1]", rsqrt_15: "f32[32, 512, 1]", rsqrt_14: "f32[32, 512, 1]", rsqrt_13: "f32[32, 512, 1]", rsqrt_12: "f32[32, 512, 1]", rsqrt_11: "f32[32, 512, 1]", rsqrt_10: "f32[32, 512, 1]", rsqrt_9: "f32[32, 512, 1]", rsqrt_8: "f32[32, 512, 1]", rsqrt_7: "f32[32, 512, 1]", rsqrt_6: "f32[32, 512, 1]", rsqrt_5: "f32[32, 512, 1]", rsqrt_4: "f32[32, 512, 1]", rsqrt_3: "f32[32, 512, 1]", rsqrt_2: "f32[32, 512, 1]", rsqrt_1: "f32[32, 512, 1]", rsqrt: "f32[32, 512, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:378 in forward, code: hidden_states = self.decoder(hidden_states)
        slice_tensor: "f32[16384, 30522]" = torch.ops.aten.slice.Tensor(addmm_default, 1, 0, -2);  addmm_default = None
        reshape_default: "f32[32, 512, 30522]" = torch.ops.aten.reshape.default(slice_tensor, _shape_param_0);  slice_tensor = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:633 in forward, code: prediction_scores.view(-1, self.config.vocab_size),
        reshape_default_1: "f32[16384, 30522]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:634 in forward, code: labels.view(-1),
        reshape_default_2: "i64[16384]" = torch.ops.aten.reshape.default(primals_211, [-1]);  primals_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:632 in forward, code: masked_lm_loss = loss_fct(
        amax_default: "f32[16384, 1]" = torch.ops.aten.amax.default(reshape_default_1, [1], True)
        sub_tensor: "f32[16384, 30522]" = torch.ops.aten.sub.Tensor(reshape_default_1, amax_default);  reshape_default_1 = amax_default = None
        exp_default: "f32[16384, 30522]" = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list: "f32[16384, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True);  exp_default = None
        log_default: "f32[16384, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1: "f32[16384, 30522]" = torch.ops.aten.sub.Tensor(sub_tensor, log_default);  sub_tensor = log_default = None
        ne_scalar: "b8[16384]" = torch.ops.aten.ne.Scalar(reshape_default_2, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[16384]" = torch.ops.aten.where.self(ne_scalar, reshape_default_2, full_default);  reshape_default_2 = full_default = None
        unsqueeze_default: "i64[16384, 1]" = torch.ops.aten.unsqueeze.default(where_self, 1);  where_self = None
        gather_default: "f32[16384, 1]" = torch.ops.aten.gather.default(sub_tensor_1, 1, unsqueeze_default);  sub_tensor_1 = unsqueeze_default = None
        squeeze_dim: "f32[16384]" = torch.ops.aten.squeeze.dim(gather_default, 1);  gather_default = None
        neg_default: "f32[16384]" = torch.ops.aten.neg.default(squeeze_dim);  squeeze_dim = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f32[16384]" = torch.ops.aten.where.self(ne_scalar, neg_default, full_default_1);  neg_default = full_default_1 = None
        sum_default: "i64[]" = torch.ops.aten.sum.default(ne_scalar);  ne_scalar = None
        convert_element_type_default: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default, torch.float32);  sum_default = None
        sum_default_1: "f32[]" = torch.ops.aten.sum.default(where_self_1);  where_self_1 = None
        div_tensor: "f32[]" = torch.ops.aten.div.Tensor(sum_default_1, convert_element_type_default);  sum_default_1 = convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_1: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_2: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_3: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_4: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_5: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_6: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_7: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_8: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_9: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_10: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_11: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_12: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_13: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_14: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_15: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_16: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_17: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_18: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_19: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_20: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_21: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_22: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_23: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_24: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:119 in forward, code: embeddings = self.LayerNorm(embeddings)
        div_tensor_25: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        return (div_tensor, div_tensor_1, div_tensor_2, div_tensor_3, div_tensor_4, div_tensor_5, div_tensor_6, div_tensor_7, div_tensor_8, div_tensor_9, div_tensor_10, div_tensor_11, div_tensor_12, div_tensor_13, div_tensor_14, div_tensor_15, div_tensor_16, div_tensor_17, div_tensor_18, div_tensor_19, div_tensor_20, div_tensor_21, div_tensor_22, div_tensor_23, div_tensor_24, div_tensor_25)


def _default_make_inputs():
    return [
    torch.randn([16384, 30524], dtype=torch.float32, device='cuda'),
    [32, 512, 30522],  # _shape_param_0
    [-1, 30522],  # _shape_param_1
    torch.randint(0, 2, [32, 512], dtype=torch.int64, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
