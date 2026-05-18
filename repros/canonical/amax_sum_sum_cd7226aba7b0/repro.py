"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_train
Pattern hash: cd7226aba7b0
Shape hash: 9fabe7a2
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
    def forward(self, addmm_default: "f32[8192, 50268]", _shape_param_0, _shape_param_1, primals_8: "i64[8, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1281 in forward, code: x = self.decoder(x)
        slice_tensor: "f32[8192, 50265]" = torch.ops.aten.slice.Tensor(addmm_default, 1, 0, -3);  addmm_default = None
        reshape_default: "f32[8, 1024, 50265]" = torch.ops.aten.reshape.default(slice_tensor, _shape_param_0);  slice_tensor = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1635 in torch_dynamo_resume_in_forward_at_1616, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        reshape_default_1: "f32[8192, 50265]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        reshape_default_2: "i64[8192]" = torch.ops.aten.reshape.default(primals_8, [-1]);  primals_8 = None
        amax_default: "f32[8192, 1]" = torch.ops.aten.amax.default(reshape_default_1, [1], True)
        sub_tensor: "f32[8192, 50265]" = torch.ops.aten.sub.Tensor(reshape_default_1, amax_default);  reshape_default_1 = amax_default = None
        exp_default: "f32[8192, 50265]" = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True);  exp_default = None
        log_default: "f32[8192, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1: "f32[8192, 50265]" = torch.ops.aten.sub.Tensor(sub_tensor, log_default);  sub_tensor = log_default = None
        ne_scalar: "b8[8192]" = torch.ops.aten.ne.Scalar(reshape_default_2, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[8192]" = torch.ops.aten.where.self(ne_scalar, reshape_default_2, full_default);  reshape_default_2 = full_default = None
        unsqueeze_default: "i64[8192, 1]" = torch.ops.aten.unsqueeze.default(where_self, 1);  where_self = None
        gather_default: "f32[8192, 1]" = torch.ops.aten.gather.default(sub_tensor_1, 1, unsqueeze_default);  sub_tensor_1 = unsqueeze_default = None
        squeeze_dim: "f32[8192]" = torch.ops.aten.squeeze.dim(gather_default, 1);  gather_default = None
        neg_default: "f32[8192]" = torch.ops.aten.neg.default(squeeze_dim);  squeeze_dim = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f32[8192]" = torch.ops.aten.where.self(ne_scalar, neg_default, full_default_1);  neg_default = full_default_1 = None
        sum_default: "i64[]" = torch.ops.aten.sum.default(ne_scalar);  ne_scalar = None
        convert_element_type_default: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default, torch.float32);  sum_default = None
        sum_default_1: "f32[]" = torch.ops.aten.sum.default(where_self_1);  where_self_1 = None
        div_tensor: "f32[]" = torch.ops.aten.div.Tensor(sum_default_1, convert_element_type_default);  sum_default_1 = convert_element_type_default = None
        return div_tensor


def _default_make_inputs():
    return [
    torch.randn([8192, 50268], dtype=torch.float32, device='cuda'),
    [8, 1024, 50265],  # _shape_param_0
    [-1, 50265],  # _shape_param_1
    torch.randint(0, 2, [8, 1024], dtype=torch.int64, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
