"""
Standalone repro captured via capture_hook.
Label: hf_XLNetLMHeadModel_infer
Pattern hash: 7f552650eac2
Shape hash: bbdf5a1e
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
    def forward(self, arg363_1: "i64[16, 512]", addmm_48: "f32[8192, 32000]", embedding: "f32[512, 16, 1024]", add_10: "f32[512, 16, 1024]", add_21: "f32[512, 16, 1024]", add_32: "f32[512, 16, 1024]", add_43: "f32[512, 16, 1024]", add_54: "f32[512, 16, 1024]", add_65: "f32[512, 16, 1024]", add_76: "f32[512, 16, 1024]", add_87: "f32[512, 16, 1024]", add_98: "f32[512, 16, 1024]", add_109: "f32[512, 16, 1024]", add_120: "f32[512, 16, 1024]", add_131: "f32[512, 16, 1024]", add_142: "f32[512, 16, 1024]", add_153: "f32[512, 16, 1024]", add_164: "f32[512, 16, 1024]", add_175: "f32[512, 16, 1024]", add_186: "f32[512, 16, 1024]", add_197: "f32[512, 16, 1024]", add_208: "f32[512, 16, 1024]", add_219: "f32[512, 16, 1024]", add_230: "f32[512, 16, 1024]", add_241: "f32[512, 16, 1024]", add_252: "f32[512, 16, 1024]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1432 in forward, code: loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        reshape_default: "i64[8192]" = torch.ops.aten.reshape.default(arg363_1, [-1]);  arg363_1 = None
        ne_scalar: "b8[8192]" = torch.ops.aten.ne.Scalar(reshape_default, -100)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1426 in forward, code: logits = self.lm_loss(hidden_states[:, slice_indices, :])
        reshape_default_1: "f32[16, 512, 32000]" = torch.ops.aten.reshape.default(addmm_48, _shape_param_0);  addmm_48 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1432 in forward, code: loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        reshape_default_2: "f32[8192, 32000]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_1);  reshape_default_1 = _shape_param_1 = None
        amax_default: "f32[8192, 1]" = torch.ops.aten.amax.default(reshape_default_2, [1], True)
        sub_tensor: "f32[8192, 32000]" = torch.ops.aten.sub.Tensor(reshape_default_2, amax_default);  reshape_default_2 = amax_default = None
        exp_default: "f32[8192, 32000]" = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True);  exp_default = None
        log_default: "f32[8192, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1: "f32[8192, 32000]" = torch.ops.aten.sub.Tensor(sub_tensor, log_default);  sub_tensor = log_default = None
        ne_scalar_1: "b8[8192]" = torch.ops.aten.ne.Scalar(reshape_default, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[8192]" = torch.ops.aten.where.self(ne_scalar_1, reshape_default, full_default);  ne_scalar_1 = full_default = None
        unsqueeze_default: "i64[8192, 1]" = torch.ops.aten.unsqueeze.default(where_self, 1);  where_self = None
        gather_default: "f32[8192, 1]" = torch.ops.aten.gather.default(sub_tensor_1, 1, unsqueeze_default);  sub_tensor_1 = unsqueeze_default = None
        squeeze_dim: "f32[8192]" = torch.ops.aten.squeeze.dim(gather_default, 1);  gather_default = None
        neg_default: "f32[8192]" = torch.ops.aten.neg.default(squeeze_dim);  squeeze_dim = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f32[8192]" = torch.ops.aten.where.self(ne_scalar, neg_default, full_default_1);  ne_scalar = neg_default = full_default_1 = None
        sum_default: "f32[]" = torch.ops.aten.sum.default(where_self_1);  where_self_1 = None
        ne_scalar_2: "b8[8192]" = torch.ops.aten.ne.Scalar(reshape_default, -100);  reshape_default = None
        sum_default_1: "i64[]" = torch.ops.aten.sum.default(ne_scalar_2);  ne_scalar_2 = None
        convert_element_type_default: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_1, torch.float32);  sum_default_1 = None
        div_tensor: "f32[]" = torch.ops.aten.div.Tensor(sum_default, convert_element_type_default);  sum_default = convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:923 in cache_mem, code: new_mem = curr_out[cutoff:]
        slice_tensor: "f32[512, 16, 1024]" = torch.ops.aten.slice.Tensor(embedding, 0, -512, 9223372036854775807);  embedding = None
        slice_tensor_1: "f32[512, 16, 1024]" = torch.ops.aten.slice.Tensor(add_10, 0, -512, 9223372036854775807);  add_10 = None
        slice_tensor_2: "f32[512, 16, 1024]" = torch.ops.aten.slice.Tensor(add_21, 0, -512, 9223372036854775807);  add_21 = None
        slice_tensor_3: "f32[512, 16, 1024]" = torch.ops.aten.slice.Tensor(add_32, 0, -512, 9223372036854775807);  add_32 = None
        slice_tensor_4: "f32[512, 16, 1024]" = torch.ops.aten.slice.Tensor(add_43, 0, -512, 9223372036854775807);  add_43 = None
        slice_tensor_5: "f32[512, 16, 1024]" = torch.ops.aten.slice.Tensor(add_54, 0, -512, 9223372036854775807);  add_54 = None
        slice_tensor_6: "f32[512, 16, 1024]" = torch.ops.aten.slice.Tensor(add_65, 0, -512, 9223372036854775807);  add_65 = None
        slice_tensor_7: "f32[512, 16, 1024]" = torch.ops.aten.slice.Tensor(add_76, 0, -512, 9223372036854775807);  add_76 = None
        slice_tensor_8: "f32[512, 16, 1024]" = torch.ops.aten.slice.Tensor(add_87, 0, -512, 9223372036854775807);  add_87 = None
        slice_tensor_9: "f32[512, 16, 1024]" = torch.ops.aten.slice.Tensor(add_98, 0, -512, 9223372036854775807);  add_98 = None
        slice_tensor_10: "f32[512, 16, 1024]" = torch.ops.aten.slice.Tensor(add_109, 0, -512, 9223372036854775807);  add_109 = None
        slice_tensor_11: "f32[512, 16, 1024]" = torch.ops.aten.slice.Tensor(add_120, 0, -512, 9223372036854775807);  add_120 = None
        slice_tensor_12: "f32[512, 16, 1024]" = torch.ops.aten.slice.Tensor(add_131, 0, -512, 9223372036854775807);  add_131 = None
        slice_tensor_13: "f32[512, 16, 1024]" = torch.ops.aten.slice.Tensor(add_142, 0, -512, 9223372036854775807);  add_142 = None
        slice_tensor_14: "f32[512, 16, 1024]" = torch.ops.aten.slice.Tensor(add_153, 0, -512, 9223372036854775807);  add_153 = None
        slice_tensor_15: "f32[512, 16, 1024]" = torch.ops.aten.slice.Tensor(add_164, 0, -512, 9223372036854775807);  add_164 = None
        slice_tensor_16: "f32[512, 16, 1024]" = torch.ops.aten.slice.Tensor(add_175, 0, -512, 9223372036854775807);  add_175 = None
        slice_tensor_17: "f32[512, 16, 1024]" = torch.ops.aten.slice.Tensor(add_186, 0, -512, 9223372036854775807);  add_186 = None
        slice_tensor_18: "f32[512, 16, 1024]" = torch.ops.aten.slice.Tensor(add_197, 0, -512, 9223372036854775807);  add_197 = None
        slice_tensor_19: "f32[512, 16, 1024]" = torch.ops.aten.slice.Tensor(add_208, 0, -512, 9223372036854775807);  add_208 = None
        slice_tensor_20: "f32[512, 16, 1024]" = torch.ops.aten.slice.Tensor(add_219, 0, -512, 9223372036854775807);  add_219 = None
        slice_tensor_21: "f32[512, 16, 1024]" = torch.ops.aten.slice.Tensor(add_230, 0, -512, 9223372036854775807);  add_230 = None
        slice_tensor_22: "f32[512, 16, 1024]" = torch.ops.aten.slice.Tensor(add_241, 0, -512, 9223372036854775807);  add_241 = None
        slice_tensor_23: "f32[512, 16, 1024]" = torch.ops.aten.slice.Tensor(add_252, 0, -512, 9223372036854775807);  add_252 = None
        return (div_tensor, slice_tensor, slice_tensor_1, slice_tensor_2, slice_tensor_3, slice_tensor_4, slice_tensor_5, slice_tensor_6, slice_tensor_7, slice_tensor_8, slice_tensor_9, slice_tensor_10, slice_tensor_11, slice_tensor_12, slice_tensor_13, slice_tensor_14, slice_tensor_15, slice_tensor_16, slice_tensor_17, slice_tensor_18, slice_tensor_19, slice_tensor_20, slice_tensor_21, slice_tensor_22, slice_tensor_23)


def _default_make_inputs():
    return [
    torch.randint(0, 512, [16, 512], dtype=torch.int64, device='cuda'),
    torch.randn([8192, 32000], dtype=torch.float32, device='cuda'),
    torch.randn([512, 16, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([512, 16, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([512, 16, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([512, 16, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([512, 16, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([512, 16, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([512, 16, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([512, 16, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([512, 16, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([512, 16, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([512, 16, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([512, 16, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([512, 16, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([512, 16, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([512, 16, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([512, 16, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([512, 16, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([512, 16, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([512, 16, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([512, 16, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([512, 16, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([512, 16, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([512, 16, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([512, 16, 1024], dtype=torch.float32, device='cuda'),
    [16, 512, 32000],  # _shape_param_0
    [-1, 32000],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
