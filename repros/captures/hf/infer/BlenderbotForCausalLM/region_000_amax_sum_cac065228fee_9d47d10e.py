"""
Standalone repro captured via capture_hook.
Label: hf_BlenderbotForCausalLM_infer
Pattern hash: cac065228fee
Shape hash: 9d47d10e
"""
_shapes_config = "(T([32, 128], i64), T([4096, 8008], f32), S([32, 128, 8008]), S([-1, 8008]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, arg389_1: "i64[32, 128]", mm: "f32[4096, 8008]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:1004 in forward, code: loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        reshape_default: "i64[4096]" = torch.ops.aten.reshape.default(arg389_1, [-1]);  arg389_1 = None
        ne_scalar: "b8[4096]" = torch.ops.aten.ne.Scalar(reshape_default, -100)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:998 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        reshape_default_1: "f32[32, 128, 8008]" = torch.ops.aten.reshape.default(mm, _shape_param_0);  mm = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:1004 in forward, code: loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        reshape_default_2: "f32[4096, 8008]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_1);  reshape_default_1 = _shape_param_1 = None
        amax_default: "f32[4096, 1]" = torch.ops.aten.amax.default(reshape_default_2, [1], True)
        sub_tensor: "f32[4096, 8008]" = torch.ops.aten.sub.Tensor(reshape_default_2, amax_default);  reshape_default_2 = amax_default = None
        exp_default: "f32[4096, 8008]" = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list: "f32[4096, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True);  exp_default = None
        log_default: "f32[4096, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1: "f32[4096, 8008]" = torch.ops.aten.sub.Tensor(sub_tensor, log_default);  sub_tensor = log_default = None
        ne_scalar_1: "b8[4096]" = torch.ops.aten.ne.Scalar(reshape_default, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[4096]" = torch.ops.aten.where.self(ne_scalar_1, reshape_default, full_default);  ne_scalar_1 = full_default = None
        unsqueeze_default: "i64[4096, 1]" = torch.ops.aten.unsqueeze.default(where_self, 1);  where_self = None
        gather_default: "f32[4096, 1]" = torch.ops.aten.gather.default(sub_tensor_1, 1, unsqueeze_default);  sub_tensor_1 = unsqueeze_default = None
        squeeze_dim: "f32[4096]" = torch.ops.aten.squeeze.dim(gather_default, 1);  gather_default = None
        neg_default: "f32[4096]" = torch.ops.aten.neg.default(squeeze_dim);  squeeze_dim = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f32[4096]" = torch.ops.aten.where.self(ne_scalar, neg_default, full_default_1);  ne_scalar = neg_default = full_default_1 = None
        sum_default: "f32[]" = torch.ops.aten.sum.default(where_self_1);  where_self_1 = None
        ne_scalar_2: "b8[4096]" = torch.ops.aten.ne.Scalar(reshape_default, -100);  reshape_default = None
        sum_default_1: "i64[]" = torch.ops.aten.sum.default(ne_scalar_2);  ne_scalar_2 = None
        convert_element_type_default: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_1, torch.float32);  sum_default_1 = None
        div_tensor: "f32[]" = torch.ops.aten.div.Tensor(sum_default, convert_element_type_default);  sum_default = convert_element_type_default = None
        return div_tensor



def make_inputs():
    return [
    torch.randint(0, 128, [32, 128], dtype=torch.int64, device='cuda'),
    torch.randn([4096, 8008], dtype=torch.float32, device='cuda'),
    [32, 128, 8008],  # _shape_param_0
    [-1, 8008],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
