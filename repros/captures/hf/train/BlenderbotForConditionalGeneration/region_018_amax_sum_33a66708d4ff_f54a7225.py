"""
Standalone repro captured via capture_hook.
Label: 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_train
Pattern hash: 33a66708d4ff
Shape hash: f54a7225
"""
_shapes_config = "(T([2048, 8008], f32), T([1, 8008], f32), T([16, 128], i64), S([16, 128, 8008]), S([-1, 8008]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm: "f32[2048, 8008]", primals_3: "f32[1, 8008]", primals_4: "i64[16, 128]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:885 in torch_dynamo_resume_in_forward_at_872, code: lm_logits = self.lm_head(outputs[0])
        reshape_default: "f32[16, 128, 8008]" = torch.ops.aten.reshape.default(mm, _shape_param_0);  mm = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:886 in torch_dynamo_resume_in_forward_at_872, code: lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)
        add_tensor: "f32[16, 128, 8008]" = torch.ops.aten.add.Tensor(reshape_default, primals_3);  reshape_default = primals_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:892 in torch_dynamo_resume_in_forward_at_872, code: masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        reshape_default_1: "f32[2048, 8008]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_1);  add_tensor = _shape_param_1 = None
        reshape_default_2: "i64[2048]" = torch.ops.aten.reshape.default(primals_4, [-1]);  primals_4 = None
        amax_default: "f32[2048, 1]" = torch.ops.aten.amax.default(reshape_default_1, [1], True)
        sub_tensor: "f32[2048, 8008]" = torch.ops.aten.sub.Tensor(reshape_default_1, amax_default);  reshape_default_1 = amax_default = None
        exp_default: "f32[2048, 8008]" = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True);  exp_default = None
        log_default: "f32[2048, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1: "f32[2048, 8008]" = torch.ops.aten.sub.Tensor(sub_tensor, log_default);  sub_tensor = log_default = None
        ne_scalar: "b8[2048]" = torch.ops.aten.ne.Scalar(reshape_default_2, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[2048]" = torch.ops.aten.where.self(ne_scalar, reshape_default_2, full_default);  reshape_default_2 = full_default = None
        unsqueeze_default: "i64[2048, 1]" = torch.ops.aten.unsqueeze.default(where_self, 1);  where_self = None
        gather_default: "f32[2048, 1]" = torch.ops.aten.gather.default(sub_tensor_1, 1, unsqueeze_default);  sub_tensor_1 = unsqueeze_default = None
        squeeze_dim: "f32[2048]" = torch.ops.aten.squeeze.dim(gather_default, 1);  gather_default = None
        neg_default: "f32[2048]" = torch.ops.aten.neg.default(squeeze_dim);  squeeze_dim = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f32[2048]" = torch.ops.aten.where.self(ne_scalar, neg_default, full_default_1);  neg_default = full_default_1 = None
        sum_default: "i64[]" = torch.ops.aten.sum.default(ne_scalar);  ne_scalar = None
        convert_element_type_default: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default, torch.float32);  sum_default = None
        sum_default_1: "f32[]" = torch.ops.aten.sum.default(where_self_1);  where_self_1 = None
        div_tensor: "f32[]" = torch.ops.aten.div.Tensor(sum_default_1, convert_element_type_default);  sum_default_1 = convert_element_type_default = None
        return div_tensor



def make_inputs():
    return [
    torch.randn([2048, 8008], dtype=torch.float32, device='cuda'),
    torch.randn([1, 8008], dtype=torch.float32, device='cuda'),
    torch.randint(0, 128, [16, 128], dtype=torch.int64, device='cuda'),
    [16, 128, 8008],  # _shape_param_0
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
