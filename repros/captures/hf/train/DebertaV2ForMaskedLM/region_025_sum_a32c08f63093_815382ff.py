"""
Standalone repro captured via capture_hook.
Label: hf_DebertaV2ForMaskedLM_train
Pattern hash: a32c08f63093
Shape hash: 815382ff
"""
_shapes_config = "(T([], f32), T([], f32), T([8, 512], i64), T([8, 512, 128100], f32), T([4096, 1], f32), T([4096, 1], f32), T([8, 512, 128100], f32), T([128100, 1536], f32), S([1, 128100]), S([4096, 128100]), S([-1, 128100]), S([8, 512, 128100]), S([4096, 128100]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, tangents_1: "f32[]", convert_element_type_24: "f32[]", primals_396: "i64[8, 512]", view_531: "f32[8, 512, 128100]", amax_24: "f32[4096, 1]", log: "f32[4096, 1]", tangents_2: "f32[8, 512, 128100]", primals_3: "f32[128100, 1536]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:968 in forward, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        div_tensor: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_24);  tangents_1 = convert_element_type_24 = None
        reshape_default: "i64[4096]" = torch.ops.aten.reshape.default(primals_396, [-1]);  primals_396 = None
        unsqueeze_default: "i64[4096, 1]" = torch.ops.aten.unsqueeze.default(reshape_default, 1);  reshape_default = None
        ne_scalar: "b8[4096, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_default, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[4096, 1]" = torch.ops.aten.where.self(ne_scalar, unsqueeze_default, full_default);  unsqueeze_default = full_default = None

        # No stacktrace found for following nodes
        iota_default: "i64[128100]" = torch.ops.prims.iota.default(128100, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        reshape_default_1: "i64[1, 128100]" = torch.ops.aten.reshape.default(iota_default, _shape_param_0);  iota_default = _shape_param_0 = None
        expand_default: "i64[4096, 128100]" = torch.ops.aten.expand.default(where_self, _shape_param_1);  where_self = _shape_param_1 = None
        eq_tensor: "b8[4096, 128100]" = torch.ops.aten.eq.Tensor(expand_default, reshape_default_1);  expand_default = reshape_default_1 = None
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:968 in forward, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        where_self_1: "f32[4096, 128100]" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_2: "f32[4096, 1]" = torch.ops.aten.where.self(ne_scalar, div_tensor, full_default_1);  ne_scalar = div_tensor = full_default_1 = None
        mul_tensor: "f32[4096, 128100]" = torch.ops.aten.mul.Tensor(where_self_1, where_self_2);  where_self_1 = where_self_2 = None
        reshape_default_2: "f32[4096, 128100]" = torch.ops.aten.reshape.default(view_531, _shape_param_2);  view_531 = _shape_param_2 = None
        sub_tensor: "f32[4096, 128100]" = torch.ops.aten.sub.Tensor(reshape_default_2, amax_24);  reshape_default_2 = amax_24 = None
        sub_tensor_1: "f32[4096, 128100]" = torch.ops.aten.sub.Tensor(sub_tensor, log);  sub_tensor = log = None
        exp_default: "f32[4096, 128100]" = torch.ops.aten.exp.default(sub_tensor_1);  sub_tensor_1 = None
        sum_dim_int_list: "f32[4096, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [1], True)
        mul_tensor_1: "f32[4096, 128100]" = torch.ops.aten.mul.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        sub_tensor_2: "f32[4096, 128100]" = torch.ops.aten.sub.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None
        reshape_default_3: "f32[8, 512, 128100]" = torch.ops.aten.reshape.default(sub_tensor_2, _shape_param_3);  sub_tensor_2 = _shape_param_3 = None
        add_tensor: "f32[8, 512, 128100]" = torch.ops.aten.add.Tensor(tangents_2, reshape_default_3);  tangents_2 = reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:838 in forward, code: hidden_states = self.decoder(hidden_states)
        reshape_default_4: "f32[4096, 128100]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_4);  add_tensor = _shape_param_4 = None
        permute_default: "f32[1536, 128100]" = torch.ops.aten.permute.default(primals_3, [1, 0]);  primals_3 = None
        permute_default_1: "f32[128100, 1536]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_4, permute_default_1)



def make_inputs():
    return [
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randint(0, 512, [8, 512], dtype=torch.int64, device='cuda'),
    torch.randn([8, 512, 128100], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 128100], dtype=torch.float32, device='cuda'),
    torch.randn([128100, 1536], dtype=torch.float32, device='cuda'),
    [1, 128100],  # _shape_param_0
    [4096, 128100],  # _shape_param_1
    [-1, 128100],  # _shape_param_2
    [8, 512, 128100],  # _shape_param_3
    [4096, 128100],  # _shape_param_4
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
