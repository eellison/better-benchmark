"""
Standalone repro captured via capture_hook.
Label: hf_PLBartForCausalLM_train
Pattern hash: cd1d18c11a67
Shape hash: 13271f7e
"""
_shapes_config = "(T([], f32), T([], f32), T([16, 1024], i64), T([16, 1024, 50005], f32, stride=(51208192, 50008, 1)), T([16384, 1], f32), T([16384, 1], f32), T([16, 1024, 50005], f32), T([50005, 768], f32), S([1, 50005]), S([16384, 50005]), S([-1, 50005]), S([16, 1024, 50005]), S([16384, 50005]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, tangents_1: "f32[]", convert_element_type: "f32[]", primals_3: "i64[16, 1024]", view_1: "f32[16, 1024, 50005]", amax: "f32[16384, 1]", log: "f32[16384, 1]", tangents_2: "f32[16, 1024, 50005]", primals_2: "f32[50005, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:1131 in torch_dynamo_resume_in_forward_at_1111, code: loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        div_tensor: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type);  tangents_1 = convert_element_type = None
        reshape_default: "i64[16384]" = torch.ops.aten.reshape.default(primals_3, [-1]);  primals_3 = None
        unsqueeze_default: "i64[16384, 1]" = torch.ops.aten.unsqueeze.default(reshape_default, 1);  reshape_default = None
        ne_scalar: "b8[16384, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_default, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[16384, 1]" = torch.ops.aten.where.self(ne_scalar, unsqueeze_default, full_default);  unsqueeze_default = full_default = None

        # No stacktrace found for following nodes
        iota_default: "i64[50005]" = torch.ops.prims.iota.default(50005, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        reshape_default_1: "i64[1, 50005]" = torch.ops.aten.reshape.default(iota_default, _shape_param_0);  iota_default = _shape_param_0 = None
        expand_default: "i64[16384, 50005]" = torch.ops.aten.expand.default(where_self, _shape_param_1);  where_self = _shape_param_1 = None
        eq_tensor: "b8[16384, 50005]" = torch.ops.aten.eq.Tensor(expand_default, reshape_default_1);  expand_default = reshape_default_1 = None
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:1131 in torch_dynamo_resume_in_forward_at_1111, code: loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        where_self_1: "f32[16384, 50005]" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_2: "f32[16384, 1]" = torch.ops.aten.where.self(ne_scalar, div_tensor, full_default_1);  ne_scalar = div_tensor = full_default_1 = None
        mul_tensor: "f32[16384, 50005]" = torch.ops.aten.mul.Tensor(where_self_1, where_self_2);  where_self_1 = where_self_2 = None
        reshape_default_2: "f32[16384, 50005]" = torch.ops.aten.reshape.default(view_1, _shape_param_2);  view_1 = _shape_param_2 = None
        sub_tensor: "f32[16384, 50005]" = torch.ops.aten.sub.Tensor(reshape_default_2, amax);  reshape_default_2 = amax = None
        sub_tensor_1: "f32[16384, 50005]" = torch.ops.aten.sub.Tensor(sub_tensor, log);  sub_tensor = log = None
        exp_default: "f32[16384, 50005]" = torch.ops.aten.exp.default(sub_tensor_1);  sub_tensor_1 = None
        sum_dim_int_list: "f32[16384, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [1], True)
        mul_tensor_1: "f32[16384, 50005]" = torch.ops.aten.mul.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        sub_tensor_2: "f32[16384, 50005]" = torch.ops.aten.sub.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None
        reshape_default_3: "f32[16, 1024, 50005]" = torch.ops.aten.reshape.default(sub_tensor_2, _shape_param_3);  sub_tensor_2 = _shape_param_3 = None
        add_tensor: "f32[16, 1024, 50005]" = torch.ops.aten.add.Tensor(tangents_2, reshape_default_3);  tangents_2 = reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:1125 in torch_dynamo_resume_in_forward_at_1111, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        reshape_default_4: "f32[16384, 50005]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_4);  add_tensor = _shape_param_4 = None
        permute_default: "f32[50005, 16384]" = torch.ops.aten.permute.default(reshape_default_4, [1, 0])
        constant_pad_nd_default: "f32[50008, 16384]" = torch.ops.aten.constant_pad_nd.default(permute_default, [0, 0, 0, 3]);  permute_default = None
        permute_default_1: "f32[768, 50005]" = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
        permute_default_2: "f32[50005, 768]" = torch.ops.aten.permute.default(permute_default_1, [1, 0]);  permute_default_1 = None
        constant_pad_nd_default_1: "f32[16384, 50008]" = torch.ops.aten.constant_pad_nd.default(reshape_default_4, [0, 3, 0, 0]);  reshape_default_4 = None
        constant_pad_nd_default_2: "f32[50008, 768]" = torch.ops.aten.constant_pad_nd.default(permute_default_2, [0, 0, 0, 3]);  permute_default_2 = None
        return (constant_pad_nd_default, constant_pad_nd_default_1, constant_pad_nd_default_2)



def make_inputs():
    return [
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randint(0, 1024, [16, 1024], dtype=torch.int64, device='cuda'),
    torch.randn(819331069, dtype=torch.float32, device='cuda').as_strided([16, 1024, 50005], [51208192, 50008, 1]),  # view_1
    torch.randn([16384, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 1024, 50005], dtype=torch.float32, device='cuda'),
    torch.randn([50005, 768], dtype=torch.float32, device='cuda'),
    [1, 50005],  # _shape_param_0
    [16384, 50005],  # _shape_param_1
    [-1, 50005],  # _shape_param_2
    [16, 1024, 50005],  # _shape_param_3
    [16384, 50005],  # _shape_param_4
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
