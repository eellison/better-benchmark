"""
Standalone repro captured via capture_hook.
Label: hf_GPTNeoForSequenceClassification_train
Pattern hash: 066cf404f9c7
Shape hash: 6e041acd
"""
_shapes_config = "(T([], f32), T([], f32), T([32, 2], b8), T([32, 1], b8), T([32, 2], f32), T([32, 2], f32), T([32], i64), T([32], i64), T([2, 2048], f32), S([4096, 2]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, tangents_1: "f32[]", convert_element_type_1: "f32[]", eq_tensor: "b8[32, 2]", ne_5: "b8[32, 1]", index_2: "f32[32, 2]", tangents_2: "f32[32, 2]", iota_1: "i64[32]", argmax: "i64[32]", primals_342: "f32[2, 2048]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:718 in forward, code: loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
        div_tensor: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_1);  tangents_1 = convert_element_type_1 = None

        # No stacktrace found for following nodes
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:718 in forward, code: loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
        where_self: "f32[32, 2]" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:718 in forward, code: loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
        where_self_1: "f32[32, 1]" = torch.ops.aten.where.self(ne_5, div_tensor, full_default);  ne_5 = div_tensor = full_default = None
        mul_tensor: "f32[32, 2]" = torch.ops.aten.mul.Tensor(where_self, where_self_1);  where_self = where_self_1 = None
        amax_default: "f32[32, 1]" = torch.ops.aten.amax.default(index_2, [1], True)
        sub_tensor: "f32[32, 2]" = torch.ops.aten.sub.Tensor(index_2, amax_default);  index_2 = amax_default = None
        exp_default: "f32[32, 2]" = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list: "f32[32, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True);  exp_default = None
        log_default: "f32[32, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1: "f32[32, 2]" = torch.ops.aten.sub.Tensor(sub_tensor, log_default);  sub_tensor = log_default = None
        exp_default_1: "f32[32, 2]" = torch.ops.aten.exp.default(sub_tensor_1);  sub_tensor_1 = None
        sum_dim_int_list_1: "f32[32, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [1], True)
        mul_tensor_1: "f32[32, 2]" = torch.ops.aten.mul.Tensor(exp_default_1, sum_dim_int_list_1);  exp_default_1 = sum_dim_int_list_1 = None
        sub_tensor_2: "f32[32, 2]" = torch.ops.aten.sub.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None
        add_tensor: "f32[32, 2]" = torch.ops.aten.add.Tensor(tangents_2, sub_tensor_2);  tangents_2 = sub_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:698 in forward, code: pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]
        full_default_1: "f32[32, 128, 2]" = torch.ops.aten.full.default([32, 128, 2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[32, 128, 2]" = torch.ops.aten.index_put.default(full_default_1, [iota_1, argmax], add_tensor, True);  full_default_1 = iota_1 = argmax = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:675 in forward, code: logits = self.score(hidden_states)
        reshape_default: "f32[4096, 2]" = torch.ops.aten.reshape.default(index_put_default, _shape_param_0);  index_put_default = _shape_param_0 = None
        permute_default: "f32[2048, 2]" = torch.ops.aten.permute.default(primals_342, [1, 0]);  primals_342 = None
        permute_default_1: "f32[2, 2048]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default, permute_default_1)



def make_inputs():
    return [
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [32, 2], dtype=torch.bool, device='cuda'),
    torch.randint(0, 2, [32, 1], dtype=torch.bool, device='cuda'),
    torch.randn([32, 2], dtype=torch.float32, device='cuda'),
    torch.randn([32, 2], dtype=torch.float32, device='cuda'),
    torch.randint(0, 32, [32], dtype=torch.int64, device='cuda'),
    torch.randint(0, 32, [32], dtype=torch.int64, device='cuda'),
    torch.randn([2, 2048], dtype=torch.float32, device='cuda'),
    [4096, 2],  # _shape_param_0
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
