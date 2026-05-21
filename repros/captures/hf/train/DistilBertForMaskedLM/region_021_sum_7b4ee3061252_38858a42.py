"""
Standalone repro captured via capture_hook.
Label: hf_DistilBertForMaskedLM_train
Pattern hash: 7b4ee3061252
Shape hash: 38858a42
"""
_shapes_config = "(T([32768, 768], f32), T([768], f32), T([32768, 768], f32), T([256, 128, 1], f32), T([256, 128, 1], f32), T([768, 768], f32), S([256, 128, 768]), S([256, 128, 768]), S([32768, 768]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_default_1: "f32[32768, 768]", primals_105: "f32[768]", addmm_36: "f32[32768, 768]", getitem_27: "f32[256, 128, 1]", rsqrt_13: "f32[256, 128, 1]", primals_103: "f32[768, 768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:517 in forward, code: prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)
        reshape_default: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(mm_default_1, _shape_param_0);  mm_default_1 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:516 in forward, code: prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
        mul_tensor: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(reshape_default, primals_105);  reshape_default = primals_105 = None
        mul_tensor_1: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:514 in forward, code: prediction_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
        reshape_default_1: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(addmm_36, _shape_param_1);  addmm_36 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_tensor_2: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(reshape_default_1, 0.5)
        mul_tensor_3: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(reshape_default_1, 0.7071067811865476)
        erf_default: "f32[256, 128, 768]" = torch.ops.aten.erf.default(mul_tensor_3);  mul_tensor_3 = None
        add_tensor: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_4: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, add_tensor);  mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:516 in forward, code: prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
        sub_tensor: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_4, getitem_27);  mul_tensor_4 = getitem_27 = None
        mul_tensor_5: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_13);  sub_tensor = None
        mul_tensor_6: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_5);  mul_tensor = None
        sum_dim_int_list_1: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [2], True);  mul_tensor_6 = None
        mul_tensor_7: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_5, sum_dim_int_list_1);  mul_tensor_5 = sum_dim_int_list_1 = None
        sub_tensor_1: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_7);  sub_tensor_1 = mul_tensor_7 = None
        div_tensor: "f32[256, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
        mul_tensor_8: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_tensor_9: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_tensor, 0.5);  add_tensor = None
        mul_tensor_10: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(reshape_default_1, reshape_default_1)
        mul_tensor_11: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_10, -0.5);  mul_tensor_10 = None
        exp_default: "f32[256, 128, 768]" = torch.ops.aten.exp.default(mul_tensor_11);  mul_tensor_11 = None
        mul_tensor_12: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(exp_default, 0.3989422804014327);  exp_default = None
        mul_tensor_13: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(reshape_default_1, mul_tensor_12);  reshape_default_1 = mul_tensor_12 = None
        add_tensor_1: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_tensor_9, mul_tensor_13);  mul_tensor_9 = mul_tensor_13 = None
        mul_tensor_14: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_8, add_tensor_1);  mul_tensor_8 = add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:514 in forward, code: prediction_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
        reshape_default_2: "f32[32768, 768]" = torch.ops.aten.reshape.default(mul_tensor_14, _shape_param_2);  mul_tensor_14 = _shape_param_2 = None
        permute_default: "f32[768, 768]" = torch.ops.aten.permute.default(primals_103, [1, 0]);  primals_103 = None
        permute_default_1: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_2, permute_default_1)



def make_inputs():
    return [
    torch.randn([32768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([32768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([256, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([256, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    [256, 128, 768],  # _shape_param_0
    [256, 128, 768],  # _shape_param_1
    [32768, 768],  # _shape_param_2
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
