"""
Standalone repro captured via capture_hook.
Label: hf_ElectraForCausalLM_train
Pattern hash: 7b4ee3061252
Shape hash: c1378116
"""
_shapes_config = "(T([32768, 128], f32), T([128], f32), T([32768, 128], f32), T([64, 512, 1], f32), T([64, 512, 1], f32), T([128, 256], f32), S([64, 512, 128]), S([64, 512, 128]), S([32768, 128]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_default: "f32[32768, 128]", primals_206: "f32[128]", addmm_73: "f32[32768, 128]", getitem_51: "f32[64, 512, 1]", rsqrt_25: "f32[64, 512, 1]", primals_204: "f32[128, 256]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:1351 in forward, code: logits = self.generator_lm_head(self.generator_predictions(hidden_states[:, slice_indices, :]))
        reshape_default: "f32[64, 512, 128]" = torch.ops.aten.reshape.default(mm_default, _shape_param_0);  mm_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:501 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        mul_tensor: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(reshape_default, primals_206);  reshape_default = primals_206 = None
        mul_tensor_1: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor, 128)
        sum_dim_int_list: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:499 in forward, code: hidden_states = self.dense(generator_hidden_states)
        reshape_default_1: "f32[64, 512, 128]" = torch.ops.aten.reshape.default(addmm_73, _shape_param_1);  addmm_73 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_tensor_2: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(reshape_default_1, 0.5)
        mul_tensor_3: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(reshape_default_1, 0.7071067811865476)
        erf_default: "f32[64, 512, 128]" = torch.ops.aten.erf.default(mul_tensor_3);  mul_tensor_3 = None
        add_tensor: "f32[64, 512, 128]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_4: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_2, add_tensor);  mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:501 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        sub_tensor: "f32[64, 512, 128]" = torch.ops.aten.sub.Tensor(mul_tensor_4, getitem_51);  mul_tensor_4 = getitem_51 = None
        mul_tensor_5: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_25);  sub_tensor = None
        mul_tensor_6: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_5);  mul_tensor = None
        sum_dim_int_list_1: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [2], True);  mul_tensor_6 = None
        mul_tensor_7: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_5, sum_dim_int_list_1);  mul_tensor_5 = sum_dim_int_list_1 = None
        sub_tensor_1: "f32[64, 512, 128]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2: "f32[64, 512, 128]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_7);  sub_tensor_1 = mul_tensor_7 = None
        div_tensor: "f32[64, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 128);  rsqrt_25 = None
        mul_tensor_8: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_tensor_9: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(add_tensor, 0.5);  add_tensor = None
        mul_tensor_10: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(reshape_default_1, reshape_default_1)
        mul_tensor_11: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_10, -0.5);  mul_tensor_10 = None
        exp_default: "f32[64, 512, 128]" = torch.ops.aten.exp.default(mul_tensor_11);  mul_tensor_11 = None
        mul_tensor_12: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(exp_default, 0.3989422804014327);  exp_default = None
        mul_tensor_13: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(reshape_default_1, mul_tensor_12);  reshape_default_1 = mul_tensor_12 = None
        add_tensor_1: "f32[64, 512, 128]" = torch.ops.aten.add.Tensor(mul_tensor_9, mul_tensor_13);  mul_tensor_9 = mul_tensor_13 = None
        mul_tensor_14: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_8, add_tensor_1);  mul_tensor_8 = add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:499 in forward, code: hidden_states = self.dense(generator_hidden_states)
        reshape_default_2: "f32[32768, 128]" = torch.ops.aten.reshape.default(mul_tensor_14, _shape_param_2);  mul_tensor_14 = _shape_param_2 = None
        permute_default: "f32[256, 128]" = torch.ops.aten.permute.default(primals_204, [1, 0]);  primals_204 = None
        permute_default_1: "f32[128, 256]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_2, permute_default_1)



def make_inputs():
    return [
    torch.randn([32768, 128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([32768, 128], dtype=torch.float32, device='cuda'),
    torch.randn([64, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([64, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128, 256], dtype=torch.float32, device='cuda'),
    [64, 512, 128],  # _shape_param_0
    [64, 512, 128],  # _shape_param_1
    [32768, 128],  # _shape_param_2
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
