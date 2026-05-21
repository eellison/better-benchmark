"""
Standalone repro captured via capture_hook.
Label: hf_GPTJForQuestionAnswering_train
Pattern hash: 559e0e3d0eb0
Shape hash: 2cf5e8a0
"""
_shapes_config = "(T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([4096], f32), T([4096], f32), T([2, 4096], f32), S([1, 128, 4096]), S([1, 128, 4096]), S([128, 4096]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_111: "f32[128, 4096]", addmm_55: "f32[128, 4096]", add_245: "f32[1, 128, 4096]", primals_311: "f32[4096]", primals_312: "f32[4096]", primals_313: "f32[2, 4096]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        reshape_default: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_111, _shape_param_0);  mm_111 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        reshape_default_1: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_55, _shape_param_1);  addmm_55 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_tensor: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(reshape_default, reshape_default_1);  reshape_default = reshape_default_1 = None
        add_tensor_1: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_tensor, add_245);  add_tensor = add_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:542 in forward, code: hidden_states = self.ln_f(hidden_states)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_1, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 128, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 128, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_2: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_tensor_1, getitem_1);  add_tensor_1 = getitem_1 = None
        mul_tensor: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_tensor, primals_311);  mul_tensor = primals_311 = None
        add_tensor_3: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_tensor_1, primals_312);  mul_tensor_1 = primals_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:815 in forward, code: logits = self.qa_outputs(sequence_output)
        reshape_default_2: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_tensor_3, _shape_param_2);  add_tensor_3 = _shape_param_2 = None
        permute_default: "f32[4096, 2]" = torch.ops.aten.permute.default(primals_313, [1, 0]);  primals_313 = None
        return (reshape_default_2, permute_default)



def make_inputs():
    return [
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([4096], dtype=torch.float32, device='cuda'),
    torch.randn([4096], dtype=torch.float32, device='cuda'),
    torch.randn([2, 4096], dtype=torch.float32, device='cuda'),
    [1, 128, 4096],  # _shape_param_0
    [1, 128, 4096],  # _shape_param_1
    [128, 4096],  # _shape_param_2
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
