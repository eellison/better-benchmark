"""
Standalone repro captured via capture_hook.
Label: hf_GPTNeoForSequenceClassification_train
Pattern hash: 9e14b9e8936c
Shape hash: ca3606d2
"""
_shapes_config = "(T([4096, 8192], f32), T([2048, 8192], f32), S([32, 128, 8192]), S([4096, 8192]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_70: "f32[4096, 8192]", primals_338: "f32[2048, 8192]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        reshape_default: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_70, _shape_param_0);  addmm_70 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_tensor: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(reshape_default, 0.5)
        pow_tensor_scalar: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(reshape_default, 3.0)
        mul_tensor_1: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_tensor_scalar, 0.044715);  pow_tensor_scalar = None
        add_tensor: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(reshape_default, mul_tensor_1);  reshape_default = mul_tensor_1 = None
        mul_tensor_2: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_tensor, 0.7978845608028654);  add_tensor = None
        tanh_default: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_tensor_2);  mul_tensor_2 = None
        add_tensor_1: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_default, 1.0);  tanh_default = None
        mul_tensor_3: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor_1);  mul_tensor = add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        reshape_default_1: "f32[4096, 8192]" = torch.ops.aten.reshape.default(mul_tensor_3, _shape_param_1);  mul_tensor_3 = _shape_param_1 = None
        permute_default: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_338, [1, 0]);  primals_338 = None
        return (reshape_default_1, permute_default)



def make_inputs():
    return [
    torch.randn([4096, 8192], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 8192], dtype=torch.float32, device='cuda'),
    [32, 128, 8192],  # _shape_param_0
    [4096, 8192],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
