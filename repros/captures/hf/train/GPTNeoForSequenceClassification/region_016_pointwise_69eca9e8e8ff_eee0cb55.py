"""
Standalone repro captured via capture_hook.
Label: hf_GPTNeoForSequenceClassification_train
Pattern hash: 69eca9e8e8ff
Shape hash: eee0cb55
"""
_shapes_config = "(T([4096, 8192], f32), T([4096, 8192], f32), T([8192, 2048], f32), S([32, 128, 8192]), S([32, 128, 8192]), S([4096, 8192]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_351: "f32[4096, 8192]", addmm_1: "f32[4096, 8192]", primals_14: "f32[8192, 2048]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        reshape_default: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(mm_351, _shape_param_0);  mm_351 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        reshape_default_1: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_1, _shape_param_1);  addmm_1 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_tensor: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(reshape_default_1, 0.5)
        mul_tensor_1: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(reshape_default, mul_tensor);  mul_tensor = None
        pow_tensor_scalar: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(reshape_default_1, 3.0)
        mul_tensor_2: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_tensor_scalar, 0.044715);  pow_tensor_scalar = None
        add_tensor: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(reshape_default_1, mul_tensor_2);  mul_tensor_2 = None
        mul_tensor_3: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_tensor, 0.7978845608028654);  add_tensor = None
        tanh_default: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_tensor_3);  mul_tensor_3 = None
        add_tensor_1: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_default, 1.0)
        mul_tensor_4: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(reshape_default, add_tensor_1);  reshape_default = add_tensor_1 = None
        mul_tensor_5: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_default, tanh_default);  tanh_default = None
        sub_tensor: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_tensor_5);  mul_tensor_5 = None
        mul_tensor_6: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_tensor_1, sub_tensor);  mul_tensor_1 = sub_tensor = None
        mul_tensor_7: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_tensor_6, 0.7978845608028654);  mul_tensor_6 = None
        mul_tensor_8: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_tensor_7, 0.044715)
        pow_tensor_scalar_1: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(reshape_default_1, 2.0);  reshape_default_1 = None
        mul_scalar: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 3.0);  pow_tensor_scalar_1 = None
        mul_tensor_9: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_tensor_8, mul_scalar);  mul_tensor_8 = mul_scalar = None
        add_tensor_2: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(mul_tensor_7, mul_tensor_9);  mul_tensor_7 = mul_tensor_9 = None
        mul_tensor_10: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_tensor_4, 0.5);  mul_tensor_4 = None
        add_tensor_3: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_10);  add_tensor_2 = mul_tensor_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        reshape_default_2: "f32[4096, 8192]" = torch.ops.aten.reshape.default(add_tensor_3, _shape_param_2);  add_tensor_3 = _shape_param_2 = None
        permute_default: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_14, [1, 0]);  primals_14 = None
        permute_default_1: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_2, permute_default_1)



def make_inputs():
    return [
    torch.randn([4096, 8192], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 8192], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 2048], dtype=torch.float32, device='cuda'),
    [32, 128, 8192],  # _shape_param_0
    [32, 128, 8192],  # _shape_param_1
    [4096, 8192],  # _shape_param_2
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
