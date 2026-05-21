"""
Standalone repro captured via capture_hook.
Label: hf_GPT2ForSequenceClassification_train
Pattern hash: d7ea2582c273
Shape hash: 8d1da404
"""
_shapes_config = "(T([8192, 3072], f32), T([8192, 3072], f32), T([768, 3072], f32), S([8, 1024, 3072]), S([8, 1024, 3072]), S([8192, 3072]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_91: "f32[8192, 3072]", addmm_2: "f32[8192, 3072]", primals_13: "f32[768, 3072]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        reshape_default: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_91, _shape_param_0);  mm_91 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        reshape_default_1: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_2, _shape_param_1);  addmm_2 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_tensor: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(reshape_default_1, 0.5)
        mul_tensor_1: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(reshape_default, mul_tensor);  mul_tensor = None
        pow_tensor_scalar: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(reshape_default_1, 3.0)
        mul_tensor_2: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_tensor_scalar, 0.044715);  pow_tensor_scalar = None
        add_tensor: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(reshape_default_1, mul_tensor_2);  mul_tensor_2 = None
        mul_tensor_3: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_tensor, 0.7978845608028654);  add_tensor = None
        tanh_default: "f32[8, 1024, 3072]" = torch.ops.aten.tanh.default(mul_tensor_3);  mul_tensor_3 = None
        add_tensor_1: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_default, 1.0)
        mul_tensor_4: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(reshape_default, add_tensor_1);  reshape_default = add_tensor_1 = None
        mul_tensor_5: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(tanh_default, tanh_default);  tanh_default = None
        sub_tensor: "f32[8, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_tensor_5);  mul_tensor_5 = None
        mul_tensor_6: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_tensor_1, sub_tensor);  mul_tensor_1 = sub_tensor = None
        mul_tensor_7: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_tensor_6, 0.7978845608028654);  mul_tensor_6 = None
        mul_tensor_8: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_tensor_7, 0.044715)
        pow_tensor_scalar_1: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(reshape_default_1, 2.0);  reshape_default_1 = None
        mul_scalar: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 3.0);  pow_tensor_scalar_1 = None
        mul_tensor_9: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_tensor_8, mul_scalar);  mul_tensor_8 = mul_scalar = None
        add_tensor_2: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_tensor_7, mul_tensor_9);  mul_tensor_7 = mul_tensor_9 = None
        mul_tensor_10: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_tensor_4, 0.5);  mul_tensor_4 = None
        add_tensor_3: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_10);  add_tensor_2 = mul_tensor_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        reshape_default_2: "f32[8192, 3072]" = torch.ops.aten.reshape.default(add_tensor_3, _shape_param_2);  add_tensor_3 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_default: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
        return (reshape_default_2, permute_default)



def make_inputs():
    return [
    torch.randn([8192, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    [8, 1024, 3072],  # _shape_param_0
    [8, 1024, 3072],  # _shape_param_1
    [8192, 3072],  # _shape_param_2
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
