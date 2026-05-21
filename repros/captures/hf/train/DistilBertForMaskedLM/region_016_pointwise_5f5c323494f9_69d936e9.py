"""
Standalone repro captured via capture_hook.
Label: hf_DistilBertForMaskedLM_train
Pattern hash: 5f5c323494f9
Shape hash: 69d936e9
"""
_shapes_config = "(T([32768, 3072], f32), T([32768, 3072], f32), T([3072, 768], f32), S([256, 128, 3072]), S([256, 128, 3072]), S([32768, 3072]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_64: "f32[32768, 3072]", addmm_4: "f32[32768, 3072]", primals_17: "f32[3072, 768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:226 in ff_chunk, code: x = self.lin2(x)
        reshape_default: "f32[256, 128, 3072]" = torch.ops.aten.reshape.default(mm_64, _shape_param_0);  mm_64 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:224 in ff_chunk, code: x = self.lin1(input)
        reshape_default_1: "f32[256, 128, 3072]" = torch.ops.aten.reshape.default(addmm_4, _shape_param_1);  addmm_4 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_tensor: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(reshape_default_1, 0.7071067811865476)
        erf_default: "f32[256, 128, 3072]" = torch.ops.aten.erf.default(mul_tensor);  mul_tensor = None
        add_tensor: "f32[256, 128, 3072]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_1: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(add_tensor, 0.5);  add_tensor = None
        mul_tensor_2: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(reshape_default_1, reshape_default_1)
        mul_tensor_3: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_tensor_2, -0.5);  mul_tensor_2 = None
        exp_default: "f32[256, 128, 3072]" = torch.ops.aten.exp.default(mul_tensor_3);  mul_tensor_3 = None
        mul_tensor_4: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_default, 0.3989422804014327);  exp_default = None
        mul_tensor_5: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(reshape_default_1, mul_tensor_4);  reshape_default_1 = mul_tensor_4 = None
        add_tensor_1: "f32[256, 128, 3072]" = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_5);  mul_tensor_1 = mul_tensor_5 = None
        mul_tensor_6: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(reshape_default, add_tensor_1);  reshape_default = add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:224 in ff_chunk, code: x = self.lin1(input)
        reshape_default_2: "f32[32768, 3072]" = torch.ops.aten.reshape.default(mul_tensor_6, _shape_param_2);  mul_tensor_6 = _shape_param_2 = None
        permute_default: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_17, [1, 0]);  primals_17 = None
        permute_default_1: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_2, permute_default_1)



def make_inputs():
    return [
    torch.randn([32768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([32768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    [256, 128, 3072],  # _shape_param_0
    [256, 128, 3072],  # _shape_param_1
    [32768, 3072],  # _shape_param_2
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
