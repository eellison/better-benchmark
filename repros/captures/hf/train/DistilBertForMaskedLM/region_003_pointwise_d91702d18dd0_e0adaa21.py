"""
Standalone repro captured via capture_hook.
Label: hf_DistilBertForMaskedLM_train
Pattern hash: d91702d18dd0
Shape hash: e0adaa21
"""
_shapes_config = "(T([32768, 3072], f32), T([768, 3072], f32), S([256, 128, 3072]), S([32768, 3072]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_34: "f32[32768, 3072]", primals_99: "f32[768, 3072]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:224 in ff_chunk, code: x = self.lin1(input)
        reshape_default: "f32[256, 128, 3072]" = torch.ops.aten.reshape.default(addmm_34, _shape_param_0);  addmm_34 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_tensor: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(reshape_default, 0.5)
        mul_tensor_1: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(reshape_default, 0.7071067811865476);  reshape_default = None
        erf_default: "f32[256, 128, 3072]" = torch.ops.aten.erf.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor: "f32[256, 128, 3072]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_2: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  mul_tensor = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:226 in ff_chunk, code: x = self.lin2(x)
        reshape_default_1: "f32[32768, 3072]" = torch.ops.aten.reshape.default(mul_tensor_2, _shape_param_1);  mul_tensor_2 = _shape_param_1 = None
        permute_default: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_99, [1, 0]);  primals_99 = None
        return (reshape_default_1, permute_default)



def make_inputs():
    return [
    torch.randn([32768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    [256, 128, 3072],  # _shape_param_0
    [32768, 3072],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
