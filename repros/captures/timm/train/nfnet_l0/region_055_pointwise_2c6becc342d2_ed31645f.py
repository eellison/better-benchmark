"""
Standalone repro captured via capture_hook.
Label: timm_nfnet_l0_train
Pattern hash: 2c6becc342d2
Shape hash: ed31645f
"""
_shapes_config = "(T([128, 1536, 7, 7], f32, stride=(75264, 1, 10752, 1536)), T([128, 1536, 1, 1], f32), T([128, 1536, 1, 1], f32), S([128, 1536, 7, 7]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mul_360: "f32[128, 1536, 7, 7]", sigmoid_9: "f32[128, 1536, 1, 1]", getitem_156: "f32[128, 1536, 1, 1]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_tensor: "f32[128, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_360, sigmoid_9);  mul_360 = sigmoid_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_default: "f32[128, 1536, 7, 7]" = torch.ops.aten.expand.default(getitem_156, _shape_param_0);  getitem_156 = _shape_param_0 = None
        div_scalar: "f32[128, 1536, 7, 7]" = torch.ops.aten.div.Scalar(expand_default, 49);  expand_default = None
        add_tensor: "f32[128, 1536, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor, div_scalar);  mul_tensor = div_scalar = None
        return add_tensor



def make_inputs():
    return [
    torch.randn(9633792, dtype=torch.float32, device='cuda').as_strided([128, 1536, 7, 7], [75264, 1, 10752, 1536]),  # mul_360
    torch.randn([128, 1536, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128, 1536, 1, 1], dtype=torch.float32, device='cuda'),
    [128, 1536, 7, 7],  # _shape_param_0
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
