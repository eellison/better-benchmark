"""
Standalone repro captured via capture_hook.
Label: timm_mobilevit_s_infer
Pattern hash: eecc7f6f5a71
Shape hash: c43a2e69
"""
_shapes_config = "(T([32768, 384], f32), T([192, 384], f32), S([512, 64, 384]), S([32768, 384]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_22: "f32[32768, 384]", arg190_1: "f32[192, 384]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        reshape_default: "f32[512, 64, 384]" = torch.ops.aten.reshape.default(addmm_22, _shape_param_0);  addmm_22 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        neg_default: "f32[512, 64, 384]" = torch.ops.aten.neg.default(reshape_default)
        exp_default: "f32[512, 64, 384]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[512, 64, 384]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[512, 64, 384]" = torch.ops.aten.div.Tensor(reshape_default, add_tensor);  reshape_default = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default_1: "f32[32768, 384]" = torch.ops.aten.reshape.default(div_tensor, _shape_param_1);  div_tensor = _shape_param_1 = None
        permute_default: "f32[384, 192]" = torch.ops.aten.permute.default(arg190_1, [1, 0]);  arg190_1 = None
        return (reshape_default_1, permute_default)



def make_inputs():
    return [
    torch.randn([32768, 384], dtype=torch.float32, device='cuda'),
    torch.randn([192, 384], dtype=torch.float32, device='cuda'),
    [512, 64, 384],  # _shape_param_0
    [32768, 384],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
