"""
Standalone repro captured via capture_hook.
Label: timm_mobilevit_s_train
Pattern hash: eecc7f6f5a71
Shape hash: 83d1fcdd
"""
_shapes_config = "(T([8192, 480], f32), T([240, 480], f32), S([512, 16, 480]), S([8192, 480]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_34: "f32[8192, 480]", primals_289: "f32[240, 480]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        reshape_default: "f32[512, 16, 480]" = torch.ops.aten.reshape.default(addmm_34, _shape_param_0);  addmm_34 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        neg_default: "f32[512, 16, 480]" = torch.ops.aten.neg.default(reshape_default)
        exp_default: "f32[512, 16, 480]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[512, 16, 480]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[512, 16, 480]" = torch.ops.aten.div.Tensor(reshape_default, add_tensor);  reshape_default = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default_1: "f32[8192, 480]" = torch.ops.aten.reshape.default(div_tensor, _shape_param_1);  div_tensor = _shape_param_1 = None
        permute_default: "f32[480, 240]" = torch.ops.aten.permute.default(primals_289, [1, 0]);  primals_289 = None
        return (reshape_default_1, permute_default)



def make_inputs():
    return [
    torch.randn([8192, 480], dtype=torch.float32, device='cuda'),
    torch.randn([240, 480], dtype=torch.float32, device='cuda'),
    [512, 16, 480],  # _shape_param_0
    [8192, 480],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
