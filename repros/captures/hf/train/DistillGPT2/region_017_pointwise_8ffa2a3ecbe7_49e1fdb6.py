"""
Standalone repro captured via capture_hook.
Label: hf_DistillGPT2_train
Pattern hash: 8ffa2a3ecbe7
Shape hash: 49e1fdb6
"""
_shapes_config = "(T([32, 512, 50257], f32), T([50257, 768], f32), S([16384, 50257]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, tangents_1: "f32[32, 512, 50257]", primals_2: "f32[50257, 768]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:706 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        reshape_default: "f32[16384, 50257]" = torch.ops.aten.reshape.default(tangents_1, _shape_param_0);  tangents_1 = _shape_param_0 = None
        permute_default: "f32[768, 50257]" = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
        permute_default_1: "f32[50257, 768]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        constant_pad_nd_default: "f32[16384, 50260]" = torch.ops.aten.constant_pad_nd.default(reshape_default, [0, 3, 0, 0]);  reshape_default = None
        constant_pad_nd_default_1: "f32[50260, 768]" = torch.ops.aten.constant_pad_nd.default(permute_default_1, [0, 0, 0, 3]);  permute_default_1 = None
        return (constant_pad_nd_default, constant_pad_nd_default_1)



def make_inputs():
    return [
    torch.randn([32, 512, 50257], dtype=torch.float32, device='cuda'),
    torch.randn([50257, 768], dtype=torch.float32, device='cuda'),
    [16384, 50257],  # _shape_param_0
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
