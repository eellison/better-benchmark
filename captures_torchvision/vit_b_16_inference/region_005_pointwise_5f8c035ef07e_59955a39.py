"""
Standalone repro captured via capture_hook.
Label: vit_b_16_inference
Pattern hash: 5f8c035ef07e
Shape hash: 59955a39
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_44: "f32[197, 2304]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:113 in forward, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        reshape_default: "f32[197, 1, 2304]" = torch.ops.aten.reshape.default(addmm_44, _shape_param_0);  addmm_44 = _shape_param_0 = None
        reshape_default_1: "f32[197, 1, 3, 768]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        unsqueeze_default: "f32[1, 197, 1, 3, 768]" = torch.ops.aten.unsqueeze.default(reshape_default_1, 0);  reshape_default_1 = None
        permute_default: "f32[3, 197, 1, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_default, [3, 1, 2, 0, 4]);  unsqueeze_default = None
        squeeze_dim: "f32[3, 197, 1, 768]" = torch.ops.aten.squeeze.dim(permute_default, -2);  permute_default = None
        clone_default: "f32[3, 197, 1, 768]" = torch.ops.aten.clone.default(squeeze_dim, memory_format = torch.contiguous_format);  squeeze_dim = None
        select_int: "f32[197, 1, 768]" = torch.ops.aten.select.int(clone_default, 0, 0)
        reshape_default_2: "f32[197, 12, 64]" = torch.ops.aten.reshape.default(select_int, _shape_param_2);  select_int = _shape_param_2 = None
        permute_default_1: "f32[12, 197, 64]" = torch.ops.aten.permute.default(reshape_default_2, [1, 0, 2]);  reshape_default_2 = None
        reshape_default_3: "f32[1, 12, 197, 64]" = torch.ops.aten.reshape.default(permute_default_1, _shape_param_3);  permute_default_1 = _shape_param_3 = None
        select_int_1: "f32[197, 1, 768]" = torch.ops.aten.select.int(clone_default, 0, 1)
        reshape_default_4: "f32[197, 12, 64]" = torch.ops.aten.reshape.default(select_int_1, _shape_param_4);  select_int_1 = _shape_param_4 = None
        permute_default_2: "f32[12, 197, 64]" = torch.ops.aten.permute.default(reshape_default_4, [1, 0, 2]);  reshape_default_4 = None
        reshape_default_5: "f32[1, 12, 197, 64]" = torch.ops.aten.reshape.default(permute_default_2, _shape_param_5);  permute_default_2 = _shape_param_5 = None
        select_int_2: "f32[197, 1, 768]" = torch.ops.aten.select.int(clone_default, 0, 2);  clone_default = None
        reshape_default_6: "f32[197, 12, 64]" = torch.ops.aten.reshape.default(select_int_2, _shape_param_6);  select_int_2 = _shape_param_6 = None
        permute_default_3: "f32[12, 197, 64]" = torch.ops.aten.permute.default(reshape_default_6, [1, 0, 2]);  reshape_default_6 = None
        reshape_default_7: "f32[1, 12, 197, 64]" = torch.ops.aten.reshape.default(permute_default_3, _shape_param_7);  permute_default_3 = _shape_param_7 = None
        return (reshape_default_3, reshape_default_5, reshape_default_7)



def make_inputs():
    return [
    torch.randn([197, 2304], dtype=torch.float32, device='cuda'),
    [197, 1, 2304],  # _shape_param_0
    [197, 1, 3, 768],  # _shape_param_1
    [197, 12, 64],  # _shape_param_2
    [1, 12, 197, 64],  # _shape_param_3
    [197, 12, 64],  # _shape_param_4
    [1, 12, 197, 64],  # _shape_param_5
    [197, 12, 64],  # _shape_param_6
    [1, 12, 197, 64],  # _shape_param_7
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
