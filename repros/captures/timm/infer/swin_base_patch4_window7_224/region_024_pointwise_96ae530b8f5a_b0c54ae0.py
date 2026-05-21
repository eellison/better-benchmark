"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_infer
Pattern hash: 96ae530b8f5a
Shape hash: b0c54ae0
"""
_shapes_config = "(T([100352, 768], f32), S([2048, 49, 768]), S([2048, 49, 3, 8, -1]), S([2048, 8, 49, 32]), S([16384, 49, 32]), S([2048, 8, 32, 49]), S([16384, 32, 49]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_12: "f32[100352, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        reshape_default: "f32[2048, 49, 768]" = torch.ops.aten.reshape.default(addmm_12, _shape_param_0);  addmm_12 = _shape_param_0 = None
        reshape_default_1: "f32[2048, 49, 3, 8, 32]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f32[3, 2048, 8, 49, 32]" = torch.ops.aten.permute.default(reshape_default_1, [2, 0, 3, 1, 4]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_int = torch.ops.aten.unbind.int(permute_default);  permute_default = None
        getitem: "f32[2048, 8, 49, 32]" = unbind_int[0]
        getitem_1: "f32[2048, 8, 49, 32]" = unbind_int[1]
        getitem_2: "f32[2048, 8, 49, 32]" = unbind_int[2];  unbind_int = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_tensor: "f32[2048, 8, 49, 32]" = torch.ops.aten.mul.Tensor(getitem, 0.1767766952966369);  getitem = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_default: "f32[2048, 8, 49, 32]" = torch.ops.aten.expand.default(mul_tensor, _shape_param_2);  mul_tensor = _shape_param_2 = None
        clone_default: "f32[2048, 8, 49, 32]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        reshape_default_2: "f32[16384, 49, 32]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None
        permute_default_1: "f32[2048, 8, 32, 49]" = torch.ops.aten.permute.default(getitem_1, [0, 1, 3, 2]);  getitem_1 = None
        expand_default_1: "f32[2048, 8, 32, 49]" = torch.ops.aten.expand.default(permute_default_1, _shape_param_4);  permute_default_1 = _shape_param_4 = None
        clone_default_1: "f32[2048, 8, 32, 49]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_3: "f32[16384, 32, 49]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_5);  clone_default_1 = _shape_param_5 = None
        return (reshape_default_2, reshape_default_3, getitem_2)



def make_inputs():
    return [
    torch.randn([100352, 768], dtype=torch.float32, device='cuda'),
    [2048, 49, 768],  # _shape_param_0
    [2048, 49, 3, 8, -1],  # _shape_param_1
    [2048, 8, 49, 32],  # _shape_param_2
    [16384, 49, 32],  # _shape_param_3
    [2048, 8, 32, 49],  # _shape_param_4
    [16384, 32, 49],  # _shape_param_5
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
