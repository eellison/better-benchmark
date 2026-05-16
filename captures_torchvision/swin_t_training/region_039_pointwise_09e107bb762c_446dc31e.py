"""
Standalone repro captured via capture_hook.
Label: swin_t_training
Pattern hash: 09e107bb762c
Shape hash: 446dc31e
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_12: "f32[4, 28, 28, 1]", view_49: "f32[4, 28, 28, 192]", getitem_13: "f32[4, 28, 28, 1]", primals_37: "f32[192]", primals_38: "f32[192]", _shape_param_0, _shape_param_1, _shape_param_2, primals_41: "f32[576, 192]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:453 in forward, code: x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        add_tensor: "f32[4, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
        rsqrt_default: "f32[4, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[4, 28, 28, 192]" = torch.ops.aten.sub.Tensor(view_49, getitem_13);  view_49 = getitem_13 = None
        mul_tensor: "f32[4, 28, 28, 192]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[4, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_tensor, primals_37);  mul_tensor = primals_37 = None
        add_tensor_1: "f32[4, 28, 28, 192]" = torch.ops.aten.add.Tensor(mul_tensor_1, primals_38);  mul_tensor_1 = primals_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:171 in shifted_window_attention, code: x = x.view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
        reshape_default: "f32[4, 4, 7, 4, 7, 192]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_0);  add_tensor_1 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:172 in shifted_window_attention, code: x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size[0] * window_size[1], C)  # B*nW, Ws*Ws, C
        permute_default: "f32[4, 4, 4, 7, 7, 192]" = torch.ops.aten.permute.default(reshape_default, [0, 1, 3, 2, 4, 5]);  reshape_default = None
        clone_default: "f32[4, 4, 4, 7, 7, 192]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_1: "f32[64, 49, 192]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:179 in shifted_window_attention, code: qkv = F.linear(x, qkv_weight, qkv_bias)
        reshape_default_2: "f32[3136, 192]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None
        permute_default_1: "f32[192, 576]" = torch.ops.aten.permute.default(primals_41, [1, 0]);  primals_41 = None
        return (reshape_default_2, permute_default_1)



def make_inputs():
    return [
    torch.randn([4, 28, 28, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 28, 28, 192], dtype=torch.float32, device='cuda'),
    torch.randn([4, 28, 28, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    [4, 4, 7, 4, 7, 192],  # _shape_param_0
    [64, 49, 192],  # _shape_param_1
    [3136, 192],  # _shape_param_2
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
