"""
Standalone repro captured via capture_hook.
Label: hf_GoogleFnet_train
Pattern hash: 7978e27cdc69
Shape hash: 3e091013
"""
_shapes_config = "(T([32, 512, 768, 2], f32), T([32, 512, 768], f32), T([768], f32), T([768], f32), T([3072, 768], f32), S([16384, 768]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, view_as_real_11: "f32[32, 512, 768, 2]", add_91: "f32[32, 512, 768]", primals_99: "f32[768]", primals_100: "f32[768]", primals_101: "f32[3072, 768]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:176 in forward, code: outputs = self.fourier_transform(hidden_states).real
        select_int: "f32[32, 512, 768]" = torch.ops.aten.select.int(view_as_real_11, 3, 0);  view_as_real_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        add_tensor: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_91, select_int);  add_91 = select_int = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[32, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        mul_tensor: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, primals_99);  mul_tensor = primals_99 = None
        add_tensor_2: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, primals_100);  mul_tensor_1 = primals_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_0);  add_tensor_2 = _shape_param_0 = None
        permute_default: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_101, [1, 0]);  primals_101 = None
        return (reshape_default, permute_default)



def make_inputs():
    return [
    torch.randn([32, 512, 768, 2], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    [16384, 768],  # _shape_param_0
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
