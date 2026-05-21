"""
Standalone repro captured via capture_hook.
Label: timm_mobilevit_s_train
Pattern hash: 1c4efa8068a8
Shape hash: 4a3694c8
"""
_shapes_config = "(T([128, 256, 16, 16], f32, stride=(65536, 1, 4096, 256)), T([128, 128, 16, 16], f32, stride=(32768, 1, 2048, 128)), T([128, 128, 16, 16], f32, stride=(32768, 1, 2048, 128)), T([1, 128, 1, 1], f32), T([128], f32), T([128], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_205: "f32[128, 256, 16, 16]", getitem_230: "f32[128, 128, 16, 16]", convolution_22: "f32[128, 128, 16, 16]", unsqueeze_250: "f32[1, 128, 1, 1]", squeeze_64: "f32[128]", primals_159: "f32[128]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:277 in forward, code: x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
        slice_tensor: "f32[128, 128, 16, 16]" = torch.ops.aten.slice.Tensor(getitem_205, 1, 0, 128);  getitem_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        add_tensor: "f32[128, 128, 16, 16]" = torch.ops.aten.add.Tensor(slice_tensor, getitem_230);  slice_tensor = getitem_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_dim_int_list: "f32[128]" = torch.ops.aten.sum.dim_IntList(add_tensor, [0, 2, 3])
        sub_tensor: "f32[128, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_250);  convolution_22 = unsqueeze_250 = None
        mul_tensor: "f32[128, 128, 16, 16]" = torch.ops.aten.mul.Tensor(add_tensor, sub_tensor)
        sum_dim_int_list_1: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[128]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 3.0517578125e-05);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[128]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 3.0517578125e-05);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
        mul_tensor_4: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_159);  squeeze_64 = primals_159 = None
        unsqueeze_default_6: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[128, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[128, 128, 16, 16]" = torch.ops.aten.sub.Tensor(add_tensor, mul_tensor_6);  add_tensor = mul_tensor_6 = None
        sub_tensor_2: "f32[128, 128, 16, 16]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[128, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        return mul_tensor_7



def make_inputs():
    return [
    torch.randn(8388608, dtype=torch.float32, device='cuda').as_strided([128, 256, 16, 16], [65536, 1, 4096, 256]),  # getitem_205
    torch.randn(4194304, dtype=torch.float32, device='cuda').as_strided([128, 128, 16, 16], [32768, 1, 2048, 128]),  # getitem_230
    torch.randn(4194304, dtype=torch.float32, device='cuda').as_strided([128, 128, 16, 16], [32768, 1, 2048, 128]),  # convolution_22
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
