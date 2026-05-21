"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_train
Pattern hash: cdcf7b9e5bd1
Shape hash: e788d6a9
"""
_shapes_config = "(T([100352, 512], f32), T([512], f32), T([128, 28, 28, 512], f32), T([128, 28, 28, 1], f32), T([128, 1, 1], b8), T([128, 512], f32), S([128, 28, 28, 512]), S([128, 28, 28, 2, 2, 128]), S([128, 56, 56, 128]), S([128, 3136, 128]), S([401408, 128]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_186: "f32[100352, 512]", primals_35: "f32[512]", mul_20: "f32[128, 28, 28, 512]", div_118: "f32[128, 28, 28, 1]", lt_1: "b8[128, 1, 1]", primals_33: "f32[128, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:541 in forward, code: x = self.reduction(x)
        reshape_default: "f32[128, 28, 28, 512]" = torch.ops.aten.reshape.default(mm_186, _shape_param_0);  mm_186 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:540 in forward, code: x = self.norm(x)
        mul_tensor: "f32[128, 28, 28, 512]" = torch.ops.aten.mul.Tensor(reshape_default, primals_35);  reshape_default = primals_35 = None
        mul_tensor_1: "f32[128, 28, 28, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, 512)
        sum_dim_int_list: "f32[128, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [3], True)
        mul_tensor_2: "f32[128, 28, 28, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_20);  mul_tensor = None
        sum_dim_int_list_1: "f32[128, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [3], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[128, 28, 28, 512]" = torch.ops.aten.mul.Tensor(mul_20, sum_dim_int_list_1);  mul_20 = sum_dim_int_list_1 = None
        sub_tensor: "f32[128, 28, 28, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[128, 28, 28, 512]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[128, 28, 28, 512]" = torch.ops.aten.mul.Tensor(div_118, sub_tensor_1);  div_118 = sub_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:539 in forward, code: x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        reshape_default_1: "f32[128, 28, 28, 2, 2, 128]" = torch.ops.aten.reshape.default(mul_tensor_4, _shape_param_1);  mul_tensor_4 = _shape_param_1 = None
        permute_default: "f32[128, 28, 2, 28, 2, 128]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 4, 2, 3, 5]);  reshape_default_1 = None
        clone_default: "f32[128, 28, 2, 28, 2, 128]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_2: "f32[128, 56, 56, 128]" = torch.ops.aten.reshape.default(clone_default, _shape_param_2);  clone_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        reshape_default_3: "f32[128, 3136, 128]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_default: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_1, torch.float32);  lt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_tensor: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_default, 0.9956521736457944);  convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_tensor_5: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(reshape_default_3, div_tensor);  reshape_default_3 = div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default_4: "f32[401408, 128]" = torch.ops.aten.reshape.default(mul_tensor_5, _shape_param_4);  mul_tensor_5 = _shape_param_4 = None
        permute_default_1: "f32[512, 128]" = torch.ops.aten.permute.default(primals_33, [1, 0]);  primals_33 = None
        permute_default_2: "f32[128, 512]" = torch.ops.aten.permute.default(permute_default_1, [1, 0]);  permute_default_1 = None
        return (reshape_default_4, permute_default_2)



def make_inputs():
    return [
    torch.randn([100352, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([128, 28, 28, 512], dtype=torch.float32, device='cuda'),
    torch.randn([128, 28, 28, 1], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [128, 1, 1], dtype=torch.bool, device='cuda'),
    torch.randn([128, 512], dtype=torch.float32, device='cuda'),
    [128, 28, 28, 512],  # _shape_param_0
    [128, 28, 28, 2, 2, 128],  # _shape_param_1
    [128, 56, 56, 128],  # _shape_param_2
    [128, 3136, 128],  # _shape_param_3
    [401408, 128],  # _shape_param_4
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
