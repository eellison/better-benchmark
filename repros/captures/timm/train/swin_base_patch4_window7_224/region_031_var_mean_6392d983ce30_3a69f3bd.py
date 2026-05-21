"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_train
Pattern hash: 6392d983ce30
Shape hash: 3a69f3bd
"""
_shapes_config = "(T([401408, 128], f32), T([46], i64), T([128, 3136, 128], f32), T([512], f32), T([512], f32), T([256, 512], f32), S([128, 3136, 128]), S([128, 56, 56, 128]), S([128, 28, 2, 28, 2, 128]), S([128, 28, 28, 512]), S([100352, 512]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_7: "f32[401408, 128]", inductor_seeds_default: "i64[46]", view_48: "f32[128, 3136, 128]", primals_35: "f32[512]", primals_36: "f32[512]", primals_37: "f32[256, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default: "f32[128, 3136, 128]" = torch.ops.aten.reshape.default(addmm_7, _shape_param_0);  addmm_7 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1);  inductor_seeds_default = None
        inductor_random_default: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        lt_scalar: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default, 0.9956521736457944);  inductor_random_default = None
        convert_element_type_default: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_scalar, torch.float32);  lt_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_tensor: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_default, 0.9956521736457944);  convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_tensor: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(reshape_default, div_tensor);  reshape_default = div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_tensor: "f32[128, 3136, 128]" = torch.ops.aten.add.Tensor(view_48, mul_tensor);  view_48 = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        reshape_default_1: "f32[128, 56, 56, 128]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_1);  add_tensor = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:539 in forward, code: x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        reshape_default_2: "f32[128, 28, 2, 28, 2, 128]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None
        permute_default: "f32[128, 28, 28, 2, 2, 128]" = torch.ops.aten.permute.default(reshape_default_2, [0, 1, 3, 4, 2, 5]);  reshape_default_2 = None
        clone_default: "f32[128, 28, 28, 2, 2, 128]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_3: "f32[128, 28, 28, 512]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:540 in forward, code: x = self.norm(x)
        var_mean_correction = torch.ops.aten.var_mean.correction(reshape_default_3, [3], correction = 0, keepdim = True)
        getitem: "f32[128, 28, 28, 1]" = var_mean_correction[0]
        getitem_1: "f32[128, 28, 28, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[128, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[128, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[128, 28, 28, 512]" = torch.ops.aten.sub.Tensor(reshape_default_3, getitem_1);  reshape_default_3 = getitem_1 = None
        mul_tensor_1: "f32[128, 28, 28, 512]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_2: "f32[128, 28, 28, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_1, primals_35);  mul_tensor_1 = primals_35 = None
        add_tensor_2: "f32[128, 28, 28, 512]" = torch.ops.aten.add.Tensor(mul_tensor_2, primals_36);  mul_tensor_2 = primals_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:541 in forward, code: x = self.reduction(x)
        permute_default_1: "f32[512, 256]" = torch.ops.aten.permute.default(primals_37, [1, 0]);  primals_37 = None
        reshape_default_4: "f32[100352, 512]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_4);  add_tensor_2 = _shape_param_4 = None
        return (permute_default_1, reshape_default_4)



def make_inputs():
    return [
    torch.randn([401408, 128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 46, [46], dtype=torch.int64, device='cuda'),
    torch.randn([128, 3136, 128], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([256, 512], dtype=torch.float32, device='cuda'),
    [128, 3136, 128],  # _shape_param_0
    [128, 56, 56, 128],  # _shape_param_1
    [128, 28, 2, 28, 2, 128],  # _shape_param_2
    [128, 28, 28, 512],  # _shape_param_3
    [100352, 512],  # _shape_param_4
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
