"""
Standalone repro captured via capture_hook.
Label: hf_GoogleFnet_train
Pattern hash: b68d7ef0788f
Shape hash: f1018ca3
"""
_shapes_config = "(T([16384, 32000], f32), T([32, 512, 768, 2], f32), T([32, 512, 768], f32), T([32, 512, 768], b8), T([768, 768], f32), S([16384, 768]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, view_57: "f32[16384, 32000]", view_as_real_23: "f32[32, 512, 768, 2]", mul_446: "f32[32, 512, 768]", gt: "b8[32, 512, 768]", primals_9: "f32[768, 768]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:331 in forward, code: hidden_states = self.decoder(hidden_states)
        permute_default: "f32[32000, 16384]" = torch.ops.aten.permute.default(view_57, [1, 0]);  view_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:176 in forward, code: outputs = self.fourier_transform(hidden_states).real
        select_int: "f32[32, 512, 768]" = torch.ops.aten.select.int(view_as_real_23, 3, 0);  view_as_real_23 = None
        add_tensor: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_446, select_int);  mul_446 = select_int = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:138 in forward, code: embeddings = self.dropout(embeddings)
        convert_element_type_default: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_tensor: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.0);  convert_element_type_default = None
        mul_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_tensor, mul_tensor);  add_tensor = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:137 in forward, code: embeddings = self.projection(embeddings)
        reshape_default: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_0);  mul_tensor_1 = _shape_param_0 = None
        permute_default_1: "f32[768, 768]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
        permute_default_2: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_1, [1, 0]);  permute_default_1 = None
        return (permute_default, reshape_default, permute_default_2)



def make_inputs():
    return [
    torch.randn([16384, 32000], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 768, 2], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [32, 512, 768], dtype=torch.bool, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
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
