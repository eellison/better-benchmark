"""
Standalone repro captured via capture_hook.
Label: hf_M2M100ForConditionalGeneration_train
Pattern hash: 49245221411e
Shape hash: 2c10d469
"""
_shapes_config = "(T([64, 128, 1024], b8), T([64, 128, 1024], f32), T([1024, 4096], f32), S([8192, 1024]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, gt_3: "b8[64, 128, 1024]", tangents_1: "f32[64, 128, 1024]", primals_29: "f32[1024, 4096]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:474 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        convert_element_type_default: "f32[64, 128, 1024]" = torch.ops.prims.convert_element_type.default(gt_3, torch.float32);  gt_3 = None
        mul_tensor: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(tangents_1, mul_tensor);  tangents_1 = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:473 in forward, code: hidden_states = self.fc2(hidden_states)
        reshape_default: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_0);  mul_tensor_1 = _shape_param_0 = None
        permute_default: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_29, [1, 0]);  primals_29 = None
        permute_default_1: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default, permute_default_1)



def make_inputs():
    return [
    torch.randint(0, 2, [64, 128, 1024], dtype=torch.bool, device='cuda'),
    torch.randn([64, 128, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 4096], dtype=torch.float32, device='cuda'),
    [8192, 1024],  # _shape_param_0
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
