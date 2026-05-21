"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_infer
Pattern hash: 00de3c386307
Shape hash: 91141840
"""
_shapes_config = "(T([32768, 128], f32), T([128], f32), T([128], f32), T([128, 128], f32), T([128, 128], f32), S([256, 128, 128]), S([32768, 128]), S([32768, 128]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_347: "f32[32768, 128]", arg1073_1: "f32[128]", arg1074_1: "f32[128]", arg1075_1: "f32[128, 128]", arg1077_1: "f32[128, 128]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        reshape_default: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_347, _shape_param_0);  addmm_347 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_tensor: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(reshape_default, arg1073_1);  reshape_default = arg1073_1 = None
        add_tensor: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_tensor, arg1074_1);  mul_tensor = arg1074_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        reshape_default_1: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_1);  _shape_param_1 = None
        permute_default: "f32[128, 128]" = torch.ops.aten.permute.default(arg1075_1, [1, 0]);  arg1075_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        reshape_default_2: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_2);  add_tensor = _shape_param_2 = None
        permute_default_1: "f32[128, 128]" = torch.ops.aten.permute.default(arg1077_1, [1, 0]);  arg1077_1 = None
        return (reshape_default_1, permute_default, reshape_default_2, permute_default_1)



def make_inputs():
    return [
    torch.randn([32768, 128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([128, 128], dtype=torch.float32, device='cuda'),
    [256, 128, 128],  # _shape_param_0
    [32768, 128],  # _shape_param_1
    [32768, 128],  # _shape_param_2
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
