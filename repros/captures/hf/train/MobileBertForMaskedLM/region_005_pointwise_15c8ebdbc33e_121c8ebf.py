"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_train
Pattern hash: 15c8ebdbc33e
Shape hash: 121c8ebf
"""
_shapes_config = "(T([32768, 128], f32), T([128], f32), T([128], f32), T([32768, 128], f32), T([128], f32), T([128], f32), T([512, 128], f32), S([256, 128, 128]), S([256, 128, 128]), S([32768, 128]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_346: "f32[32768, 128]", primals_1070: "f32[128]", primals_1071: "f32[128]", addmm_351: "f32[32768, 128]", primals_1084: "f32[128]", primals_1085: "f32[128]", primals_1086: "f32[512, 128]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        reshape_default: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_346, _shape_param_0);  addmm_346 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_tensor: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(reshape_default, primals_1070);  reshape_default = primals_1070 = None
        add_tensor: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_tensor, primals_1071);  mul_tensor = primals_1071 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        reshape_default_1: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_351, _shape_param_1);  addmm_351 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_tensor_1: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(reshape_default_1, add_tensor);  reshape_default_1 = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_tensor_1: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_tensor_1, primals_1084);  add_tensor_1 = primals_1084 = None
        add_tensor_2: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_tensor_1, primals_1085);  mul_tensor_1 = primals_1085 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default_2: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_2);  add_tensor_2 = _shape_param_2 = None
        permute_default: "f32[128, 512]" = torch.ops.aten.permute.default(primals_1086, [1, 0]);  primals_1086 = None
        return (reshape_default_2, permute_default)



def make_inputs():
    return [
    torch.randn([32768, 128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([32768, 128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([512, 128], dtype=torch.float32, device='cuda'),
    [256, 128, 128],  # _shape_param_0
    [256, 128, 128],  # _shape_param_1
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
