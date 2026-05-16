"""
Standalone repro captured via capture_hook.
Label: qwen2_0.5b
Pattern hash: b430273ac746
Shape hash: 6be633b6
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_93: "f16[2048, 4864]", _shape_param_0, mm_94: "f16[2048, 4864]", _shape_param_1, _shape_param_2, arg291_1: "f16[896, 4864]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        reshape_default: "f16[4, 512, 4864]" = torch.ops.aten.reshape.default(mm_93, _shape_param_0);  mm_93 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_default: "f32[4, 512, 4864]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float32);  reshape_default = None
        neg_default: "f32[4, 512, 4864]" = torch.ops.aten.neg.default(convert_element_type_default)
        exp_default: "f32[4, 512, 4864]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[4, 512, 4864]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[4, 512, 4864]" = torch.ops.aten.div.Tensor(convert_element_type_default, add_tensor);  convert_element_type_default = add_tensor = None
        convert_element_type_default_1: "f16[4, 512, 4864]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.float16);  div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        reshape_default_1: "f16[4, 512, 4864]" = torch.ops.aten.reshape.default(mm_94, _shape_param_1);  mm_94 = _shape_param_1 = None
        mul_tensor: "f16[4, 512, 4864]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, reshape_default_1);  convert_element_type_default_1 = reshape_default_1 = None
        reshape_default_2: "f16[2048, 4864]" = torch.ops.aten.reshape.default(mul_tensor, _shape_param_2);  mul_tensor = _shape_param_2 = None
        permute_default: "f16[4864, 896]" = torch.ops.aten.permute.default(arg291_1, [1, 0]);  arg291_1 = None
        return (reshape_default_2, permute_default)



def make_inputs():
    return [
    torch.randn([2048, 4864], dtype=torch.float16, device='cuda'),
    [4, 512, 4864],  # _shape_param_0
    torch.randn([2048, 4864], dtype=torch.float16, device='cuda'),
    [4, 512, 4864],  # _shape_param_1
    [2048, 4864],  # _shape_param_2
    torch.randn([896, 4864], dtype=torch.float16, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
