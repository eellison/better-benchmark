"""
Standalone repro captured via capture_hook.
Label: phi_2
Pattern hash: 38d5f8e025b9
Shape hash: 5cb44bad
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convert_element_type_624: "f32[4, 512, 2560]", getitem_342: "f32[4, 512, 1]", getitem_341: "f32[4, 512, 1]", arg438_1: "f16[2560]", arg439_1: "f16[2560]", _shape_param_0, arg440_1: "f16[2560, 2560]", _shape_param_1, arg442_1: "f16[2560, 2560]", _shape_param_2, arg444_1: "f16[2560, 2560]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:291 in forward, code: hidden_states = self.input_layernorm(hidden_states)
        sub_tensor: "f32[4, 512, 2560]" = torch.ops.aten.sub.Tensor(convert_element_type_624, getitem_342);  convert_element_type_624 = getitem_342 = None
        add_tensor: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_341, 1e-05);  getitem_341 = None
        rsqrt_default: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[4, 512, 2560]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[4, 512, 2560]" = torch.ops.aten.mul.Tensor(mul_tensor, arg438_1);  mul_tensor = arg438_1 = None
        add_tensor_1: "f32[4, 512, 2560]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg439_1);  mul_tensor_1 = arg439_1 = None
        convert_element_type_default: "f16[4, 512, 2560]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float16);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:208 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default: "f16[2048, 2560]" = torch.ops.aten.reshape.default(convert_element_type_default, _shape_param_0);  _shape_param_0 = None
        permute_default: "f16[2560, 2560]" = torch.ops.aten.permute.default(arg440_1, [1, 0]);  arg440_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:209 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_1: "f16[2048, 2560]" = torch.ops.aten.reshape.default(convert_element_type_default, _shape_param_1);  _shape_param_1 = None
        permute_default_1: "f16[2560, 2560]" = torch.ops.aten.permute.default(arg442_1, [1, 0]);  arg442_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:210 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_2: "f16[2048, 2560]" = torch.ops.aten.reshape.default(convert_element_type_default, _shape_param_2);  convert_element_type_default = _shape_param_2 = None
        permute_default_2: "f16[2560, 2560]" = torch.ops.aten.permute.default(arg444_1, [1, 0]);  arg444_1 = None
        return (reshape_default, permute_default, reshape_default_1, permute_default_1, reshape_default_2, permute_default_2)



def make_inputs():
    return [
    torch.randn([4, 512, 2560], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([2560], dtype=torch.float16, device='cuda'),
    torch.randn([2560], dtype=torch.float16, device='cuda'),
    [2048, 2560],  # _shape_param_0
    torch.randn([2560, 2560], dtype=torch.float16, device='cuda'),
    [2048, 2560],  # _shape_param_1
    torch.randn([2560, 2560], dtype=torch.float16, device='cuda'),
    [2048, 2560],  # _shape_param_2
    torch.randn([2560, 2560], dtype=torch.float16, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
