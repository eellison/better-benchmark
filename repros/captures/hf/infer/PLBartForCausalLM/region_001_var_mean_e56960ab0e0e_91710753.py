"""
Standalone repro captured via capture_hook.
Label: hf_PLBartForCausalLM_infer
Pattern hash: e56960ab0e0e
Shape hash: 91710753
"""
_shapes_config = "(T([16384, 768], f32), T([16, 1024, 768], f32), T([768], f32), T([768], f32), T([50005, 768], f32), S([16, 1024, 768]), S([16384, 768]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_35: "f32[16384, 768]", add_46: "f32[16, 1024, 768]", arg99_1: "f32[768]", arg100_1: "f32[768]", arg1_1: "f32[50005, 768]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:476 in forward, code: hidden_states = self.fc2(hidden_states)
        reshape_default: "f32[16, 1024, 768]" = torch.ops.aten.reshape.default(addmm_35, _shape_param_0);  addmm_35 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:478 in forward, code: hidden_states = residual + hidden_states
        add_tensor: "f32[16, 1024, 768]" = torch.ops.aten.add.Tensor(add_46, reshape_default);  add_46 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:479 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[16, 1024, 1]" = var_mean_correction[0]
        getitem_1: "f32[16, 1024, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[16, 1024, 768]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        add_tensor_1: "f32[16, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[16, 1024, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[16, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[16, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg99_1);  mul_tensor = arg99_1 = None
        add_tensor_2: "f32[16, 1024, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg100_1);  mul_tensor_1 = arg100_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:1125 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        reshape_default_1: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_1);  add_tensor_2 = _shape_param_1 = None
        permute_default: "f32[768, 50005]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        constant_pad_nd_default: "f32[768, 50008]" = torch.ops.aten.constant_pad_nd.default(permute_default, [0, 3, 0, 0]);  permute_default = None
        return (reshape_default_1, constant_pad_nd_default)



def make_inputs():
    return [
    torch.randn([16384, 768], dtype=torch.float32, device='cuda'),
    torch.randn([16, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([50005, 768], dtype=torch.float32, device='cuda'),
    [16, 1024, 768],  # _shape_param_0
    [16384, 768],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
