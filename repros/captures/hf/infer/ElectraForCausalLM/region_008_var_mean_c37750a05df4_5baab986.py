"""
Standalone repro captured via capture_hook.
Label: hf_ElectraForCausalLM_infer
Pattern hash: c37750a05df4
Shape hash: 5baab986
"""
_shapes_config = "(T([32768, 256], f32), T([64, 512, 256], f32), T([256], f32), T([256], f32), T([256, 256], f32), T([256, 256], f32), S([64, 512, 256]), S([32768, 256]), S([32768, 256]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_66: "f32[32768, 256]", add_89: "f32[64, 512, 256]", arg185_1: "f32[256]", arg186_1: "f32[256]", arg187_1: "f32[256, 256]", arg189_1: "f32[256, 256]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_66, _shape_param_0);  addmm_66 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_tensor: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(reshape_default, add_89);  reshape_default = add_89 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[64, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[64, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        add_tensor_1: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_tensor, arg185_1);  mul_tensor = arg185_1 = None
        add_tensor_2: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg186_1);  mul_tensor_1 = arg186_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default_1: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_1);  _shape_param_1 = None
        permute_default: "f32[256, 256]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default_2: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_2);  add_tensor_2 = _shape_param_2 = None
        permute_default_1: "f32[256, 256]" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
        return (reshape_default_1, permute_default, reshape_default_2, permute_default_1)



def make_inputs():
    return [
    torch.randn([32768, 256], dtype=torch.float32, device='cuda'),
    torch.randn([64, 512, 256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256, 256], dtype=torch.float32, device='cuda'),
    torch.randn([256, 256], dtype=torch.float32, device='cuda'),
    [64, 512, 256],  # _shape_param_0
    [32768, 256],  # _shape_param_1
    [32768, 256],  # _shape_param_2
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
