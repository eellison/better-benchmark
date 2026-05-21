"""
Standalone repro captured via capture_hook.
Label: hf_BlenderbotForCausalLM_train
Pattern hash: 42a431323819
Shape hash: 858b1c30
"""
_shapes_config = "(T([32, 128, 2560], f32), T([2560], f32), T([2560], f32), T([2560, 2560], f32), T([2560, 2560], f32), T([2560, 2560], f32), S([4096, 2560]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, primals_3: "f32[32, 128, 2560]", primals_1: "f32[2560]", primals_2: "f32[2560]", primals_4: "f32[2560, 2560]", primals_6: "f32[2560, 2560]", primals_8: "f32[2560, 2560]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_correction = torch.ops.aten.var_mean.correction(primals_3, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 128, 1]" = var_mean_correction[0]
        getitem_1: "f32[32, 128, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[32, 128, 2560]" = torch.ops.aten.sub.Tensor(primals_3, getitem_1);  primals_3 = getitem_1 = None
        mul_tensor: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_tensor, primals_1);  mul_tensor = primals_1 = None
        add_tensor_1: "f32[32, 128, 2560]" = torch.ops.aten.add.Tensor(mul_tensor_1, primals_2);  mul_tensor_1 = primals_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default: "f32[4096, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_0);  add_tensor_1 = _shape_param_0 = None
        permute_default: "f32[2560, 2560]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        permute_default_1: "f32[2560, 2560]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        permute_default_2: "f32[2560, 2560]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        return (reshape_default, permute_default, permute_default_1, permute_default_2)



def make_inputs():
    return [
    torch.randn([32, 128, 2560], dtype=torch.float32, device='cuda'),
    torch.randn([2560], dtype=torch.float32, device='cuda'),
    torch.randn([2560], dtype=torch.float32, device='cuda'),
    torch.randn([2560, 2560], dtype=torch.float32, device='cuda'),
    torch.randn([2560, 2560], dtype=torch.float32, device='cuda'),
    torch.randn([2560, 2560], dtype=torch.float32, device='cuda'),
    [4096, 2560],  # _shape_param_0
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
