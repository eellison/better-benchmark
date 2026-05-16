"""
Standalone repro captured via capture_hook.
Label: phi_2
Pattern hash: 65633e5584ac
Shape hash: 1c6cff02
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_189: "f16[2048, 2560]", _shape_param_0, addmm_191: "f16[2048, 2560]", _shape_param_1, add_250: "f16[4, 512, 2560]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:252 in forward, code: attn_output = self.dense(attn_output)
        reshape_default: "f16[4, 512, 2560]" = torch.ops.aten.reshape.default(addmm_189, _shape_param_0);  addmm_189 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:267 in forward, code: hidden_states = self.fc2(hidden_states)
        reshape_default_1: "f16[4, 512, 2560]" = torch.ops.aten.reshape.default(addmm_191, _shape_param_1);  addmm_191 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:305 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_tensor: "f16[4, 512, 2560]" = torch.ops.aten.add.Tensor(reshape_default, reshape_default_1);  reshape_default = reshape_default_1 = None
        add_tensor_1: "f16[4, 512, 2560]" = torch.ops.aten.add.Tensor(add_tensor, add_250);  add_tensor = add_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:398 in forward, code: hidden_states = self.final_layernorm(hidden_states)
        convert_element_type_default: "f32[4, 512, 2560]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float32);  add_tensor_1 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True);  convert_element_type_default = None
        getitem: "f32[4, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[4, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_1)



def make_inputs():
    return [
    torch.randn([2048, 2560], dtype=torch.float16, device='cuda'),
    [4, 512, 2560],  # _shape_param_0
    torch.randn([2048, 2560], dtype=torch.float16, device='cuda'),
    [4, 512, 2560],  # _shape_param_1
    torch.randn([4, 512, 2560], dtype=torch.float16, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
