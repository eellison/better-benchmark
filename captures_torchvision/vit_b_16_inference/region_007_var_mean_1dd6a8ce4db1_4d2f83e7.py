"""
Standalone repro captured via capture_hook.
Label: vit_b_16_inference
Pattern hash: 1dd6a8ce4db1
Shape hash: 4d2f83e7
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, arg3_1: "f32[1, 1, 768]", convolution: "f32[1, 768, 14, 14]", _shape_param_0, arg4_1: "f32[1, 197, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:295 in forward, code: batch_class_token = self.class_token.expand(n, -1, -1)
        expand_default: "f32[1, 1, 768]" = torch.ops.aten.expand.default(arg3_1, [1, -1, -1]);  arg3_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:279 in _process_input, code: x = x.reshape(n, self.hidden_dim, n_h * n_w)
        reshape_default: "f32[1, 768, 196]" = torch.ops.aten.reshape.default(convolution, _shape_param_0);  convolution = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:285 in _process_input, code: x = x.permute(0, 2, 1)
        permute_default: "f32[1, 196, 768]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:296 in forward, code: x = torch.cat([batch_class_token, x], dim=1)
        cat_default: "f32[1, 197, 768]" = torch.ops.aten.cat.default([expand_default, permute_default], 1);  expand_default = permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:156 in forward, code: input = input + self.pos_embedding
        add_tensor: "f32[1, 197, 768]" = torch.ops.aten.add.Tensor(cat_default, arg4_1);  cat_default = arg4_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:112 in forward, code: x = self.ln_1(input)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True);  add_tensor = None
        getitem: "f32[1, 197, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 197, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_1)



def make_inputs():
    return [
    torch.randn([1, 1, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 14, 14], dtype=torch.float32, device='cuda'),
    [1, 768, 196],  # _shape_param_0
    torch.randn([1, 197, 768], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
