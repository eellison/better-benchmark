class GraphModule(torch.nn.Module):
    def forward(self, primals_2: "f32[128, 3, 224, 224]"):
        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_default: "f32[128, 3, 225, 225]" = torch.ops.aten.constant_pad_nd.default(primals_2, [0, 1, 0, 1], 0.0);  primals_2 = None
        return constant_pad_nd_default
