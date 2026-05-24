import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "bf16[6144, 50304]", arg1_1: "f8e4m3fn[50304, 768]", arg2_1: "f8e4m3fn[6144, 768]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:79 in impl, code: x_inv_s = grad.new_tensor(x_s, dtype=torch.float32)
        _tensor_constant0: "f32[]" = self._tensor_constant0;  _tensor_constant0 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:80 in impl, code: w_inv_s = grad.new_tensor(w_s, dtype=torch.float32)
        _tensor_constant1: "f32[]" = self._tensor_constant1;  _tensor_constant1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:81 in impl, code: grad_inv_s = grad.new_tensor(grad_s, dtype=torch.float32)
        _tensor_constant2: "f32[]" = self._tensor_constant2;  _tensor_constant2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:82 in impl, code: grad_f8 = grad.div(grad_s).to(torch.float8_e5m2)
        div: "bf16[6144, 50304]" = torch.ops.aten.div.Tensor(arg0_1, 0.002232142857142857);  arg0_1 = None
        convert_element_type: "f8e5m2[6144, 50304]" = torch.ops.prims.convert_element_type.default(div, torch.float8_e5m2);  div = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:85 in impl, code: w_f8.T.contiguous().T,
        permute: "f8e4m3fn[768, 50304]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        clone: "f8e4m3fn[768, 50304]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        permute_1: "f8e4m3fn[50304, 768]" = torch.ops.aten.permute.default(clone, [1, 0]);  clone = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:81 in impl, code: grad_inv_s = grad.new_tensor(grad_s, dtype=torch.float32)
        full_default_2: "f32[]" = torch.ops.aten.full.default([], 0.0022321429569274187, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:80 in impl, code: w_inv_s = grad.new_tensor(w_s, dtype=torch.float32)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.001953125, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:83 in impl, code: grad_x = torch._scaled_mm(
        _scaled_mm: "bf16[6144, 768]" = torch.ops.aten._scaled_mm.default(convert_element_type, permute_1, full_default_2, full_default_1, None, None, torch.bfloat16);  permute_1 = full_default_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:93 in impl, code: x_f8.T.contiguous(),
        permute_2: "f8e4m3fn[768, 6144]" = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
        clone_1: "f8e4m3fn[768, 6144]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:94 in impl, code: grad_f8.T.contiguous().T,
        permute_3: "f8e5m2[50304, 6144]" = torch.ops.aten.permute.default(convert_element_type, [1, 0]);  convert_element_type = None
        clone_2: "f8e5m2[50304, 6144]" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None
        permute_4: "f8e5m2[6144, 50304]" = torch.ops.aten.permute.default(clone_2, [1, 0]);  clone_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:79 in impl, code: x_inv_s = grad.new_tensor(x_s, dtype=torch.float32)
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.06185895577073097, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:92 in impl, code: grad_w = torch._scaled_mm(
        _scaled_mm_1: "f32[768, 50304]" = torch.ops.aten._scaled_mm.default(clone_1, permute_4, full_default, full_default_2, None, None, torch.float32);  clone_1 = permute_4 = full_default = full_default_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:99 in impl, code: ).T
        permute_5: "f32[50304, 768]" = torch.ops.aten.permute.default(_scaled_mm_1, [1, 0]);  _scaled_mm_1 = None
        return (_scaled_mm, permute_5)
