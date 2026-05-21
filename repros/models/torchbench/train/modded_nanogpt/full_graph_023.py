class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "bf16[6144, 768]", arg1_1: "f32[50304, 768]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:54 in impl, code: scale_a=x.new_tensor(x_s, dtype=torch.float32),
        _tensor_constant0: "f32[]" = self._tensor_constant0;  _tensor_constant0 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:55 in impl, code: scale_b=x.new_tensor(w_s, dtype=torch.float32),
        _tensor_constant1: "f32[]" = self._tensor_constant1;  _tensor_constant1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:48 in impl, code: x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        div: "bf16[6144, 768]" = torch.ops.aten.div.Tensor(arg0_1, 0.06185895741317419);  arg0_1 = None
        convert_element_type: "f8e4m3fn[6144, 768]" = torch.ops.prims.convert_element_type.default(div, torch.float8_e4m3fn);  div = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:49 in impl, code: w_f8 = w.div(w_s).to(torch.float8_e4m3fn)
        div_1: "f32[50304, 768]" = torch.ops.aten.div.Tensor(arg1_1, 0.001953125);  arg1_1 = None
        convert_element_type_1: "f8e4m3fn[50304, 768]" = torch.ops.prims.convert_element_type.default(div_1, torch.float8_e4m3fn);  div_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:52 in impl, code: w_f8.T,
        permute: "f8e4m3fn[768, 50304]" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0])

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:54 in impl, code: scale_a=x.new_tensor(x_s, dtype=torch.float32),
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.06185895577073097, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:55 in impl, code: scale_b=x.new_tensor(w_s, dtype=torch.float32),
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.001953125, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:50 in impl, code: out = torch._scaled_mm(
        _scaled_mm: "bf16[6144, 50304]" = torch.ops.aten._scaled_mm.default(convert_element_type, permute, full_default, full_default_1, None, None, torch.bfloat16, True);  permute = full_default = full_default_1 = None
        return (_scaled_mm, convert_element_type, convert_element_type_1)
