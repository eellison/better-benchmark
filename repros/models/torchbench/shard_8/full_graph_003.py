class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "Sym(s77)", tanh: "f16[s77, 1][1, 1]cuda:0", tangents_1: "f16[s77, 1][1, 1]cuda:0"):
        # File: /torchbench/torchbenchmark/util/distribution.py:31 in _call, code: return x.tanh()
        convert_element_type: "f32[s77, 1][1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(tangents_1, torch.float32);  tangents_1 = None
        convert_element_type_1: "f32[s77, 1][1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(tanh, torch.float32);  tanh = None
        mul_2: "f32[s77, 1][1, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1, convert_element_type_1);  convert_element_type_1 = None
        sub_2: "f32[s77, 1][1, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_2);  mul_2 = None
        mul_3: "f32[s77, 1][1, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type, sub_2);  convert_element_type = sub_2 = None
        convert_element_type_2: "f16[s77, 1][1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3, torch.float16);  mul_3 = None
        return (None, None, convert_element_type_2)
