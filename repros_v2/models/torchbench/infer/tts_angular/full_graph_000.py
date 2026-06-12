class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "bf16[64, 50, 256][12800, 256, 1]cuda:0"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/tts_angular/model.py:75 in torch_dynamo_resume_in_forward_at_73, code: d = torch.nn.functional.normalize(d[:, -1], p=2, dim=1)
        select: "bf16[64, 256][12800, 1]cuda:0" = torch.ops.aten.select.int(arg0_1, 1, -1);  arg0_1 = None
        convert_element_type: "f32[64, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(select, torch.float32)
        pow_1: "f32[64, 256][256, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 2);  convert_element_type = None
        sum_1: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(pow_1, [1], True);  pow_1 = None
        pow_2: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(sum_1, 0.5);  sum_1 = None
        convert_element_type_1: "bf16[64, 1][1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(pow_2, torch.bfloat16);  pow_2 = None
        clamp_min: "bf16[64, 1][1, 1]cuda:0" = torch.ops.aten.clamp_min.default(convert_element_type_1, 1e-12);  convert_element_type_1 = None
        expand: "bf16[64, 256][1, 0]cuda:0" = torch.ops.aten.expand.default(clamp_min, [64, 256]);  clamp_min = None
        div: "bf16[64, 256][256, 1]cuda:0" = torch.ops.aten.div.Tensor(select, expand);  select = expand = None
        return (div,)
