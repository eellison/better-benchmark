class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[50265, 1024]", primals_2: "f32[64, 256, 1024]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:756 in torch_dynamo_resume_in_forward_at_743, code: logits = self.output_projection(outputs[0])
        permute_default: "f32[1024, 50265]" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        reshape_default: "f32[16384, 1024]" = torch.ops.aten.reshape.default(primals_2, _shape_param_0);  primals_2 = _shape_param_0 = None
        constant_pad_nd_default: "f32[1024, 50268]" = torch.ops.aten.constant_pad_nd.default(permute_default, [0, 3, 0, 0]);  permute_default = None
        return (reshape_default, constant_pad_nd_default)
