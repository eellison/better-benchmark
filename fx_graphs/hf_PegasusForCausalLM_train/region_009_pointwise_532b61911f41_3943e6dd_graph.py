class GraphModule(torch.nn.Module):
    def forward(self, primals_2: "f32[50265, 1024]", primals_1: "f32[128, 128, 1024]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:1114 in torch_dynamo_resume_in_forward_at_1100, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        permute_default: "f32[1024, 50265]" = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
        reshape_default: "f32[16384, 1024]" = torch.ops.aten.reshape.default(primals_1, _shape_param_0);  primals_1 = _shape_param_0 = None
        constant_pad_nd_default: "f32[1024, 50268]" = torch.ops.aten.constant_pad_nd.default(permute_default, [0, 3, 0, 0]);  permute_default = None
        return (reshape_default, constant_pad_nd_default)
