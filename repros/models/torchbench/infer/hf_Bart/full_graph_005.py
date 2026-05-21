class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f16[50265, 768]", arg1_1: "f16[1, 512, 768]", arg2_1: "f16[1, 50265]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:939 in torch_dynamo_resume_in_forward_at_926, code: lm_logits = self.lm_head(outputs[0])
        view: "f16[512, 768]" = torch.ops.aten.reshape.default(arg1_1, [512, 768]);  arg1_1 = None
        permute: "f16[768, 50265]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        constant_pad_nd_default: "f16[768, 50272]" = torch.ops.aten.constant_pad_nd.default(permute, [0, 7, 0, 0]);  permute = None
        mm_default: "f16[512, 50272]" = torch.ops.aten.mm.default(view, constant_pad_nd_default);  view = constant_pad_nd_default = None
        slice_tensor: "f16[512, 50265]" = torch.ops.aten.slice.Tensor(mm_default, 1, 0, -7);  mm_default = None
        view_1: "f16[1, 512, 50265]" = torch.ops.aten.reshape.default(slice_tensor, [1, 512, 50265]);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:940 in torch_dynamo_resume_in_forward_at_926, code: lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)
        add: "f16[1, 512, 50265]" = torch.ops.aten.add.Tensor(view_1, arg2_1);  view_1 = arg2_1 = None
        return (add,)
