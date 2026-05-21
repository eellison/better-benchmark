class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "Sym(s57)", arg1_1: "Sym(s39)", arg2_1: "f32[64, 128, s57, s39]", arg3_1: "f32[64, 128, s57, s39]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:102 in torch_dynamo_resume_in_forward_at_100, code: out += identity
        add_10: "f32[64, 128, s57, s39]" = torch.ops.aten.add.Tensor(arg2_1, arg3_1);  arg3_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in torch_dynamo_resume_in_forward_at_100, code: out = self.relu(out)
        relu: "f32[64, 128, s57, s39]" = torch.ops.aten.relu.default(add_10);  add_10 = None
        copy_: "f32[64, 128, s57, s39]" = torch.ops.aten.copy_.default(arg2_1, relu);  arg2_1 = relu = None
        return (copy_,)
