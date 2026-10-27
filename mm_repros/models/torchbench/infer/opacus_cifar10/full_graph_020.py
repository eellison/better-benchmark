class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "Sym(s49)", arg1_1: "Sym(s15)", arg2_1: "Sym(s39)", arg3_1: "bf16[64, s49, s15, s39][s15*s39*s49, s15*s39, s39, 1]cuda:0", arg4_1: "bf16[64, s49, s15, s39][s15*s39*s49, s15*s39, s39, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:102 in torch_dynamo_resume_in_forward_at_97, code: out += identity
        add_10: "bf16[64, s49, s15, s39][s15*s39*s49, s15*s39, s39, 1]cuda:0" = torch.ops.aten.add.Tensor(arg3_1, arg4_1);  arg4_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in torch_dynamo_resume_in_forward_at_97, code: out = self.relu(out)
        relu: "bf16[64, s49, s15, s39][s15*s39*s49, s15*s39, s39, 1]cuda:0" = torch.ops.aten.relu.default(add_10);  add_10 = None
        copy_: "bf16[64, s49, s15, s39][s15*s39*s49, s15*s39, s39, 1]cuda:0" = torch.ops.aten.copy_.default(arg3_1, relu);  arg3_1 = relu = None
        return (copy_,)
