class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "bf16[64, 64, 8, 8][4096, 64, 8, 1]cuda:0", arg1_1: "bf16[64, 64, 8, 8][4096, 64, 8, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:102 in torch_dynamo_resume_in_forward_at_97, code: out += identity
        add: "bf16[64, 64, 8, 8][4096, 64, 8, 1]cuda:0" = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg1_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in torch_dynamo_resume_in_forward_at_97, code: out = self.relu(out)
        relu: "bf16[64, 64, 8, 8][4096, 64, 8, 1]cuda:0" = torch.ops.aten.relu.default(add);  add = None
        copy_: "bf16[64, 64, 8, 8][4096, 64, 8, 1]cuda:0" = torch.ops.aten.copy_.default(arg0_1, relu);  arg0_1 = relu = None
        return (copy_,)
