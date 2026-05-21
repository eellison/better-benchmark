class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[64, 64, 8, 8]", arg1_1: "f32[64, 64, 8, 8]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:102 in torch_dynamo_resume_in_forward_at_97, code: out += identity
        add: "f32[64, 64, 8, 8]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg1_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in torch_dynamo_resume_in_forward_at_97, code: out = self.relu(out)
        relu: "f32[64, 64, 8, 8]" = torch.ops.aten.relu.default(add);  add = None
        copy_: "f32[64, 64, 8, 8]" = torch.ops.aten.copy_.default(arg0_1, relu);  arg0_1 = relu = None
        return (copy_,)
