class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "Sym(s67)", arg1_1: "bf16[64, s67, 1, 1][s67, 1, 1, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:94 in torch_dynamo_resume_in_forward_at_93, code: out = self.relu(out)
        relu: "bf16[64, s67, 1, 1][s67, 1, 1, 1]cuda:0" = torch.ops.aten.relu.default(arg1_1)
        copy_: "bf16[64, s67, 1, 1][s67, 1, 1, 1]cuda:0" = torch.ops.aten.copy_.default(arg1_1, relu);  arg1_1 = relu = None
        return (copy_,)
