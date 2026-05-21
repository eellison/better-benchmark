class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "Sym(s67)", arg1_1: "f32[64, s67, 1, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:278 in torch_dynamo_resume_in__forward_impl_at_276, code: x = self.avgpool(x)
        mean: "f32[64, s67, 1, 1]" = torch.ops.aten.mean.dim(arg1_1, [-1, -2], True);  arg1_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:279 in torch_dynamo_resume_in__forward_impl_at_276, code: x = torch.flatten(x, 1)
        view: "f32[64, s67]" = torch.ops.aten.reshape.default(mean, [64, arg0_1]);  mean = arg0_1 = None
        return (view,)
