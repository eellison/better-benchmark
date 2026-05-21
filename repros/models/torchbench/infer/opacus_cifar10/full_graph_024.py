class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[10, 512]", arg1_1: "f32[10]", arg2_1: "f32[64, 512]"):
        # No stacktrace found for following nodes
        permute: "f32[512, 10]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        addmm: "f32[64, 10]" = torch.ops.aten.addmm.default(arg1_1, arg2_1, permute);  arg1_1 = arg2_1 = permute = None
        return (addmm,)
