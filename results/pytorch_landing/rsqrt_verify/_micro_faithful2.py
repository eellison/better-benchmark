"""Faithful two-branch microbench, top-level jit kernels (sqrt_rn+div vs rsqrt)."""
import torch, triton, triton.language as tl
from triton.language.extra import libdevice
from triton.testing import do_bench

N = 102760448
C = 64

@triton.jit
def k_sqrt(io_ptr,p1,p2,p3,p4,p5,p6,p7,p8, xnumel, XBLOCK: tl.constexpr):
    xoffset=tl.program_id(0)*XBLOCK; xindex=xoffset+tl.arange(0,XBLOCK)[:]
    x2=xindex; x0=xindex%64
    tmp0=tl.load(io_ptr+x2,None).to(tl.float32)
    tmp2=tl.load(p1+x0,None,eviction_policy='evict_last').to(tl.float32)
    tmp5=tl.load(p2+x0,None,eviction_policy='evict_last').to(tl.float32)
    tmp13=tl.load(p3+x0,None,eviction_policy='evict_last').to(tl.float32)
    tmp16=tl.load(p4+x0,None,eviction_policy='evict_last').to(tl.float32)
    tmp22=tl.load(p5+x0,None,eviction_policy='evict_last').to(tl.float32)
    tmp25=tl.load(p6+x0,None,eviction_policy='evict_last').to(tl.float32)
    tmp31=tl.load(p7+x0,None,eviction_policy='evict_last').to(tl.float32)
    tmp34=tl.load(p8+x0,None,eviction_policy='evict_last').to(tl.float32)
    eps=tl.full([1],1e-5,tl.float32); one=tl.full([1],1.0,tl.float32)
    tmp4=tmp0-tmp2; tmp8=tmp5+eps
    tmp9=tl.sqrt_rn(tmp8); tmp11=one/tmp9; tmp11b=tmp11*one; tmp12=tmp4*tmp11b
    tmp15=tmp12*tmp13; tmp18=tmp15+tmp16
    tmp24=tmp0-tmp22; tmp27=tmp25+eps
    tmp28=tl.sqrt_rn(tmp27); tmp29=one/tmp28; tmp29b=tmp29*one; tmp30=tmp24*tmp29b
    tmp33=tmp30*tmp31; tmp36=tmp33+tmp34
    tmp38=tmp18+tmp36; z=tl.full([1],0,tl.int32)
    tmp40=tl.maximum(z,tmp38)
    tl.store(io_ptr+x2, tmp40.to(tl.bfloat16), None)

@triton.jit
def k_rsqrt(io_ptr,p1,p2,p3,p4,p5,p6,p7,p8, xnumel, XBLOCK: tl.constexpr):
    xoffset=tl.program_id(0)*XBLOCK; xindex=xoffset+tl.arange(0,XBLOCK)[:]
    x2=xindex; x0=xindex%64
    tmp0=tl.load(io_ptr+x2,None).to(tl.float32)
    tmp2=tl.load(p1+x0,None,eviction_policy='evict_last').to(tl.float32)
    tmp5=tl.load(p2+x0,None,eviction_policy='evict_last').to(tl.float32)
    tmp13=tl.load(p3+x0,None,eviction_policy='evict_last').to(tl.float32)
    tmp16=tl.load(p4+x0,None,eviction_policy='evict_last').to(tl.float32)
    tmp22=tl.load(p5+x0,None,eviction_policy='evict_last').to(tl.float32)
    tmp25=tl.load(p6+x0,None,eviction_policy='evict_last').to(tl.float32)
    tmp31=tl.load(p7+x0,None,eviction_policy='evict_last').to(tl.float32)
    tmp34=tl.load(p8+x0,None,eviction_policy='evict_last').to(tl.float32)
    eps=tl.full([1],1e-5,tl.float32); one=tl.full([1],1.0,tl.float32)
    tmp4=tmp0-tmp2; tmp8=tmp5+eps
    tmp9=libdevice.rsqrt(tmp8); tmp11=tmp9*one; tmp12=tmp4*tmp11
    tmp15=tmp12*tmp13; tmp18=tmp15+tmp16
    tmp24=tmp0-tmp22; tmp27=tmp25+eps
    tmp28=libdevice.rsqrt(tmp27); tmp29=tmp28*one; tmp30=tmp24*tmp29
    tmp33=tmp30*tmp31; tmp36=tmp33+tmp34
    tmp38=tmp18+tmp36; z=tl.full([1],0,tl.int32)
    tmp40=tl.maximum(z,tmp38)
    tl.store(io_ptr+x2, tmp40.to(tl.bfloat16), None)

io = torch.randn(N, dtype=torch.bfloat16, device='cuda')
ps = [torch.rand(C, dtype=torch.bfloat16, device='cuda').abs()+0.1 for _ in range(8)]

def bench(kern):
    best=None
    for XBLOCK in [256,512,1024,2048]:
        for nw in [4,8]:
            grid=(triton.cdiv(N,XBLOCK),)
            def run(): kern[grid](io,*ps,N,XBLOCK=XBLOCK,num_warps=nw)
            try:
                run(); torch.cuda.synchronize()
                t=do_bench(run,return_mode="min")*1000
            except Exception as e:
                t=float('inf')
            if best is None or t<best[0]: best=(t,XBLOCK,nw)
    return best

bs=bench(k_sqrt); br=bench(k_rsqrt)
print(f"sqrt_rn+div (pos22): {bs[0]:7.1f} us  XBLOCK={bs[1]} nw={bs[2]}")
print(f"rsqrt       (pos23): {br[0]:7.1f} us  XBLOCK={br[1]} nw={br[2]}")
print(f"speedup: {bs[0]/br[0]:.2f}x")
src=torch.empty(N,dtype=torch.bfloat16,device='cuda'); dst=torch.empty_like(src)
print(f"copy SOL (1R1W): {do_bench(lambda: dst.copy_(src),return_mode='min')*1000:.1f} us")
