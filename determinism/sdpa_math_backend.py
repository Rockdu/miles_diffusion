import torch, torch.nn.functional as F, time
from torch.nn.attention import SDPBackend, sdpa_kernel
dev="cuda"; dt=torch.bfloat16; B,H,S,D=8,24,1024,128
def make():
    g=torch.Generator(device=dev).manual_seed(123)
    q=torch.randn(B,H,S,D,generator=g,device=dev,dtype=dt,requires_grad=True)
    k=torch.randn(B,H,S,D,generator=g,device=dev,dtype=dt,requires_grad=True)
    v=torch.randn(B,H,S,D,generator=g,device=dev,dtype=dt,requires_grad=True)
    go=torch.randn(B,H,S,D,generator=g,device=dev,dtype=dt)
    return q,k,v,go
def run(be):
    q,k,v,go=make()
    ctx = sdpa_kernel(be) if be else torch.autograd.grad_mode.no_grad.__enter__ and __import__("contextlib").nullcontext()
    import contextlib
    with (sdpa_kernel(be) if be else contextlib.nullcontext()):
        out=F.scaled_dot_product_attention(q,k,v); out.backward(go)
    return q.grad.clone()
def eq(a,b): return bool(torch.equal(a,b))
def tm(be,n=20):
    q,k,v,go=make(); import contextlib
    torch.cuda.synchronize(); t=time.time()
    for _ in range(n):
        q.grad=None
        with (sdpa_kernel(be) if be else contextlib.nullcontext()):
            out=F.scaled_dot_product_attention(q,k,v); out.backward(go)
    torch.cuda.synchronize(); return (time.time()-t)/n*1000
for be,name in [(None,"DEFAULT(auto)"),(SDPBackend.MATH,"MATH")]:
    d1=run(be); d2=run(be)
    print(f"{name:14} dQ bit-identical(2 runs): {eq(d1,d2)}   bwd+fwd time={tm(be):.2f} ms")
