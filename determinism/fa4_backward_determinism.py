import torch, inspect
from flash_attn.cute import flash_attn_func
print("FA4 flash_attn_func params:", list(inspect.signature(flash_attn_func).parameters.keys()))
has_det = "deterministic" in inspect.signature(flash_attn_func).parameters
dev="cuda"; dt=torch.bfloat16
B,S,H,D=8,1024,24,128  # non-GQA: same heads for q/k/v
def make():
    g=torch.Generator(device=dev).manual_seed(123)
    q=torch.randn(B,S,H,D,generator=g,device=dev,dtype=dt,requires_grad=True)
    k=torch.randn(B,S,H,D,generator=g,device=dev,dtype=dt,requires_grad=True)
    v=torch.randn(B,S,H,D,generator=g,device=dev,dtype=dt,requires_grad=True)
    go=torch.randn(B,S,H,D,generator=g,device=dev,dtype=dt)
    return q,k,v,go
def run(det):
    q,k,v,go=make()
    kw={"deterministic":det} if has_det else {}
    out=flash_attn_func(q,k,v,**kw)
    if isinstance(out,tuple): out=out[0]
    out.backward(go)
    return q.grad.clone(),k.grad.clone(),v.grad.clone()
def eq(a,b): return bool(torch.equal(a,b))
def rel(a,b):
    d=(a.float()-b.float()).abs().max().item(); s=b.float().abs().max().item()+1e-30; return d/s
for det in ([False,True] if has_det else [False]):
    try:
        d1=run(det); d2=run(det)
        print(f"=== FA4 deterministic={det} ===")
        print(f"  dQ bit-identical: {eq(d1[0],d2[0])}  maxrel={rel(d1[0],d2[0]):.2e}")
        print(f"  dK bit-identical: {eq(d1[1],d2[1])}  maxrel={rel(d1[1],d2[1]):.2e}")
        print(f"  dV bit-identical: {eq(d1[2],d2[2])}  maxrel={rel(d1[2],d2[2]):.2e}")
    except Exception as e:
        print(f"=== FA4 deterministic={det}: ERROR {type(e).__name__}: {str(e)[:120]}")
