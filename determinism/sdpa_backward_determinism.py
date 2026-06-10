import torch, torch.nn.functional as F
torch.manual_seed(0)
dev="cuda"; dt=torch.bfloat16
# representative attention shape (qwen-image-ish): batch, heads, seq, head_dim
B,H,S,D = 8, 24, 1024, 128

def make():
    g=torch.Generator(device=dev).manual_seed(123)
    q=torch.randn(B,H,S,D,generator=g,device=dev,dtype=dt,requires_grad=True)
    k=torch.randn(B,H,S,D,generator=g,device=dev,dtype=dt,requires_grad=True)
    v=torch.randn(B,H,S,D,generator=g,device=dev,dtype=dt,requires_grad=True)
    go=torch.randn(B,H,S,D,generator=g,device=dev,dtype=dt)
    return q,k,v,go

def run_once():
    q,k,v,go=make()
    out=F.scaled_dot_product_attention(q,k,v)
    out.backward(go)
    return out.detach().clone(), q.grad.detach().clone(), k.grad.detach().clone(), v.grad.detach().clone()

def bitcmp(a,b): return bool(torch.equal(a,b))

def maxreldiff(a,b):
    d=(a.float()-b.float()).abs().max().item()
    s=b.float().abs().max().item()+1e-30
    return d/s

print("=== which SDPA backend is selected for this shape ===")
from torch.nn.attention import SDPBackend, sdpa_kernel
q,k,v,go=make()
for be in [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH]:
    try:
        with sdpa_kernel(be):
            o=F.scaled_dot_product_attention(q,k,v)
        print(f"  {be.name}: usable")
    except Exception as e:
        print(f"  {be.name}: NOT usable ({type(e).__name__})")

print("=== DEFAULT mode (deterministic_algorithms OFF) ===")
print("deterministic_enabled =", torch.are_deterministic_algorithms_enabled())
o1,dq1,dk1,dv1=run_once()
o2,dq2,dk2,dv2=run_once()
print(f"  forward  bit-identical: {bitcmp(o1,o2)}")
print(f"  dQ       bit-identical: {bitcmp(dq1,dq2)}   maxrel={maxreldiff(dq1,dq2):.2e}")
print(f"  dK       bit-identical: {bitcmp(dk1,dk2)}   maxrel={maxreldiff(dk1,dk2):.2e}")
print(f"  dV       bit-identical: {bitcmp(dv1,dv2)}   maxrel={maxreldiff(dv1,dv2):.2e}")

print("=== deterministic mode ON (use_deterministic_algorithms warn_only) ===")
torch.use_deterministic_algorithms(True, warn_only=True)
o1,dq1,dk1,dv1=run_once()
o2,dq2,dk2,dv2=run_once()
print(f"  forward  bit-identical: {bitcmp(o1,o2)}")
print(f"  dQ       bit-identical: {bitcmp(dq1,dq2)}   maxrel={maxreldiff(dq1,dq2):.2e}")
print(f"  dK       bit-identical: {bitcmp(dk1,dk2)}   maxrel={maxreldiff(dk1,dk2):.2e}")
print(f"  dV       bit-identical: {bitcmp(dv1,dv2)}   maxrel={maxreldiff(dv1,dv2):.2e}")
