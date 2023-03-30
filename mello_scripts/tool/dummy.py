import torch

a = torch.randn(10000, 1000).cuda(0)
c = torch.randn(10000, 1000).cuda(1)
e = torch.randn(10000, 1000).cuda(2)
g = torch.randn(10000, 1000).cuda(3)
i = torch.randn(10000, 1000).cuda(4)
k = torch.randn(10000, 1000).cuda(5)
m = torch.randn(10000, 1000).cuda(6)
o = torch.randn(10000, 1000).cuda(7)

while True:
    b = torch.einsum('ih,jh->ij', (a, a))
    d = torch.einsum('ih,jh->ij', (c, c))
    e = torch.einsum('ih,jh->ij', (e, e))
    h = torch.einsum('ih,jh->ij', (g, g))
    j = torch.einsum('ih,jh->ij', (i, i))
    l = torch.einsum('ih,jh->ij', (k, k))
    n = torch.einsum('ih,jh->ij', (m, m))
    p = torch.einsum('ih,jh->ij', (o, o))

# python3 mello_scripts/tool/dummy.py