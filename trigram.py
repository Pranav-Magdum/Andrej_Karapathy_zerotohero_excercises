import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

words=open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi={s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos={i:s for s, i in stoi.items()}

xs, ys = [], []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2, ch3 in zip (chs,chs[1:],chs[2:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        ix3 = stoi[ch3]
        xs.append((ix1,ix2))
        ys.append(ix3)

xs= torch.tensor(xs)
ys= torch.tensor(ys)
w = torch.randn((54,27), requires_grad=True)

for k in range(50):
    xenc=F.one_hot(xs, num_classes=27).float().view(-1,54)
  
    logits=xenc @ w
    counts = logits.exp()
    probs=counts/counts.sum(1, keepdim=True)
    loss = -probs[torch.arange(len(ys)), ys].log().mean()
    w.grad=None
    loss.backward()

    w.data += -.1*w.grad
  
for i in range(10):
    out= []
    ix1, ix2=0, 0
  
    while True:
         xenc = F.one_hot(torch.tensor([[ix1,ix2]]), num_classes=27).float().view(-1,54)
         logits = xenc@w # predict log-counts
         counts = logits.exp() # counts, equivalent to N
         p =counts/counts.sum(1, keepdim=True)
         ix_next=torch.multinomial(p, num_samples=1, replacement=True).item()
         out.append(itos[ix_next])
         if ix_next==0:
            break
         ix1, ix2 = ix2, ix_next
    print(''.join(out))
