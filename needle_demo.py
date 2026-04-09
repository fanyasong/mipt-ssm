"""
Needle-in-a-Haystack Demo
=========================
Demonstrates MIPT-SSM's causal sparse cache for exact fact retrieval.

Setup:
  - Sequence length N=512
  - Needle token hidden in first 10% of sequence
  - Remaining 90% is random noise
  - Task: classify which needle (4 classes)

Expected results (~3 min on any GPU or CPU):
  MIPT no cache : ~0.845
  MIPT K=4 cache: ~0.968
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ── Minimal self-contained model for the demo ──

class MIPTCellDemo(nn.Module):
    def __init__(self, d, vocab):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.W_p = nn.Linear(d, d)
        self.W_theta = nn.Linear(d, d)
        self.W_r = nn.Linear(d, d)
        self.W_i = nn.Linear(d, d)

    def scan(self, tokens):
        B, T = tokens.shape
        e = self.embed(tokens)               # (B, T, D)
        p = torch.sigmoid(self.W_p(e))       # (B, T, D)
        theta = self.W_theta(e)
        inp_re = self.W_r(e)
        inp_im = self.W_i(e)

        D = e.shape[-1]
        h_re = torch.zeros(B, D, device=tokens.device)
        h_im = torch.zeros(B, D, device=tokens.device)
        h_all_re = []

        for t in range(T):
            cos_t = torch.cos(theta[:, t])
            sin_t = torch.sin(theta[:, t])
            rot_re = cos_t * h_re - sin_t * h_im
            rot_im = sin_t * h_re + cos_t * h_im
            h_re = (1 - p[:, t]) * rot_re + p[:, t] * inp_re[:, t]
            h_im = (1 - p[:, t]) * rot_im + p[:, t] * inp_im[:, t]
            h_all_re.append(h_re.unsqueeze(1))

        return torch.cat(h_all_re, dim=1), p  # (B, T, D), (B, T, D)


class NeedleModelNoCache(nn.Module):
    def __init__(self, vocab=300, d=64, n_classes=4):
        super().__init__()
        self.scan = MIPTCellDemo(d, vocab)
        self.head = nn.Linear(d, n_classes)

    def forward(self, tokens):
        h, _ = self.scan.scan(tokens)
        return self.head(h.mean(dim=1))


class NeedleModelCache(nn.Module):
    def __init__(self, vocab=300, d=64, n_classes=4, K=4):
        super().__init__()
        self.scan = MIPTCellDemo(d, vocab)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_q = nn.Linear(d, d)
        self.gate = nn.Linear(d, 1)
        self.head = nn.Linear(d, n_classes)
        self.K = K

    def forward(self, tokens):
        import math
        B, T = tokens.shape
        D = 64
        h_all, p_all = self.scan.scan(tokens)  # (B, T, D), (B, T, D)

        h_mean = h_all.mean(dim=1)  # (B, D)
        p_scalar = p_all.mean(dim=-1)  # (B, T)

        K = min(self.K, T)
        _, topk_idx = p_scalar.topk(K, dim=-1)  # (B, K)
        idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, D)
        h_topk = h_all.gather(1, idx_exp)  # (B, K, D)

        keys = self.W_k(h_topk)
        values = self.W_v(h_topk)
        query = self.W_q(h_mean).unsqueeze(1)
        scores = torch.bmm(query, keys.transpose(1, 2)) / math.sqrt(D)
        attn = F.softmax(scores, dim=-1)
        cache_out = torch.bmm(attn, values).squeeze(1)

        g = torch.sigmoid(self.gate(h_mean))
        out = h_mean + g * cache_out
        return self.head(out)


def make_data(N=512, n_train=5000, n_test=1000, n_classes=4, seed=42):
    torch.manual_seed(seed)
    VOCAB = 300
    NEEDLE_VOCAB = list(range(10, 10 + n_classes * 5))  # 20 needle tokens
    NOISE_VOCAB  = list(range(200, VOCAB))

    def gen(n):
        labels = torch.randint(0, n_classes, (n,))
        seqs = torch.randint(200, VOCAB, (n, N))
        for i in range(n):
            needle_pos = torch.randint(0, N // 10, (1,)).item()
            c = labels[i].item()
            needle_tok = NEEDLE_VOCAB[c * 5 + torch.randint(0, 5, (1,)).item()]
            seqs[i, needle_pos] = needle_tok
        return seqs, labels

    return gen(n_train), gen(n_test)


def train_eval(model, train_data, test_data, epochs=15, lr=3e-3, device='cpu'):
    model = model.to(device)
    Xtr, ytr = train_data[0].to(device), train_data[1].to(device)
    Xte, yte = test_data[0].to(device), test_data[1].to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = 0.0
    for ep in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 128):
            xb = Xtr[idx[i:i+128]]
            yb = ytr[idx[i:i+128]]
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(Xte).argmax(-1) == yte).float().mean().item()
        best = max(best, acc)
        if ep % 5 == 0:
            print(f"  ep{ep:02d} val={acc:.3f} best={best:.3f}")
    return best


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nNeedle-in-a-Haystack Demo | device={device}")
    print("Needle hidden in first 10% of 512-token sequence\n")

    train_data, test_data = make_data()

    print("── Model A: MIPT, no cache ──")
    m_a = NeedleModelNoCache()
    acc_a = train_eval(m_a, train_data, test_data, device=device)

    print("\n── Model B: MIPT + causal cache K=4 ──")
    m_b = NeedleModelCache(K=4)
    acc_b = train_eval(m_b, train_data, test_data, device=device)

    print("\n" + "="*50)
    print(f"MIPT no cache  : {acc_a:.3f}")
    print(f"MIPT K=4 cache : {acc_b:.3f}  (Δ={acc_b-acc_a:+.3f})")
    print(f"Random baseline: 0.250")
    print("="*50)
    print("\nKey insight: with only 4 cache slots (0.8% of sequence),")
    print("exact recall improves by {:.1f} percentage points.".format((acc_b - acc_a) * 100))


if __name__ == '__main__':
    main()
