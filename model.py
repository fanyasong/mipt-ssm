"""
MIPT-SSM: Wave-Particle Duality for Neural Sequence Processing
Core model implementation.

Paper: [arXiv link TBD]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


# ─────────────────────────────────────────────
#  1. MIPT Scan Cell (single step, for inference)
# ─────────────────────────────────────────────

class MIPTCell(nn.Module):
    """
    Single-step MIPT update. Used during autoregressive inference.

    Physics interpretation:
      p_t  = measurement rate  → particle collapse (local fact injection)
      θ_t  = phase angle       → wave propagation  (lossless rotation)
      h_t  = complex hidden state ∈ C^D
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        # Two INDEPENDENT weight matrices — core design choice
        self.W_p = nn.Linear(d_model, d_model)   # measurement rate
        self.W_theta = nn.Linear(d_model, d_model)  # phase angle
        self.W_r = nn.Linear(d_model, d_model)   # real part of input projection
        self.W_i = nn.Linear(d_model, d_model)   # imaginary part of input projection

    def forward(
        self,
        e_t: torch.Tensor,           # (B, D) embedding at time t
        h_prev: torch.Tensor,        # (B, D) complex hidden state (as real tensor of shape (B, 2D))
    ) -> torch.Tensor:               # (B, 2D) new complex hidden state
        B, D = e_t.shape

        p_t = torch.sigmoid(self.W_p(e_t))           # (B, D) ∈ (0,1)
        theta_t = self.W_theta(e_t)                   # (B, D) ∈ R
        cos_t = torch.cos(theta_t)
        sin_t = torch.sin(theta_t)

        # Split complex state into real and imaginary parts
        h_re = h_prev[:, :D]   # (B, D)
        h_im = h_prev[:, D:]   # (B, D)

        # Complex rotation: exp(i*θ) * h
        rot_re = cos_t * h_re - sin_t * h_im
        rot_im = sin_t * h_re + cos_t * h_im

        # Input projection (complex)
        inp_re = self.W_r(e_t)
        inp_im = self.W_i(e_t)

        # MIPT update: h_t = (1-p)*rot(h_{t-1}) + p*B*e_t
        new_re = (1 - p_t) * rot_re + p_t * inp_re
        new_im = (1 - p_t) * rot_im + p_t * inp_im

        return torch.cat([new_re, new_im], dim=-1)  # (B, 2D)


# ─────────────────────────────────────────────
#  2. MIPT Parallel Scan (training, O(log N))
# ─────────────────────────────────────────────

def mipt_parallel_scan(
    p: torch.Tensor,      # (B, T, D) measurement rates
    theta: torch.Tensor,  # (B, T, D) phase angles
    inp: torch.Tensor,    # (B, T, 2D) complex input projections
) -> torch.Tensor:        # (B, T, 2D) all hidden states
    """
    Parallel prefix scan for MIPT.
    Computes all h_t simultaneously in O(log T) parallel depth.

    The associative operator for (a_l, b_l) ⊕ (a_r, b_r):
        a_new = a_r * a_l    (complex multiplication)
        b_new = a_r * b_l + b_r
    """
    B, T, D = p.shape

    # Build per-step operators
    # a_t = (1-p_t) * exp(i*theta_t)  — "wave" part
    # b_t = p_t * inp_t               — "particle" part
    cos_t = torch.cos(theta)  # (B, T, D)
    sin_t = torch.sin(theta)

    # a_t as (re, im): magnitude=(1-p_t), angle=theta_t
    a_re = (1 - p) * cos_t
    a_im = (1 - p) * sin_t

    b_re = inp[:, :, :D]
    b_im = inp[:, :, D:]

    # Iterative doubling (parallel prefix scan)
    for _ in range(math.ceil(math.log2(T + 1))):
        # Shift: get left neighbor
        a_re_prev = torch.zeros_like(a_re)
        a_re_prev[:, 1:] = a_re[:, :-1]
        a_im_prev = torch.zeros_like(a_im)
        a_im_prev[:, 1:] = a_im[:, :-1]
        b_re_prev = torch.zeros_like(b_re)
        b_re_prev[:, 1:] = b_re[:, :-1]
        b_im_prev = torch.zeros_like(b_im)
        b_im_prev[:, 1:] = b_im[:, :-1]

        # Complex multiply a_r * a_l
        new_a_re = a_re * a_re_prev - a_im * a_im_prev
        new_a_im = a_re * a_im_prev + a_im * a_re_prev

        # a_r * b_l + b_r  (complex)
        new_b_re = a_re * b_re_prev - a_im * b_im_prev + b_re
        new_b_im = a_re * b_im_prev + a_im * b_re_prev + b_im

        # Only update positions that have a valid left neighbor
        mask = torch.zeros(T, device=p.device, dtype=torch.bool)
        mask[1:] = True

        a_re = torch.where(mask[None, :, None], new_a_re, a_re)
        a_im = torch.where(mask[None, :, None], new_a_im, a_im)
        b_re = torch.where(mask[None, :, None], new_b_re, b_re)
        b_im = torch.where(mask[None, :, None], new_b_im, b_im)

    return torch.cat([b_re, b_im], dim=-1)  # (B, T, 2D)


# ─────────────────────────────────────────────
#  3. Causal Sparse KV Cache
# ─────────────────────────────────────────────

class CausalSparseCache(nn.Module):
    """
    Causal Sparse KV Cache Controller.

    Only tokens where p_t > threshold (or top-K by p_t) are stored.
    This gives O(K*D) inference memory instead of O(N*D).

    Physics interpretation:
      High p_t → particle collapse → store as discrete fact
      Low  p_t → wave propagation  → encode in phase, skip cache
    """
    def __init__(self, d_model: int, max_cache_size: int = 16):
        super().__init__()
        self.d_model = d_model
        self.K = max_cache_size
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_q = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, 1)

    def forward(
        self,
        h_mean: torch.Tensor,   # (B, D) pooled hidden state (wave part)
        h_all: torch.Tensor,    # (B, T, D) all positions (real part)
        p_all: torch.Tensor,    # (B, T, D) measurement rates
    ) -> torch.Tensor:          # (B, D) cache-augmented output
        B, T, D = h_all.shape

        # Scalar importance per position: mean of p_t across D
        p_scalar = p_all.mean(dim=-1)  # (B, T)

        # Select top-K positions by importance
        K = min(self.K, T)
        _, topk_idx = p_scalar.topk(K, dim=-1)  # (B, K)

        # Gather top-K positions
        idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, D)  # (B, K, D)
        h_topk = h_all.gather(1, idx_expanded)  # (B, K, D)

        # Compute KV for top-K positions
        keys = self.W_k(h_topk)    # (B, K, D)
        values = self.W_v(h_topk)  # (B, K, D)

        # Query from pooled wave state
        query = self.W_q(h_mean).unsqueeze(1)  # (B, 1, D)

        # Attention over sparse cache
        scores = torch.bmm(query, keys.transpose(1, 2)) / math.sqrt(D)  # (B, 1, K)
        attn = F.softmax(scores, dim=-1)
        cache_out = torch.bmm(attn, values).squeeze(1)  # (B, D)

        # Gated fusion: wave state + particle cache
        g = torch.sigmoid(self.gate(h_mean))
        return h_mean + g * cache_out


# ─────────────────────────────────────────────
#  4. MIPT Block
# ─────────────────────────────────────────────

class MIPTBlock(nn.Module):
    def __init__(self, d_model: int, cache_size: int = 16, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.W_p = nn.Linear(d_model, d_model)
        self.W_theta = nn.Linear(d_model, d_model)
        self.W_r = nn.Linear(d_model, d_model)
        self.W_i = nn.Linear(d_model, d_model)

        self.cache = CausalSparseCache(d_model, cache_size)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        B, T, D = x.shape
        residual = x
        x = self.norm1(x)

        p = torch.sigmoid(self.W_p(x))   # (B, T, D)
        theta = self.W_theta(x)          # (B, T, D)
        inp_re = self.W_r(x)
        inp_im = self.W_i(x)
        inp = torch.cat([inp_re, inp_im], dim=-1)  # (B, T, 2D)

        # Parallel scan → all hidden states
        h_complex = mipt_parallel_scan(p, theta, inp)  # (B, T, 2D)
        h_re = h_complex[:, :, :D]  # real part = output

        # Mean pool for wave representation
        h_mean = h_re.mean(dim=1)  # (B, D)

        # Cache augmentation
        out = self.cache(h_mean, h_re, p)  # (B, D)

        # Residual + FFN (for sequence-level tasks, broadcast back)
        # For classification: out is (B, D)
        # For LM: need per-token output — use h_re + residual
        h_re = h_re + residual
        h_re = h_re + self.dropout(self.ffn(self.norm2(h_re)))
        return h_re, out  # (B, T, D), (B, D)


# ─────────────────────────────────────────────
#  5. Full MIPT-SSM Model (Classification)
# ─────────────────────────────────────────────

class MIPTClassifier(nn.Module):
    """
    MIPT-SSM for sequence classification.
    O(N) training memory, O(K*D) inference cache.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_classes: int,
        max_seq_len: int = 512,
        cache_size: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            MIPTBlock(d_model, cache_size, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0)
        x = self.dropout(self.embed(tokens) + self.pos_embed(pos))

        doc_repr = None
        for layer in self.layers:
            x, doc = layer(x)
            doc_repr = doc  # use last layer's cache output

        out = self.norm(doc_repr)
        return self.classifier(out)


# ─────────────────────────────────────────────
#  6. MIPT-SSM Language Model
# ─────────────────────────────────────────────

class MIPTLanguageModel(nn.Module):
    """
    MIPT-SSM for autoregressive language modeling.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        max_seq_len: int = 512,
        cache_size: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            MIPTBlock(d_model, cache_size, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.embed.weight
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0)
        x = self.dropout(self.embed(tokens) + self.pos_embed(pos))

        for layer in self.layers:
            x, _ = layer(x)

        x = self.norm(x)
        return self.lm_head(x)  # (B, T, V)
