# MIPT-SSM: Wave-Particle Duality for Neural Sequence Processing

> **O(N) training memory · O(K·D) inference cache · Competitive LM quality**

📄 **Paper:** *Wave-Particle Duality Neural Sequence Processing System and Method* — arXiv link coming soon  
🔬 **Patent:** CN Patent Application Filed (April 2025)

---

## The Core Idea

Every existing sequence model is stuck in a dilemma:

| Model | Memory | Long-range recall |
|-------|--------|-------------------|
| Transformer | O(N²) → OOM at N>16K | ✅ Perfect |
| Mamba / SSM | O(N) ✅ | ❌ Exponential decay |
| **MIPT-SSM** | **O(N) training, O(K·D) inference** ✅ | **✅ 96.8% recall** |

The reason is a mathematical dead-lock: in any single linear operator, **norm-preservation** (needed for long-range semantics) and **selective decay** (needed for local fact injection) **cannot hold simultaneously**.

MIPT-SSM breaks this with **wave-particle duality**:

```
h_t = (1 - p_t) ⊙ exp(i·θ_t) ⊙ h_{t-1}   ← wave: lossless phase rotation
    +       p_t ⊙ B·e_t                       ← particle: collapse to current token
```

- `p_t` (measurement rate) and `θ_t` (phase angle) are **two completely independent** learnable weight matrices — the network decides *how much* to collapse vs. propagate at each token
- `exp(i·θ_t)` has unit modulus → **no information loss** during wave propagation
- A **causal sparse KV cache** stores only high-`p_t` tokens (≈1-5%) for exact fact retrieval

---

## Key Results

### Memory vs. Sequence Length
```
N=512  : MIPT   32 MB  |  Transformer   71 MB   (2.2×)
N=2048 : MIPT  130 MB  |  Transformer  589 MB   (4.5×)
N=8192 : MIPT  810 MB  |  Transformer 34,651 MB (42.8×)
N=16384: MIPT ~1.6 GB  |  Transformer  💥 OOM
```

### Classification (AG News, N=512, 3 seeds)
```
MIPT-SSM  : 0.905 ± 0.002   ← 248K parameters
Transformer: 0.736 ± 0.001   ← 422K parameters
Improvement: +16.6%  with 41% fewer parameters
```

### Needle-in-a-Haystack (N=512, needle in first 10%)
```
No cache (pure MIPT)  : 0.845
Causal cache K=1      : 0.960   (1 slot!)
Causal cache K=4      : 0.968   (gap to perfect: only 3.2%)
Causal cache K=16     : 0.992
Transformer (full attn): 1.000  (needs 34,651 MB at N=8K)
```

### Language Modeling PPL (WikiText-103, 14M params)
```
Transformer baseline  : PPL = 90.5
MIPT (K=0, pure SSM)  : PPL = 102.2  (+13%)
MIPT + cache K=16     : PPL = 96.3   (+6.4%)
MIPT + cache K=64     : PPL = 92.1   (+1.8%) ← 5400× less inference memory
```

### Training Throughput (tokens/sec)
```
N=128 : MIPT 0.46× TF   (short seq, TF wins with FlashAttn)
N=512 : MIPT 1.09× TF   ← crossover point
N=2048: MIPT 1.67× TF
N=4096: MIPT 2.16× TF   ← O(N log N) vs O(N²) FLOPs
```

### In-Context Learning
MIPT achieves **1.000 accuracy by epoch 2** on associative recall, demonstrating natural emergence of induction heads — a known failure mode for pure SSMs.

---

## Architecture

```
Input tokens
    ↓
Embedding (BPE, V=100,277)
    ↓
L × MIPT-Block:
  ├─ LayerNorm
  ├─ Parallel MIPT Scan (O(log N) depth)
  │    p_t  = σ(W_p · e_t)          ← measurement rate
  │    θ_t  = W_θ · e_t             ← phase angle (W_p, W_θ independent)
  │    h_t  = (1-p_t)·exp(iθ_t)·h_{t-1} + p_t·B·e_t
  ├─ Causal Sparse Cache (top-K by p_t)
  │    write: if p_scalar_t > τ → KV slot
  │    read:  attend(h_mean, cache_KV)
  ├─ Gated fusion: h_out = h_wave + g · h_cache
  └─ FFN + residual
    ↓
Classification head / LM head
```

---

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/mipt-ssm
cd mipt-ssm
pip install -r requirements.txt
```

### Needle-in-a-Haystack Demo
```bash
python demos/needle_demo.py
# Expected output:
# MIPT (no cache): 0.845
# MIPT + K=4 causal cache: 0.968
# Runtime: ~3 minutes on any GPU
```

### Text Classification
```bash
python experiments/classify_agnews.py --model mipt --d_model 128 --n_layers 2
```

### Language Modeling
```bash
python experiments/language_model.py --cache_k 16 --data_path ./data/wikitext
```

---

## What's Missing (Call for Collaboration)

This repo proves the theory at **14M–31M parameter scale**. To reach production:

1. **Triton/CUDA kernel** for the parallel scan — current PyTorch loop is 10-50× slower than it needs to be. If you're a CUDA wizard, this is the highest-leverage contribution.

2. **100M+ parameter pretraining** — one A100 week (~$1000) would give us a real scaling curve. If you have compute to spare, let's talk.

3. **Downstream task evaluation** — GLUE, long-context benchmarks (SCROLLS, RULER)

If you want to collaborate: open an issue or email `[your email]`

---

## Physical Motivation

The architecture is grounded in **Measurement-Induced Phase Transitions (MIPT)** from quantum information theory. In monitored quantum circuits, frequent measurement (high `p_t`) drives the system toward an area-law entanglement phase (particle-like, local facts); rare measurement (low `p_t`) drives it toward a volume-law phase (wave-like, global semantics).

The critical point N* ≈ D·16 (where D is state dimension) matches the empirically observed context length where Transformer's quadratic cost starts dominating — providing a physical explanation for why O(N²) attention exists and how to escape it.

See the paper for the full mathematical treatment including the dead-lock proof, parallel scan derivation, and entanglement entropy analysis.

---

## Citation

```bibtex
@article{mipt-ssm-2025,
  title   = {Wave-Particle Duality Neural Sequence Processing System and Method},
  author  = {[Your Name]},
  journal = {arXiv},
  year    = {2025},
  note    = {arXiv link TBD}
}
```

---

## License

MIT
