# NanoGPT Reimplementation: From GPT-2 to Modern LLM Inference

A from-scratch reimplementation of nanoGPT with modern LLM optimizations.

The goal of this project is not only to reproduce GPT-2, but also to understand the evolution path from GPT-2 to modern LLM architectures such as LLaMA, Qwen and DeepSeek.

---

## Roadmap

### Phase 1 — Reproduce GPT-2 Architecture

* [ ] Character-level tokenizer
* [ ] BPE tokenizer integration
* [ ] Token Embedding
* [ ] GPT-2 Learned Position Embedding
* [ ] Multi-Head Self Attention
* [ ] Causal Mask
* [ ] Feed Forward Network (MLP)
* [ ] LayerNorm
* [ ] Residual Connection
* [ ] Training Pipeline
* [ ] Text Generation

Goal:

```text
Input Tokens
    ↓
Embedding
    ↓
Transformer Blocks
    ↓
LM Head
    ↓
Next Token Prediction
```

---

### Phase 2 — RoPE Position Embedding

Replace GPT-2 learned positional embeddings with Rotary Position Embeddings (RoPE).

#### GPT-2

```python
x = token_embedding + position_embedding
```

#### RoPE

```python
q = apply_rope(q)
k = apply_rope(k)
```

Topics:

* Sinusoidal Encoding
* Rotary Position Embedding
* Relative Position Awareness
* Long Context Extrapolation

Expected Outcome:

* Understand why modern LLMs abandoned learned position embeddings.
* Compare GPT-2 Position Embedding vs RoPE.

---

### Phase 3 — Flash Attention

Replace naive attention implementation.

#### Original

```python
att = q @ k.transpose(-2, -1)
att = softmax(att)
y = att @ v
```

#### Flash Attention

```python
F.scaled_dot_product_attention(...)
```

Topics:

* Attention Complexity
* GPU Memory Bottleneck
* Attention Tiling
* Flash Attention V1/V2

Benchmark:

* Memory Usage
* Throughput
* Training Speed

---

### Phase 4 — KV Cache

Implement autoregressive inference optimization.

#### Without KV Cache

```text
Generate token 100

Recompute:
1~99
```

#### With KV Cache

```text
Store previous K/V

Only compute:
token 100
```

Topics:

* Autoregressive Decoding
* Incremental Inference
* Cache Management

Benchmark:

* Latency
* Tokens/sec

---

### Phase 5 — Grouped Query Attention (GQA)

Implement modern LLM attention mechanism.

#### Multi Head Attention

```text
Q = 32 heads
K = 32 heads
V = 32 heads
```

#### Grouped Query Attention

```text
Q = 32 heads
K = 8 heads
V = 8 heads
```

Topics:

* MHA
* MQA
* GQA
* Memory Efficiency

Models Using GQA:

* LLaMA 3
* Qwen2
* DeepSeek

Benchmark:

* KV Cache Size
* Inference Throughput
* GPU Memory Usage

---

## Project Structure

```text
nanogpt-reimplementation/

├── tokenizer/
│   ├── char_tokenizer.py
│   └── bpe_tokenizer.py
│
├── models/
│   ├── embedding.py
│   ├── attention.py
│   ├── rope.py
│   ├── flash_attention.py
│   ├── kv_cache.py
│   ├── gqa.py
│   └── gpt.py
│
├── train/
│   ├── train.py
│   └── dataset.py
│
├── inference/
│   ├── generate.py
│   └── benchmark.py
│
├── experiments/
│   ├── rope_vs_gpt2.md
│   ├── flash_attention.md
│   ├── kv_cache.md
│   └── gqa.md
│
└── README.md
```

---

## Monthly Plan

### Week 1

* Understand nanoGPT codebase
* Reimplement GPT-2 architecture
* Train tiny Shakespeare

Deliverable:

```text
GPT-2 Reimplementation
```

---

### Week 2

* Implement RoPE
* Compare with GPT-2 Position Embedding
* Write experiment notes

Deliverable:

```text
GPT-2 + RoPE
```

---

### Week 3

* Implement Flash Attention
* Add performance benchmark
* Profile GPU memory

Deliverable:

```text
GPT-2 + RoPE + FlashAttention
```

---

### Week 4

* Implement KV Cache
* Implement GQA
* Write final report

Deliverable:

```text
Modern GPT Inference Stack
```

---

## Learning Goals

By the end of this project I should be able to explain:

* Why Q/K/V emerge during training
* Why attention uses QKᵀ
* Why RoPE replaced learned position embeddings
* Why Flash Attention is faster
* Why KV Cache reduces inference cost
* Why modern LLMs use GQA

---

## References

* nanoGPT — Andrej Karpathy
* Attention Is All You Need
* RoFormer
* FlashAttention
* LLaMA 2
* LLaMA 3
* Qwen2 Technical Report
* DeepSeek Technical Report
