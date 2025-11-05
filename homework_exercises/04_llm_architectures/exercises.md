# LLM Architecture Exercises

## Exercise 1: Transformer Architecture from Scratch (Advanced)

**Objective**: Implement the complete Transformer architecture from the "Attention is All You Need" paper.

**Problem**: 
Build a full encoder-decoder Transformer for machine translation.

**Tasks**:
1. Implement complete Transformer components:
   - Input embedding layer
   - Positional encoding
   - Encoder stack (N=6 layers)
   - Decoder stack (N=6 layers)
   - Output projection
2. Configure architecture:
   ```
   d_model = 512 (model dimension)
   d_ff = 2048 (FFN dimension)
   h = 8 (number of heads)
   N = 6 (number of layers)
   dropout = 0.1
   ```
3. Implement training loop with teacher forcing
4. Add label smoothing (ε = 0.1)
5. Implement learning rate scheduling:
   ```
   lr = d_model^(-0.5) × min(step^(-0.5), step × warmup_steps^(-1.5))
   ```
6. Train on translation task
7. Implement inference with beam search

**Architecture Overview**:
```
Source → Encoder → Encoder Output
                        ↓
Target → Decoder → Output Probabilities
```

---

## Exercise 2: GPT (Decoder-Only) Architecture (Advanced)

**Objective**: Implement a GPT-style autoregressive language model.

**Problem**: 
Build a decoder-only Transformer for causal language modeling.

**Tasks**:
1. Implement GPT architecture:
   - Token embeddings
   - Positional embeddings (learned)
   - Decoder blocks with causal masking
   - Language modeling head
2. Key differences from standard Transformer:
   - No encoder
   - No cross-attention
   - Causal self-attention only
3. Implement causal attention mask:
   ```python
   mask = torch.tril(torch.ones(seq_len, seq_len))
   ```
4. Train with next-token prediction objective
5. Implement text generation:
   - Greedy decoding
   - Top-k sampling
   - Top-p (nucleus) sampling
   - Temperature scaling
6. Analyze perplexity on validation set

**Configuration** (GPT-2 Small):
```
n_layers = 12
n_heads = 12
d_model = 768
d_ff = 3072
vocab_size = 50257
max_seq_len = 1024
```

---

## Exercise 3: BERT (Encoder-Only) Architecture (Advanced)

**Objective**: Implement BERT for masked language modeling and next sentence prediction.

**Problem**: 
Build a bidirectional encoder for pre-training with MLM objective.

**Tasks**:
1. Implement BERT architecture:
   - Token + Segment + Position embeddings
   - Bidirectional encoder (no causal masking)
   - MLM head
   - NSP head
2. Implement masked language modeling:
   - Randomly mask 15% of tokens
   - 80% → [MASK]
   - 10% → random token
   - 10% → keep original
3. Implement next sentence prediction
4. Add special tokens: [CLS], [SEP], [MASK]
5. Train with both objectives:
   ```
   Loss = Loss_MLM + Loss_NSP
   ```
6. Fine-tune on downstream task (classification)

**Configuration** (BERT-Base):
```
n_layers = 12
n_heads = 12
d_model = 768
d_ff = 3072
vocab_size = 30522
max_seq_len = 512
```

---

## Exercise 4: T5 (Text-to-Text) Architecture (Advanced)

**Objective**: Implement T5's unified text-to-text framework.

**Problem**: 
Build an encoder-decoder model that frames all NLP tasks as text generation.

**Tasks**:
1. Implement T5 architecture:
   - Relative positional bias instead of absolute positional encoding
   - Simplified layer normalization (no bias, no learnable scale)
   - Dense-ReLU-Dense FFN with dropout
2. Implement relative position bias:
   ```python
   bias = embeddings[relative_position[i, j]]
   attention_scores += bias
   ```
3. Frame tasks as text-to-text:
   - Translation: "translate English to German: {text}"
   - Summarization: "summarize: {text}"
   - Classification: "classify: {text}"
4. Train with span corruption objective
5. Implement different prediction strategies
6. Fine-tune on multiple tasks

**Key Innovations**:
- Relative positional bias
- Simplified architecture
- Unified text-to-text interface

---

## Exercise 5: Rotary Position Embeddings (RoPE) (Intermediate)

**Objective**: Implement Rotary Position Embeddings used in modern LLMs.

**Problem**: 
Replace standard positional encodings with rotation-based encodings.

**Tasks**:
1. Implement RoPE:
   ```python
   def apply_rotary_pos_emb(q, k, pos):
       # Rotate query and key by position-dependent angle
       theta = 10000 ^ (-2i/d) for i in range(d/2)
       cos_pos = cos(pos * theta)
       sin_pos = sin(pos * theta)
       q_rot = rotate(q, cos_pos, sin_pos)
       k_rot = rotate(k, cos_pos, sin_pos)
       return q_rot, k_rot
   ```
2. Understand rotation in complex space
3. Compare with sinusoidal positional encoding
4. Test on long sequences
5. Analyze extrapolation to longer sequences
6. Visualize position-dependent rotations

**Advantages**:
- Better extrapolation to longer sequences
- Relative position encoding naturally
- Used in LLaMA, GPT-NeoX, PaLM

---

## Exercise 6: Grouped Query Attention (GQA) (Advanced)

**Objective**: Implement Grouped Query Attention for efficient inference.

**Problem**: 
Reduce KV cache size while maintaining performance.

**Tasks**:
1. Understand attention variants:
   - Multi-Head Attention (MHA): h query, h key, h value heads
   - Multi-Query Attention (MQA): h query, 1 key, 1 value head
   - Grouped Query Attention (GQA): h query, g key, g value heads
2. Implement GQA:
   ```python
   # Group queries, share KV across groups
   n_groups = 4  # h/n_groups queries per KV head
   ```
3. Compare memory usage:
   - MHA: O(h × d × n) for KV cache
   - GQA: O(g × d × n) for KV cache
4. Measure inference speed
5. Compare quality vs efficiency trade-off
6. Analyze different grouping strategies

**Used in**: LLaMA-2, Mistral

---

## Exercise 7: Flash Attention Integration (Advanced)

**Objective**: Integrate Flash Attention for memory-efficient training.

**Problem**: 
Replace standard attention with Flash Attention implementation.

**Tasks**:
1. Understand Flash Attention algorithm:
   - Tiled computation
   - Online softmax
   - Recomputation in backward pass
2. Integrate Flash Attention library:
   ```python
   from flash_attn import flash_attn_func
   ```
3. Compare memory usage:
   - Standard attention: O(n² × d)
   - Flash Attention: O(n × d)
4. Benchmark training speed
5. Test on long sequences (4k, 8k, 16k tokens)
6. Analyze throughput improvements

**Benefits**:
- 3-4x speedup
- Lower memory usage
- Enables longer context windows

---

## Exercise 8: Mixture of Experts (MoE) (Advanced)

**Objective**: Implement sparse Mixture of Experts layer.

**Problem**: 
Scale model capacity with sparse computation.

**Tasks**:
1. Implement MoE layer:
   ```python
   # Replace FFN with MoE
   router_logits = router(x)
   expert_idx = top_k(router_logits, k=2)
   output = sum(experts[i](x) * gate_weight[i] for i in expert_idx)
   ```
2. Implement gating network (router)
3. Add load balancing loss:
   ```python
   load_balance_loss = auxiliary_loss(router_logits)
   total_loss = task_loss + α × load_balance_loss
   ```
4. Implement top-k routing (k=2)
5. Handle expert parallelism
6. Compare with dense model:
   - Parameters vs active parameters
   - Training speed
   - Final performance

**Key Concepts**:
- Sparse activation
- Load balancing
- Expert capacity

**Used in**: Switch Transformer, GLaM, GPT-4 (rumored)

---

## Exercise 9: Sliding Window Attention (Intermediate)

**Objective**: Implement local attention with sliding windows.

**Problem**: 
Process long sequences efficiently with local attention.

**Tasks**:
1. Implement sliding window attention:
   ```python
   # Each token attends to w tokens before and after
   window_size = 512
   attention_mask = create_sliding_window_mask(window_size)
   ```
2. Combine with global attention tokens
3. Compare with full attention:
   - Complexity: O(n × w) vs O(n²)
   - Memory: O(n × w) vs O(n²)
4. Test on long documents
5. Analyze information flow through layers
6. Implement for autoregressive models

**Used in**: Longformer, BigBird, Mistral

---

## Exercise 10: KV Cache Optimization (Advanced)

**Objective**: Implement efficient KV caching for faster inference.

**Problem**: 
Optimize autoregressive generation with KV cache.

**Tasks**:
1. Implement basic KV cache:
   ```python
   # Cache past keys and values
   past_kv = [(k_1, v_1), ..., (k_n, v_n)]
   # At step t, only compute new KV for position t
   k_new, v_new = compute_kv(x_t)
   k_full = concat(past_kv[layer][0], k_new)
   v_full = concat(past_kv[layer][1], v_new)
   ```
2. Measure memory usage with caching
3. Compare generation speed with/without cache
4. Implement cache quantization (int8 KV cache)
5. Add cache eviction strategies for long sequences
6. Analyze memory vs quality trade-offs

**Optimizations**:
- Multi-query attention for smaller cache
- Quantization (int8/int4)
- Page attention for dynamic batching

---

## Exercise 11: Alibi Positional Encoding (Intermediate)

**Objective**: Implement Attention with Linear Biases (ALiBi).

**Problem**: 
Replace positional encodings with attention biases.

**Tasks**:
1. Implement ALiBi:
   ```python
   # Add bias to attention scores based on distance
   bias = -m × |i - j|  where m is head-specific slope
   attention_scores += bias
   ```
2. Set head-specific slopes:
   ```python
   m_h = 2^(-8/h × h_idx) for h_idx in range(h)
   ```
3. Compare with other position encodings
4. Test extrapolation to longer sequences
5. Analyze attention patterns
6. Measure zero-shot performance on long sequences

**Advantages**:
- No position embeddings needed
- Better extrapolation
- Faster training (no embedding parameters)

**Used in**: BLOOM, MPT

---

## Exercise 12: Model Architecture Comparison (Advanced)

**Objective**: Compare different LLM architectural choices empirically.

**Problem**: 
Analyze trade-offs between different design decisions.

**Tasks**:
1. Implement variants:
   - Encoder-only (BERT-style)
   - Decoder-only (GPT-style)
   - Encoder-decoder (T5-style)
2. Compare on multiple tasks:
   - Classification
   - Generation
   - Question Answering
3. Measure:
   - Training efficiency (FLOPs)
   - Inference speed
   - Memory usage
   - Task performance
4. Analyze architectural trade-offs:
   - Bidirectional vs causal attention
   - Position encoding methods
   - Normalization placement (pre-norm vs post-norm)
5. Create decision guide for architecture selection

**Comparison Matrix**:
| Architecture | Training Speed | Inference Speed | Classification | Generation |
|--------------|----------------|-----------------|----------------|------------|
| Encoder-only | Fast | Fast | Excellent | Poor |
| Decoder-only | Fast | Fast | Good | Excellent |
| Enc-Dec | Slow | Medium | Good | Good |

---

## Bonus Exercise: Implement a Mini-LLM (Advanced)

**Objective**: Build a complete small-scale LLM from scratch.

**Problem**: 
Create a working GPT-style model and train it on a corpus.

**Tasks**:
1. Design architecture (small scale):
   ```
   n_layers = 6
   n_heads = 6
   d_model = 384
   d_ff = 1536
   vocab_size = 10000
   max_seq_len = 512
   ```
2. Implement full training pipeline:
   - Data loading and tokenization
   - Training loop with gradient accumulation
   - Learning rate scheduling
   - Checkpointing
3. Train on text corpus (e.g., WikiText)
4. Implement generation with sampling
5. Evaluate perplexity
6. Fine-tune on downstream task
7. Document all design choices

**Deliverables**:
- Complete model implementation
- Training scripts
- Evaluation results
- Generated text samples
- Performance analysis
