# Neural Network Exercises

## Exercise 1: Multi-Layer Perceptron (MLP) Architecture (Beginner)

**Objective**: Build a flexible MLP class with configurable layers.

**Problem**: 
Create a modular MLP that can handle arbitrary depth and width.

**Tasks**:
1. Implement MLP class with configurable architecture:
   ```python
   class MLP:
       def __init__(self, layer_sizes, activations):
           # layer_sizes: [input_dim, hidden1, hidden2, ..., output_dim]
           # activations: list of activation functions per layer
   ```
2. Support multiple activation functions per layer
3. Implement forward pass
4. Implement backward pass
5. Add methods for parameter initialization
6. Create visualization of network architecture
7. Train on multi-class classification task

**Requirements**:
- Support variable number of hidden layers
- Handle different activation functions
- Implement proper weight initialization
- Include batch processing

---

## Exercise 2: Self-Attention Mechanism (Intermediate)

**Objective**: Implement the self-attention mechanism from scratch.

**Problem**: 
Build the core attention mechanism used in Transformers.

**Tasks**:
1. Implement scaled dot-product attention:
   ```
   Attention(Q, K, V) = softmax(QK^T / √d_k) × V
   ```
2. Create query, key, value projections:
   ```python
   Q = X × W_Q
   K = X × W_K
   V = X × W_V
   ```
3. Implement attention weights visualization
4. Add attention masking for padding
5. Implement causal masking for autoregressive models
6. Test on sequence data
7. Analyze attention patterns

**Key Components**:
- Query, Key, Value matrices
- Scaling factor √d_k
- Softmax for attention weights
- Masking mechanisms

---

## Exercise 3: Multi-Head Attention (Intermediate)

**Objective**: Extend self-attention to multi-head attention.

**Problem**: 
Implement multi-head attention as used in Transformer models.

**Tasks**:
1. Implement multi-head attention:
   ```
   MultiHead(Q,K,V) = Concat(head_1,...,head_h) × W_O
   where head_i = Attention(Q×W_Q^i, K×W_K^i, V×W_V^i)
   ```
2. Split input into multiple heads
3. Apply attention independently per head
4. Concatenate outputs from all heads
5. Apply output projection
6. Compare single-head vs multi-head performance
7. Visualize attention from different heads

**Architecture**:
- Number of heads: h = 8
- Model dimension: d_model = 512
- Per-head dimension: d_k = d_model / h = 64

---

## Exercise 4: Positional Encoding (Intermediate)

**Objective**: Implement positional encodings for sequence models.

**Problem**: 
Add positional information to token embeddings.

**Tasks**:
1. Implement sinusoidal positional encoding:
   ```
   PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
   ```
2. Implement learned positional embeddings
3. Compare fixed vs learned positional encodings
4. Visualize positional encoding patterns
5. Test on sequence modeling task
6. Analyze impact on long sequences

**Key Questions**:
- Why use sinusoidal functions?
- How do positional encodings help the model?
- Fixed vs learned: which is better?

---

## Exercise 5: Layer Normalization (Intermediate)

**Objective**: Implement layer normalization for stabilizing deep networks.

**Problem**: 
Add layer normalization to a deep neural network.

**Tasks**:
1. Implement layer normalization:
   ```
   μ = (1/H) × Σx_i  (mean across features)
   σ² = (1/H) × Σ(x_i - μ)²  (variance across features)
   x_norm = (x - μ) / √(σ² + ε)
   y = γ × x_norm + β  (scale and shift)
   ```
2. Compare with batch normalization
3. Implement for different layer types (linear, attention)
4. Add pre-norm vs post-norm configurations
5. Train models with and without layer norm
6. Analyze training stability

**Comparison**:
- Batch Norm: normalizes across batch dimension
- Layer Norm: normalizes across feature dimension
- Layer Norm is better for sequence models

---

## Exercise 6: Feedforward Network (FFN) Block (Intermediate)

**Objective**: Implement the position-wise feedforward network used in Transformers.

**Problem**: 
Build the FFN component with proper activations and dimensions.

**Tasks**:
1. Implement FFN with two linear layers:
   ```
   FFN(x) = W_2 × activation(W_1 × x + b_1) + b_2
   ```
2. Use different activations:
   - ReLU
   - GELU: 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
   - SwiGLU
3. Implement expansion factor (typically 4x):
   ```
   d_ff = 4 × d_model
   ```
4. Add dropout for regularization
5. Compare different activation functions
6. Analyze parameter count vs performance

**Standard Configuration**:
- Input dimension: d_model = 512
- Hidden dimension: d_ff = 2048
- Activation: ReLU or GELU
- Dropout: 0.1

---

## Exercise 7: Residual Connections (Advanced)

**Objective**: Implement residual connections and understand their importance.

**Problem**: 
Build a deep network with skip connections.

**Tasks**:
1. Implement residual connection:
   ```
   output = layer_norm(x + sublayer(x))
   ```
2. Compare architectures:
   - Post-norm: norm(x + F(x))
   - Pre-norm: x + F(norm(x))
3. Build deep network (12+ layers) with residuals
4. Compare with network without residuals
5. Analyze gradient flow through residual connections
6. Visualize gradient magnitudes at different depths

**Key Insights**:
- Residual connections enable training very deep networks
- Pre-norm is more stable for deep networks
- Gradients flow more easily through skip connections

---

## Exercise 8: Encoder Block (Advanced)

**Objective**: Implement a complete Transformer encoder block.

**Problem**: 
Build an encoder block combining attention, FFN, and normalization.

**Tasks**:
1. Implement encoder block structure:
   ```
   x = x + MultiHeadAttention(LayerNorm(x))
   x = x + FFN(LayerNorm(x))
   ```
2. Add dropout after each sublayer
3. Stack multiple encoder blocks (N=6)
4. Implement forward pass
5. Add proper masking for padding
6. Train on text classification
7. Visualize attention patterns across layers

**Architecture**:
```
Input → Embedding → [Encoder Block] × N → Output
where Encoder Block = Multi-Head Attention → FFN
```

---

## Exercise 9: Decoder Block with Cross-Attention (Advanced)

**Objective**: Implement a Transformer decoder block with cross-attention.

**Problem**: 
Build decoder for sequence-to-sequence tasks.

**Tasks**:
1. Implement decoder block:
   ```
   x = x + MaskedSelfAttention(LayerNorm(x))
   x = x + CrossAttention(LayerNorm(x), encoder_output)
   x = x + FFN(LayerNorm(x))
   ```
2. Implement causal masking for self-attention
3. Implement cross-attention to encoder
4. Stack multiple decoder blocks
5. Add output projection and softmax
6. Train on sequence-to-sequence task
7. Implement beam search for decoding

**Key Components**:
- Masked self-attention (causal)
- Cross-attention to encoder
- Position-wise FFN
- Output projection

---

## Exercise 10: Attention Variants (Advanced)

**Objective**: Implement and compare different attention mechanisms.

**Problem**: 
Explore various attention designs beyond standard scaled dot-product.

**Tasks**:
1. Implement additive (Bahdanau) attention:
   ```
   score(q, k) = v^T × tanh(W_q × q + W_k × k)
   ```
2. Implement multiplicative (Luong) attention:
   ```
   score(q, k) = q^T × W × k
   ```
3. Implement local attention (with window)
4. Implement sparse attention patterns
5. Compare computational complexity:
   - Standard: O(n²)
   - Local: O(n × w) where w is window size
   - Sparse: O(n × √n)
6. Evaluate on long sequences

**Complexity Analysis**:
- Memory: O(n²) for full attention
- Time: O(n² × d) for full attention
- Trade-offs between attention patterns

---

## Exercise 11: Gated Linear Units (GLU) (Intermediate)

**Objective**: Implement gated activation functions for neural networks.

**Problem**: 
Compare GLU variants in feedforward networks.

**Tasks**:
1. Implement GLU:
   ```
   GLU(x) = (x × W_1 + b_1) ⊗ σ(x × W_2 + b_2)
   ```
2. Implement variants:
   - GLU: with sigmoid gate
   - GeLU: Gaussian Error Linear Unit
   - SwiGLU: Swish-Gated Linear Unit
3. Replace FFN activation with GLU variants
4. Compare training dynamics
5. Measure performance differences
6. Analyze computational cost

**Comparison Metrics**:
- Training speed
- Final performance
- Parameter count
- Computational cost

---

## Exercise 12: Attention Pooling (Intermediate)

**Objective**: Implement different pooling strategies for sequence representations.

**Problem**: 
Convert variable-length sequences to fixed-size representations.

**Tasks**:
1. Implement pooling methods:
   - Max pooling
   - Mean pooling
   - Attention pooling (weighted by attention)
   - CLS token (learnable token)
2. Compare pooling strategies on classification task
3. Analyze which information is preserved
4. Visualize attention weights for pooling
5. Test on sequences of different lengths

**Attention Pooling**:
```
α = softmax(w^T × tanh(W × H))
representation = Σ α_i × h_i
```

---

## Bonus Exercise: Efficient Attention (Advanced)

**Objective**: Implement memory-efficient attention mechanisms.

**Problem**: 
Reduce memory complexity of attention for long sequences.

**Tasks**:
1. Implement linear attention:
   ```
   Instead of: softmax(QK^T)V
   Use: Q × (K^T × V) with feature maps
   ```
2. Implement flash attention (tiled computation)
3. Implement memory-efficient attention with checkpointing
4. Compare memory usage:
   - Standard attention
   - Linear attention
   - Flash attention
5. Benchmark on very long sequences (>1024 tokens)
6. Analyze accuracy vs efficiency trade-offs

**Key Optimization**:
- Recompute intermediate values instead of storing
- Use tiled computation for better cache utilization
- Linear complexity alternatives
