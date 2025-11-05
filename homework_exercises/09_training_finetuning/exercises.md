# Training & Fine-Tuning Exercises

## Exercise 1: Pre-training Language Model (Advanced)

**Objective**: Pre-train a small language model from scratch.

**Problem**: Train a GPT-style model on text corpus.

**Tasks**:
1. Prepare training corpus (e.g., WikiText)
2. Implement tokenization (BPE or WordPiece)
3. Set up causal language modeling objective
4. Implement training loop:
   - Data loading with proper batching
   - Forward pass with teacher forcing
   - Backward pass and optimization
5. Implement learning rate scheduling
6. Add checkpointing and logging
7. Evaluate perplexity on validation set
8. Monitor training dynamics

**Configuration**:
- Model: GPT-2 Small (124M params)
- Objective: Next token prediction
- Training: 1-2 epochs on large corpus

---

## Exercise 2: Full Fine-Tuning on Classification (Intermediate)

**Objective**: Fine-tune pre-trained LLM on text classification.

**Problem**: Adapt BERT for sentiment analysis.

**Tasks**:
1. Load pre-trained BERT model
2. Add classification head
3. Prepare labeled dataset
4. Implement fine-tuning loop:
   - Lower learning rate (1e-5 to 5e-5)
   - Shorter training (3-5 epochs)
   - Task-specific loss
5. Monitor validation metrics
6. Implement early stopping
7. Save best model checkpoint
8. Evaluate on test set

**Best Practices**:
- Use smaller learning rate than pre-training
- Fine-tune all layers initially
- Monitor for overfitting

---

## Exercise 3: LoRA (Low-Rank Adaptation) (Advanced)

**Objective**: Implement parameter-efficient fine-tuning with LoRA.

**Problem**: Fine-tune large model with minimal trainable parameters.

**Tasks**:
1. Understand LoRA decomposition:
   ```
   W' = W + BA
   where B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), r << min(d,k)
   ```
2. Implement LoRA layer:
   ```python
   output = W @ x + (B @ A) @ x
   # Only train B and A, freeze W
   ```
3. Apply LoRA to attention weights (Q, K, V, O)
4. Set rank r (typically 4-16)
5. Compare with full fine-tuning:
   - Trainable parameters
   - Training speed
   - Final performance
6. Analyze rank selection impact

**Benefits**: 
- 0.1% trainable parameters
- Faster training
- Multiple adapters for one base model

---

## Exercise 4: Prefix Tuning (Advanced)

**Objective**: Implement prefix tuning for parameter-efficient adaptation.

**Problem**: Add learnable prefix tokens instead of fine-tuning all parameters.

**Tasks**:
1. Add trainable prefix embeddings:
   ```python
   prefix = trainable_embeddings[0:prefix_length]
   input_with_prefix = concat(prefix, input_embeddings)
   ```
2. Freeze base model parameters
3. Train only prefix parameters
4. Implement for each layer (prefix in KV)
5. Tune prefix length (10-100 tokens)
6. Compare with full fine-tuning
7. Test on multiple tasks

**Advantage**: Can switch tasks by swapping prefix

---

## Exercise 5: Adapter Layers (Advanced)

**Objective**: Add small adapter modules for efficient fine-tuning.

**Problem**: Insert trainable adapters between frozen layers.

**Tasks**:
1. Implement adapter module:
   ```python
   def adapter(x):
       h = down_project(x)  # d → r
       h = activation(h)
       h = up_project(h)    # r → d
       return x + h  # residual connection
   ```
2. Insert adapters in transformer:
   - After attention
   - After FFN
3. Freeze base model, train only adapters
4. Set bottleneck dimension (typically r = 64)
5. Compare efficiency metrics
6. Analyze adapter placement impact

**Benefits**: 2-4% additional parameters, competitive performance

---

## Exercise 6: Prompt Engineering for Few-Shot Learning (Intermediate)

**Objective**: Design effective prompts for in-context learning.

**Problem**: Use LLM without fine-tuning through clever prompting.

**Tasks**:
1. Design prompt templates:
   ```
   "Classify the sentiment:
   Text: 'I love this!' Sentiment: positive
   Text: 'This is terrible.' Sentiment: negative
   Text: '{input}' Sentiment:"
   ```
2. Implement few-shot prompting (0-shot, 1-shot, few-shot)
3. Test prompt variations:
   - Different phrasings
   - Different example orders
   - Different example selections
4. Implement prompt ensembling
5. Measure performance vs fine-tuning
6. Analyze cost-performance trade-off

---

## Exercise 7: Instruction Tuning (Advanced)

**Objective**: Fine-tune model to follow instructions.

**Problem**: Train model on diverse instruction-following tasks.

**Tasks**:
1. Prepare instruction dataset:
   ```
   Input: "Translate to French: Hello"
   Output: "Bonjour"
   ```
2. Format data uniformly
3. Implement instruction fine-tuning
4. Mix multiple task types
5. Use curriculum learning (easy → hard)
6. Evaluate on held-out instructions
7. Test zero-shot generalization

**Dataset Examples**: FLAN, Natural Instructions

---

## Exercise 8: Distributed Training with DDP (Advanced)

**Objective**: Implement data-parallel distributed training.

**Problem**: Speed up training across multiple GPUs.

**Tasks**:
1. Set up DistributedDataParallel:
   ```python
   model = DDP(model, device_ids=[local_rank])
   ```
2. Initialize process group
3. Implement distributed sampler
4. Synchronize gradients across processes
5. Aggregate metrics correctly
6. Handle checkpointing
7. Measure scaling efficiency

**Scaling**: Near-linear speedup with more GPUs

---

## Exercise 9: Mixed Precision Fine-Tuning (Intermediate)

**Objective**: Use FP16/BF16 for efficient fine-tuning.

**Problem**: Reduce memory usage and increase speed.

**Tasks**:
1. Enable automatic mixed precision:
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   ```
2. Wrap forward pass with autocast
3. Scale loss for backward pass
4. Compare FP32 vs FP16:
   - Training speed
   - Memory usage
   - Final accuracy
5. Handle numerical stability
6. Use BF16 if available (better than FP16)

---

## Exercise 10: Curriculum Learning (Advanced)

**Objective**: Train with progressively harder examples.

**Problem**: Improve training efficiency with curriculum.

**Tasks**:
1. Define difficulty metric:
   - Sequence length
   - Model uncertainty
   - Loss value
2. Sort training data by difficulty
3. Implement curriculum strategies:
   - Fixed schedule
   - Dynamic (performance-based)
   - Self-paced learning
4. Compare with random sampling
5. Analyze convergence speed
6. Measure final performance

---

## Exercise 11: Catastrophic Forgetting Analysis (Advanced)

**Objective**: Study and mitigate catastrophic forgetting.

**Problem**: Fine-tune without forgetting pre-trained knowledge.

**Tasks**:
1. Fine-tune model on task A
2. Measure performance on original tasks
3. Implement mitigation strategies:
   - Elastic Weight Consolidation (EWC)
   - Progressive Neural Networks
   - Replay buffer
4. Compare strategies:
   - Retention of original performance
   - Adaptation to new task
5. Analyze forgetting patterns
6. Find optimal balance

---

## Exercise 12: Model Compression for Deployment (Advanced)

**Objective**: Compress fine-tuned model for production.

**Problem**: Reduce model size while maintaining performance.

**Tasks**:
1. Implement quantization:
   - Post-training quantization (INT8)
   - Quantization-aware training
2. Implement knowledge distillation:
   ```
   L = α × L_task + (1-α) × KL(student, teacher)
   ```
3. Implement pruning:
   - Magnitude-based pruning
   - Structured pruning
4. Measure trade-offs:
   - Model size reduction
   - Inference speed
   - Accuracy degradation
5. Combine techniques
6. Deploy compressed model

**Goal**: 4x smaller, 2x faster with <1% accuracy drop
