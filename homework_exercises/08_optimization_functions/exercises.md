# Optimization Functions Exercises

## Exercise 1: Gradient Descent Variants Implementation (Beginner)

**Objective**: Implement and compare basic gradient descent algorithms.

**Problem**: Optimize a simple function using different GD variants.

**Tasks**:
1. Implement Batch Gradient Descent
2. Implement Stochastic Gradient Descent (SGD)
3. Implement Mini-batch Gradient Descent
4. Implement SGD with Momentum:
   ```
   v_t = β × v_{t-1} + (1-β) × g_t
   θ_t = θ_{t-1} - α × v_t
   ```
5. Implement Nesterov Accelerated Gradient
6. Compare convergence on test function
7. Visualize optimization trajectories

**Test Function**: Rosenbrock function or simple neural network

---

## Exercise 2: Adam Optimizer from Scratch (Intermediate)

**Objective**: Implement the Adam optimization algorithm.

**Problem**: Build Adam optimizer with all features.

**Tasks**:
1. Implement bias-corrected moment estimates:
   ```
   m_t = β₁ × m_{t-1} + (1-β₁) × g_t
   v_t = β₂ × v_{t-1} + (1-β₂) × g_t²
   m̂_t = m_t / (1 - β₁^t)
   v̂_t = v_t / (1 - β₂^t)
   θ_t = θ_{t-1} - α × m̂_t / (√v̂_t + ε)
   ```
2. Add weight decay (AdamW variant)
3. Implement learning rate scheduling
4. Compare with SGD and RMSprop
5. Test on neural network training
6. Analyze convergence properties

**Hyperparameters**:
- β₁ = 0.9, β₂ = 0.999, ε = 1e-8

---

## Exercise 3: Loss Function Design (Intermediate)

**Objective**: Implement and compare various loss functions.

**Problem**: Choose appropriate loss for different tasks.

**Tasks**:
1. Implement classification losses:
   - Cross-entropy
   - Focal loss (for imbalanced data)
   - Label smoothing cross-entropy
2. Implement regression losses:
   - MSE, MAE, Huber loss
3. Implement ranking losses:
   - Contrastive loss
   - Triplet loss
4. Compare on appropriate tasks
5. Analyze gradient properties
6. Handle numerical stability

**Focal Loss**:
```
FL(p_t) = -α_t × (1-p_t)^γ × log(p_t)
```

---

## Exercise 4: Gradient Clipping (Intermediate)

**Objective**: Implement gradient clipping to prevent exploding gradients.

**Problem**: Stabilize RNN training with gradient clipping.

**Tasks**:
1. Implement gradient clipping by value:
   ```python
   g = clip(g, -threshold, threshold)
   ```
2. Implement gradient clipping by norm:
   ```python
   g = g × min(1, threshold / ||g||)
   ```
3. Train RNN with and without clipping
4. Monitor gradient magnitudes
5. Find optimal clipping threshold
6. Analyze impact on training stability

---

## Exercise 5: Learning Rate Warmup and Decay (Intermediate)

**Objective**: Implement learning rate schedules for stable training.

**Problem**: Combine warmup with decay for optimal training.

**Tasks**:
1. Implement warmup:
   ```python
   if step < warmup_steps:
       lr = base_lr * (step / warmup_steps)
   ```
2. Implement decay strategies:
   - Linear decay
   - Cosine decay
   - Exponential decay
3. Implement Transformer learning rate schedule:
   ```
   lr = d_model^(-0.5) × min(step^(-0.5), step × warmup^(-1.5))
   ```
4. Compare schedules on transformer training
5. Plot learning rate curves
6. Analyze impact on convergence

---

## Exercise 6: Second-Order Optimization (Advanced)

**Objective**: Implement and understand second-order optimization methods.

**Problem**: Use Hessian information for faster convergence.

**Tasks**:
1. Implement L-BFGS:
   - Maintain history of gradients
   - Approximate Hessian inverse
   - Compute search direction
2. Implement Newton's method for small networks
3. Compare with first-order methods:
   - Convergence speed
   - Computational cost
   - Memory usage
4. Test on convex problems
5. Analyze when second-order helps

**Challenge**: Second-order methods are expensive for large models

---

## Exercise 7: Gradient Accumulation (Intermediate)

**Objective**: Simulate large batch sizes with gradient accumulation.

**Problem**: Train with effective batch size larger than memory allows.

**Tasks**:
1. Implement gradient accumulation:
   ```python
   for i, batch in enumerate(data):
       loss = model(batch)
       loss = loss / accumulation_steps
       loss.backward()
       if (i+1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```
2. Compare with actual large batch training
3. Adjust learning rate appropriately
4. Measure memory savings
5. Analyze training dynamics
6. Test different accumulation factors

---

## Exercise 8: Mixed Precision Training (Advanced)

**Objective**: Implement FP16 training with loss scaling.

**Problem**: Speed up training while maintaining numerical stability.

**Tasks**:
1. Convert model to FP16
2. Implement loss scaling:
   ```python
   scaled_loss = loss * scale_factor
   scaled_loss.backward()
   gradients = gradients / scale_factor
   ```
3. Implement dynamic loss scaling
4. Handle gradient overflow/underflow
5. Measure speedup and memory savings
6. Compare FP32 vs FP16 accuracy
7. Use PyTorch AMP or implement manually

**Benefits**: 2-3x speedup, 50% memory reduction

---

## Exercise 9: Optimizer Comparison Study (Advanced)

**Objective**: Comprehensively compare optimization algorithms.

**Problem**: Find best optimizer for your task.

**Tasks**:
1. Implement/use optimizers:
   - SGD, SGD+Momentum, SGD+Nesterov
   - Adam, AdamW, RAdam
   - AdaGrad, RMSprop
   - LAMB (for large batch)
2. Compare on multiple tasks:
   - Image classification
   - Text classification
   - Language modeling
3. Measure:
   - Convergence speed
   - Final performance
   - Stability
   - Memory usage
4. Create recommendation guide
5. Analyze task-specific preferences

---

## Exercise 10: Custom Loss Function (Advanced)

**Objective**: Design custom loss function for specific task.

**Problem**: Create task-specific loss combining multiple objectives.

**Tasks**:
1. Identify task requirements
2. Design composite loss:
   ```
   L_total = α × L_task + β × L_auxiliary + γ × L_regularization
   ```
3. Implement custom backward pass
4. Balance loss components
5. Compare with standard losses
6. Analyze gradient flow
7. Tune loss weights

**Example**: Multi-task learning loss, contrastive loss with hard negative mining

---

## Exercise 11: Gradient Checkpointing (Advanced)

**Objective**: Implement gradient checkpointing to save memory.

**Problem**: Train very deep networks within memory constraints.

**Tasks**:
1. Understand memory-computation trade-off
2. Implement checkpointing:
   - Save activations at checkpoints
   - Recompute during backward pass
3. Integrate with training loop
4. Measure memory savings
5. Measure time overhead
6. Optimize checkpoint placement
7. Test on very deep networks

**Memory Savings**: O(√n) instead of O(n) for n layers

---

## Exercise 12: Optimization Debugging (Advanced)

**Objective**: Diagnose and fix optimization problems.

**Problem**: Identify and resolve training issues.

**Tasks**:
1. Implement diagnostic tools:
   - Gradient norm tracking
   - Loss landscape visualization
   - Learning rate finder
   - Activation statistics
2. Diagnose common problems:
   - Vanishing gradients
   - Exploding gradients
   - Saddle points
   - Poor initialization
3. Apply fixes:
   - Adjust learning rate
   - Change optimizer
   - Add/adjust regularization
   - Fix initialization
4. Create debugging checklist
5. Automate problem detection
