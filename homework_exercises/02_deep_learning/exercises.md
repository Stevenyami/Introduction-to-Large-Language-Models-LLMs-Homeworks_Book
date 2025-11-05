# Deep Learning Exercises

## Exercise 1: Implement Backpropagation from Scratch (Beginner)

**Objective**: Build a simple neural network and implement the backpropagation algorithm.

**Problem**: 
Create a 2-layer neural network (input → hidden → output) for binary classification.

**Tasks**:
1. Implement forward propagation:
   ```
   z1 = W1 × X + b1
   a1 = sigmoid(z1)
   z2 = W2 × a1 + b2
   a2 = sigmoid(z2)
   ```
2. Implement backward propagation:
   ```
   dz2 = a2 - y
   dW2 = (1/m) × dz2 × a1.T
   db2 = (1/m) × sum(dz2)
   dz1 = W2.T × dz2 * sigmoid'(z1)
   dW1 = (1/m) × dz1 × X.T
   db1 = (1/m) × sum(dz1)
   ```
3. Update parameters using gradient descent
4. Train for 10,000 iterations
5. Plot loss curve over iterations
6. Test on validation data

**Network Architecture**:
- Input layer: 2 neurons
- Hidden layer: 4 neurons
- Output layer: 1 neuron
- Activation: Sigmoid

---

## Exercise 2: Activation Functions Comparison (Beginner)

**Objective**: Implement and compare different activation functions.

**Problem**: 
Train the same neural network architecture with different activation functions.

**Tasks**:
1. Implement activation functions:
   - Sigmoid: σ(x) = 1 / (1 + e⁻ˣ)
   - Tanh: tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)
   - ReLU: max(0, x)
   - Leaky ReLU: max(αx, x) where α = 0.01
   - Softmax: eˣⁱ / Σeˣʲ
2. Implement their derivatives for backpropagation
3. Train models with each activation function
4. Compare:
   - Training speed (iterations to convergence)
   - Final accuracy
   - Gradient flow (check for vanishing gradients)
5. Visualize activation functions and their derivatives

**Analysis Questions**:
- Why does ReLU train faster than sigmoid?
- When would you use tanh vs ReLU?
- What is the vanishing gradient problem?

---

## Exercise 3: Mini-Batch Gradient Descent (Intermediate)

**Objective**: Implement and compare different gradient descent variants.

**Problem**: 
Train a neural network using batch, mini-batch, and stochastic gradient descent.

**Tasks**:
1. Implement batch gradient descent (entire dataset)
2. Implement mini-batch gradient descent (batch size = 32)
3. Implement stochastic gradient descent (batch size = 1)
4. For each method:
   - Track loss per epoch
   - Measure training time
   - Calculate final test accuracy
5. Plot loss curves for comparison
6. Analyze trade-offs:
   - Convergence speed
   - Memory usage
   - Noise in gradient estimates

**Comparison Metrics**:
- Time per epoch
- Number of epochs to convergence
- Final test accuracy
- Gradient noise level

---

## Exercise 4: Dropout Regularization (Intermediate)

**Objective**: Implement dropout to prevent overfitting in neural networks.

**Problem**: 
Train a deep network on a small dataset with and without dropout.

**Tasks**:
1. Implement dropout layer:
   ```python
   def dropout(X, keep_prob):
       mask = (np.random.rand(*X.shape) < keep_prob)
       return X * mask / keep_prob
   ```
2. Train two models:
   - Model A: Without dropout
   - Model B: With dropout (keep_prob = 0.5)
3. Plot training vs validation loss for both models
4. Compare test accuracy
5. Implement inverted dropout for efficient inference
6. Experiment with different dropout rates (0.2, 0.5, 0.8)

**Expected Observations**:
- Training accuracy: Model A > Model B
- Validation accuracy: Model B > Model A (less overfitting)
- Generalization gap reduction with dropout

---

## Exercise 5: Batch Normalization (Intermediate)

**Objective**: Implement batch normalization to stabilize training.

**Problem**: 
Add batch normalization layers to a deep neural network.

**Tasks**:
1. Implement batch normalization:
   ```
   μ = (1/m) × Σx_i
   σ² = (1/m) × Σ(x_i - μ)²
   x_norm = (x - μ) / √(σ² + ε)
   y = γ × x_norm + β
   ```
2. Train deep networks (5-10 layers):
   - Model A: Without batch normalization
   - Model B: With batch normalization
3. Compare:
   - Training stability
   - Convergence speed
   - Sensitivity to learning rate
4. Implement different batch norm modes (train vs eval)
5. Analyze internal covariate shift

**Key Insights**:
- Batch norm allows higher learning rates
- Reduces sensitivity to initialization
- Acts as regularization

---

## Exercise 6: Optimization Algorithms (Advanced)

**Objective**: Implement and compare advanced optimization algorithms.

**Problem**: 
Train neural networks using different optimizers.

**Tasks**:
1. Implement SGD with momentum:
   ```
   v = β × v + (1-β) × dW
   W = W - α × v
   ```
2. Implement RMSprop:
   ```
   s = β × s + (1-β) × dW²
   W = W - α × dW / √(s + ε)
   ```
3. Implement Adam optimizer:
   ```
   m = β₁ × m + (1-β₁) × dW
   v = β₂ × v + (1-β₂) × dW²
   m_corrected = m / (1 - β₁ᵗ)
   v_corrected = v / (1 - β₂ᵗ)
   W = W - α × m_corrected / √(v_corrected + ε)
   ```
4. Compare all optimizers on the same task
5. Tune hyperparameters for each
6. Plot convergence curves

**Hyperparameters**:
- Momentum: β = 0.9
- RMSprop: β = 0.999, ε = 1e-8
- Adam: β₁ = 0.9, β₂ = 0.999, ε = 1e-8

---

## Exercise 7: Convolutional Neural Networks (Advanced)

**Objective**: Implement a CNN for text classification.

**Problem**: 
Build a CNN to classify text documents by applying convolutions over word embeddings.

**Tasks**:
1. Implement 1D convolution operation:
   ```
   output[i] = Σ (input[i:i+k] * kernel)
   ```
2. Build CNN architecture:
   - Embedding layer (vocab_size → embedding_dim)
   - Conv1D layer (filters=100, kernel_size=3)
   - MaxPooling layer
   - Flatten layer
   - Dense layer (output)
3. Implement multiple parallel convolutions with different kernel sizes (3, 4, 5)
4. Concatenate outputs from parallel convolutions
5. Train on text classification task
6. Visualize learned filters

**Architecture** (Kim CNN for Text):
```
Input (text) → Embedding → [Conv3, Conv4, Conv5] → MaxPool → Concat → Dense → Output
```

---

## Exercise 8: Recurrent Neural Networks (Advanced)

**Objective**: Implement an RNN for sequence modeling.

**Problem**: 
Build an RNN to predict the next word in a sequence.

**Tasks**:
1. Implement basic RNN cell:
   ```
   h_t = tanh(W_hh × h_{t-1} + W_xh × x_t + b_h)
   y_t = W_hy × h_t + b_y
   ```
2. Implement Backpropagation Through Time (BPTT)
3. Train on language modeling task
4. Generate text using the trained model
5. Analyze gradient flow (check for vanishing gradients)
6. Implement gradient clipping to prevent exploding gradients

**Key Challenges**:
- Vanishing gradient problem
- Exploding gradient problem
- Long-term dependencies

---

## Exercise 9: LSTM Networks (Advanced)

**Objective**: Implement LSTM to handle long-term dependencies.

**Problem**: 
Replace RNN with LSTM for improved sequence modeling.

**Tasks**:
1. Implement LSTM cell with gates:
   ```
   f_t = σ(W_f × [h_{t-1}, x_t] + b_f)  # Forget gate
   i_t = σ(W_i × [h_{t-1}, x_t] + b_i)  # Input gate
   C̃_t = tanh(W_C × [h_{t-1}, x_t] + b_C)  # Candidate
   C_t = f_t * C_{t-1} + i_t * C̃_t  # Cell state
   o_t = σ(W_o × [h_{t-1}, x_t] + b_o)  # Output gate
   h_t = o_t * tanh(C_t)  # Hidden state
   ```
2. Implement forward and backward passes
3. Compare LSTM vs basic RNN performance
4. Train on long sequences (100+ tokens)
5. Visualize gate activations
6. Analyze which gates are most active for different inputs

**Comparison Metrics**:
- Perplexity on validation set
- Ability to capture long-range dependencies
- Training stability

---

## Exercise 10: Weight Initialization (Intermediate)

**Objective**: Understand the impact of weight initialization on training.

**Problem**: 
Compare different initialization strategies for deep networks.

**Tasks**:
1. Implement initialization methods:
   - Zero initialization
   - Random initialization (small values)
   - Xavier/Glorot initialization: W ~ N(0, 2/(n_in + n_out))
   - He initialization: W ~ N(0, 2/n_in)
2. Train identical networks with different initializations
3. Track:
   - Loss convergence
   - Gradient magnitudes
   - Activation distributions
4. Visualize gradient flow through layers
5. Analyze when each initialization works best

**Key Questions**:
- Why is zero initialization problematic?
- When to use Xavier vs He initialization?
- How does initialization interact with activation functions?

---

## Bonus Exercise: Residual Networks (Advanced)

**Objective**: Implement residual connections for very deep networks.

**Problem**: 
Build a deep network (20+ layers) with skip connections.

**Tasks**:
1. Implement residual block:
   ```
   F(x) = H(x) - x  (residual mapping)
   H(x) = F(x) + x  (skip connection)
   ```
2. Stack multiple residual blocks
3. Compare with plain deep network:
   - Training loss
   - Gradient flow
   - Final accuracy
4. Experiment with different skip connection patterns
5. Visualize gradient magnitudes at different depths

**Expected Results**:
- Residual networks train deeper networks more easily
- Better gradient flow through skip connections
- Improved performance on complex tasks
