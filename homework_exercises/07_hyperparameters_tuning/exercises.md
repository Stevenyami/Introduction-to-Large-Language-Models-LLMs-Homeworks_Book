# Hyperparameters Tuning Exercises

## Exercise 1: Grid Search vs Random Search (Beginner)

**Objective**: Compare systematic grid search with random search.

**Problem**: Find optimal hyperparameters for a text classifier.

**Tasks**:
1. Define hyperparameter space:
   - Learning rate: [1e-5, 1e-4, 1e-3, 1e-2]
   - Batch size: [16, 32, 64, 128]
   - Dropout: [0.1, 0.2, 0.3, 0.5]
   - Hidden size: [128, 256, 512]
2. Implement grid search (exhaustive)
3. Implement random search (N samples)
4. Compare:
   - Time to find good solution
   - Best performance achieved
   - Coverage of search space
5. Visualize search trajectories
6. Analyze computational cost

---

## Exercise 2: Bayesian Optimization with Optuna (Intermediate)

**Objective**: Use Bayesian optimization for efficient hyperparameter search.

**Problem**: Optimize LLM fine-tuning hyperparameters.

**Tasks**:
1. Define objective function
2. Set up Optuna study
3. Implement TPE sampler
4. Use pruning for early stopping
5. Track optimization history
6. Analyze parameter importance
7. Visualize optimization surface

**Code Template**:
```python
import optuna

def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    # Train and return validation metric
    return validation_score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

---

## Exercise 3: Learning Rate Scheduling (Intermediate)

**Objective**: Implement and compare learning rate schedules.

**Problem**: Find optimal learning rate schedule for training.

**Tasks**:
1. Implement schedules:
   - Constant
   - Step decay
   - Exponential decay
   - Cosine annealing
   - Warmup + decay
   - OneCycleLR
2. Implement learning rate finder
3. Train models with each schedule
4. Plot learning curves
5. Compare convergence speed
6. Analyze final performance

**Learning Rate Finder**:
```python
# Start with very small LR, increase exponentially
# Plot loss vs LR
# Select LR where loss decreases fastest
```

---

## Exercise 4: Batch Size Selection (Intermediate)

**Objective**: Understand batch size impact on training.

**Problem**: Find optimal batch size balancing speed and performance.

**Tasks**:
1. Train with batch sizes: [8, 16, 32, 64, 128, 256]
2. Adjust learning rate with batch size:
   - Linear scaling rule: lr_new = lr_base × (batch_new / batch_base)
3. Measure:
   - Training throughput (samples/sec)
   - Memory usage
   - Time to convergence
   - Final accuracy
4. Implement gradient accumulation for large effective batch sizes
5. Analyze generalization gap

---

## Exercise 5: Regularization Hyperparameters (Advanced)

**Objective**: Tune regularization to prevent overfitting.

**Problem**: Find optimal regularization strategy.

**Tasks**:
1. Tune dropout rates per layer
2. Tune weight decay (L2 regularization)
3. Tune label smoothing
4. Combine multiple regularization techniques
5. Plot validation curves
6. Analyze overfitting indicators
7. Select final configuration

**Regularization Parameters**:
- Dropout: typically 0.1-0.5
- Weight decay: typically 0.01-0.1
- Label smoothing: typically 0.1

---

## Exercise 6: Architecture Hyperparameters (Advanced)

**Objective**: Optimize neural architecture hyperparameters.

**Problem**: Find optimal architecture for your task.

**Tasks**:
1. Tune:
   - Number of layers: [2, 4, 6, 8, 12]
   - Hidden dimension: [256, 512, 768, 1024]
   - Number of attention heads: [4, 8, 12, 16]
   - FFN dimension multiplier: [2, 4]
2. Balance model capacity and overfitting
3. Consider computational constraints
4. Use nested cross-validation
5. Analyze parameter count vs performance
6. Document final architecture choices

---

## Exercise 7: Warmup Steps Optimization (Intermediate)

**Objective**: Determine optimal warmup duration.

**Problem**: Stabilize training with learning rate warmup.

**Tasks**:
1. Implement linear warmup
2. Test warmup durations:
   - Percentage of total steps: [1%, 5%, 10%, 20%]
3. Measure training stability
4. Compare with no warmup
5. Analyze loss spikes
6. Select based on task and model size

**Warmup Schedule**:
```python
if step < warmup_steps:
    lr = max_lr * (step / warmup_steps)
else:
    lr = cosine_decay(step - warmup_steps)
```

---

## Exercise 8: Multi-Objective Hyperparameter Optimization (Advanced)

**Objective**: Optimize multiple objectives simultaneously.

**Problem**: Balance accuracy, speed, and model size.

**Tasks**:
1. Define multiple objectives:
   - Maximize: validation accuracy
   - Minimize: inference latency
   - Minimize: model parameters
2. Implement Pareto optimization
3. Use multi-objective optimizer (e.g., NSGA-II)
4. Visualize Pareto front
5. Select solution based on constraints
6. Analyze trade-offs

---

## Exercise 9: Transfer Learning Hyperparameters (Advanced)

**Objective**: Optimize hyperparameters for fine-tuning.

**Problem**: Find best settings for transfer learning.

**Tasks**:
1. Tune fine-tuning learning rate (typically 10x smaller)
2. Experiment with layer-wise learning rates
3. Tune number of epochs (avoid catastrophic forgetting)
4. Optimize warmup for fine-tuning
5. Compare discriminative vs non-discriminative fine-tuning
6. Implement gradual unfreezing
7. Evaluate on target domain

---

## Exercise 10: Automated Hyperparameter Tuning Pipeline (Advanced)

**Objective**: Build end-to-end AutoML pipeline.

**Problem**: Automate the entire hyperparameter search process.

**Tasks**:
1. Integrate multiple search strategies
2. Implement experiment tracking
3. Add automatic resource management
4. Create visualization dashboard
5. Support distributed search
6. Implement model selection criteria
7. Generate hyperparameter report

**Components**:
- Search strategy selection
- Experiment logging (W&B, MLflow)
- Resource scheduling
- Result analysis
- Final model selection
