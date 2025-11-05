# Metrics & Evaluation Protocol Exercises

## Exercise 1: Classification Metrics Implementation (Beginner)

**Objective**: Implement fundamental classification metrics from scratch.

**Problem**: Calculate precision, recall, F1, and accuracy without libraries.

**Tasks**:
1. Implement confusion matrix
2. Calculate metrics:
   ```
   Accuracy = (TP + TN) / (TP + TN + FP + FN)
   Precision = TP / (TP + FP)
   Recall = TP / (TP + FN)
   F1 = 2 × (Precision × Recall) / (Precision + Recall)
   ```
3. Extend to multi-class (macro/micro/weighted averaging)
4. Implement ROC curve and AUC
5. Plot precision-recall curve
6. Handle edge cases (division by zero)
7. Compare with sklearn implementation

---

## Exercise 2: Text Generation Metrics (Intermediate)

**Objective**: Implement metrics for evaluating generated text.

**Problem**: Measure quality of machine-generated text.

**Tasks**:
1. Implement BLEU score:
   ```
   BLEU = BP × exp(Σ w_n × log(p_n))
   where p_n is n-gram precision
   BP = brevity penalty
   ```
2. Implement ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L)
3. Implement METEOR (with synonyms and stemming)
4. Implement perplexity:
   ```
   PPL = exp(-1/N × Σ log P(w_i | context))
   ```
5. Compare metrics on translation/summarization
6. Analyze correlation with human judgments
7. Understand limitations of each metric

---

## Exercise 3: BERTScore Implementation (Advanced)

**Objective**: Implement semantic similarity-based evaluation.

**Problem**: Use contextual embeddings to measure text similarity.

**Tasks**:
1. Extract BERT embeddings for tokens
2. Compute cosine similarity matrix
3. Implement greedy matching:
   ```
   precision = (1/|ŷ|) × Σ max sim(y_i, ŷ)
   recall = (1/|y|) × Σ max sim(ŷ_j, y)
   F1 = 2 × P × R / (P + R)
   ```
4. Add IDF weighting
5. Test on paraphrase detection
6. Compare with BLEU/ROUGE
7. Analyze when BERTScore is better

---

## Exercise 4: Statistical Significance Testing (Intermediate)

**Objective**: Test if model improvements are statistically significant.

**Problem**: Compare two models with proper statistical tests.

**Tasks**:
1. Implement bootstrap resampling:
   ```python
   for _ in range(n_bootstrap):
       sample = resample(predictions)
       scores.append(metric(sample))
   ```
2. Calculate confidence intervals
3. Implement paired t-test
4. Implement McNemar's test for classification
5. Implement permutation test
6. Report p-values and effect sizes
7. Handle multiple comparisons (Bonferroni correction)

**When to use**:
- Bootstrap: general purpose
- t-test: comparing means
- McNemar: comparing classifiers

---

## Exercise 5: Human Evaluation Protocol (Intermediate)

**Objective**: Design human evaluation for LLM outputs.

**Problem**: Create reliable human evaluation framework.

**Tasks**:
1. Design evaluation rubric:
   - Fluency (1-5 scale)
   - Coherence (1-5 scale)
   - Relevance (1-5 scale)
   - Factuality (correct/incorrect)
2. Create annotation guidelines
3. Implement inter-annotator agreement:
   - Cohen's Kappa
   - Fleiss' Kappa (>2 annotators)
4. Design pairwise comparison task
5. Implement quality control (attention checks)
6. Calculate agreement statistics
7. Analyze disagreement patterns

---

## Exercise 6: Benchmark Evaluation Suite (Advanced)

**Objective**: Evaluate model on multiple standard benchmarks.

**Problem**: Comprehensive evaluation across diverse tasks.

**Tasks**:
1. Implement evaluation on benchmarks:
   - GLUE (classification tasks)
   - SuperGLUE (harder reasoning)
   - SQuAD (question answering)
   - HellaSwag (commonsense reasoning)
2. Standardize data loading
3. Implement task-specific metrics
4. Create unified evaluation pipeline
5. Aggregate scores appropriately
6. Compare with published baselines
7. Generate evaluation report

---

## Exercise 7: Bias and Fairness Metrics (Advanced)

**Objective**: Measure and analyze model biases.

**Problem**: Detect demographic and social biases in models.

**Tasks**:
1. Implement demographic parity:
   ```
   P(ŷ=1 | group=A) = P(ŷ=1 | group=B)
   ```
2. Implement equalized odds:
   ```
   P(ŷ=1 | y=1, group=A) = P(ŷ=1 | y=1, group=B)
   P(ŷ=1 | y=0, group=A) = P(ŷ=1 | y=0, group=B)
   ```
3. Test gender bias in embeddings
4. Implement WEAT for bias detection
5. Test for racial/ethnic bias
6. Measure performance disparities across groups
7. Generate bias report

---

## Exercise 8: Robustness Evaluation (Advanced)

**Objective**: Test model robustness to perturbations.

**Problem**: Evaluate stability under adversarial and natural noise.

**Tasks**:
1. Implement perturbations:
   - Character-level (typos, swaps)
   - Word-level (synonyms, deletions)
   - Sentence-level (paraphrases)
2. Create adversarial examples:
   - TextFooler
   - BERT-Attack
3. Test on out-of-distribution data
4. Measure performance degradation
5. Calculate robustness metrics:
   ```
   Robustness = accuracy_perturbed / accuracy_original
   ```
6. Analyze failure modes

---

## Exercise 9: Calibration and Uncertainty (Advanced)

**Objective**: Evaluate model confidence calibration.

**Problem**: Assess if confidence scores match true probabilities.

**Tasks**:
1. Implement Expected Calibration Error (ECE):
   ```
   ECE = Σ (|bin_accuracy - bin_confidence|) × (n_bin / n_total)
   ```
2. Plot reliability diagrams
3. Implement temperature scaling for calibration
4. Measure prediction entropy
5. Implement uncertainty estimation methods:
   - MC Dropout
   - Ensemble variance
6. Correlate uncertainty with errors
7. Use uncertainty for selective prediction

---

## Exercise 10: Long-Form Generation Evaluation (Advanced)

**Objective**: Evaluate longer generated texts holistically.

**Problem**: Go beyond n-gram overlap for long texts.

**Tasks**:
1. Implement coherence metrics:
   - Entity-based coherence
   - Discourse relations
2. Measure diversity:
   - Distinct n-grams
   - Self-BLEU (lower is more diverse)
3. Detect repetitions and loops
4. Measure factual consistency:
   - NLI-based methods
   - QA-based methods
5. Implement MAUVE for distribution matching
6. Human evaluation (comparative)
7. Combine metrics into single score

---

## Exercise 11: Error Analysis Framework (Intermediate)

**Objective**: Build systematic error analysis pipeline.

**Problem**: Understand where and why model fails.

**Tasks**:
1. Categorize errors by type
2. Stratify errors by:
   - Input characteristics (length, domain)
   - Output characteristics
   - Model confidence
3. Implement error visualization
4. Find patterns in failures
5. Prioritize improvements
6. Track error distribution over time
7. Create error taxonomy

---

## Exercise 12: Evaluation Reproducibility (Advanced)

**Objective**: Ensure reproducible evaluation across runs.

**Problem**: Get consistent evaluation results.

**Tasks**:
1. Set all random seeds
2. Document:
   - Model checkpoint
   - Evaluation code version
   - Data preprocessing
   - Hyperparameters
3. Implement deterministic evaluation
4. Report variance across runs
5. Create evaluation checklist
6. Package evaluation environment
7. Share evaluation artifacts

**Best Practices**:
- Version all code and data
- Use fixed test sets
- Report confidence intervals
- Document all choices

---

## Bonus Exercise: Meta-Evaluation (Advanced)

**Objective**: Evaluate the evaluation metrics themselves.

**Problem**: Assess how well metrics correlate with human judgments.

**Tasks**:
1. Collect human judgments on model outputs
2. Calculate metric-human correlation (Pearson, Spearman)
3. Test metric reliability (consistency)
4. Analyze metric biases:
   - Length bias
   - Vocabulary bias
5. Compare multiple metrics
6. Propose metric improvements
7. Create metric recommendation guide

**Goal**: Understand which metrics are most reliable for which tasks
