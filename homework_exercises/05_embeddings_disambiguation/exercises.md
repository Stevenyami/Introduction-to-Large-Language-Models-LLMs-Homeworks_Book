# Word and Sentence Embeddings Exercises

## Exercise 1: Word2Vec Skip-gram Implementation (Intermediate)

**Objective**: Implement Word2Vec Skip-gram model from scratch.

**Problem**: 
Build Skip-gram with negative sampling to learn word embeddings.

**Tasks**:
1. Implement Skip-gram architecture:
   ```
   Input: center word → Embedding layer → Output: context words
   ```
2. Create training pairs (center word, context word)
3. Implement negative sampling:
   - Sample k negative words (k=5)
   - Use unigram distribution P(w)^(3/4)
4. Implement loss function:
   ```
   L = -log σ(v_c · v_o) - Σ log σ(-v_c · v_n)
   ```
5. Train on text corpus
6. Evaluate with word analogies
7. Visualize embeddings with t-SNE

**Key Concepts**:
- Context window
- Negative sampling
- Embedding quality

---

## Exercise 2: GloVe Embeddings (Intermediate)

**Objective**: Understand and implement GloVe (Global Vectors) embeddings.

**Problem**: 
Build word embeddings using co-occurrence statistics.

**Tasks**:
1. Build co-occurrence matrix:
   ```python
   X[i,j] = count of word j in context of word i
   ```
2. Implement GloVe objective:
   ```
   J = Σ f(X_ij) × (w_i^T w_j + b_i + b_j - log X_ij)²
   where f(x) = (x/x_max)^α if x < x_max else 1
   ```
3. Train with AdaGrad optimizer
4. Set hyperparameters:
   - Embedding dimension: 100
   - Context window: 10
   - x_max: 100
   - α: 0.75
5. Compare with Word2Vec
6. Analyze semantic relationships

---

## Exercise 3: FastText Character N-grams (Intermediate)

**Objective**: Implement FastText to handle out-of-vocabulary words.

**Problem**: 
Extend Word2Vec with subword information.

**Tasks**:
1. Generate character n-grams:
   ```python
   word = "<where>"
   n_grams = ["<wh", "whe", "her", "ere", "re>"]
   ```
2. Represent word as sum of n-gram embeddings
3. Handle OOV words using n-grams
4. Train Skip-gram with n-gram features
5. Test on morphologically rich languages
6. Compare with standard Word2Vec on OOV words
7. Analyze subword patterns learned

**Advantages**:
- Handles OOV words
- Captures morphology
- Better for rare words

---

## Exercise 4: Sentence Embeddings (Intermediate)

**Objective**: Create fixed-size representations for variable-length sentences.

**Problem**: 
Implement and compare sentence embedding methods.

**Tasks**:
1. Implement baseline methods:
   - Average of word embeddings
   - Weighted average (TF-IDF weights)
   - Max pooling over word embeddings
2. Implement Doc2Vec (Paragraph Vectors):
   - Distributed memory (DM)
   - Distributed bag of words (DBOW)
3. Implement Universal Sentence Encoder approach
4. Compare methods on semantic similarity task
5. Evaluate with cosine similarity
6. Test on sentence classification

**Evaluation**:
- Semantic Textual Similarity (STS) benchmark
- Classification accuracy
- Clustering quality

---

## Exercise 5: Contextualized Embeddings (Advanced)

**Objective**: Extract and use contextualized word representations from BERT.

**Problem**: 
Compare static vs contextualized embeddings.

**Tasks**:
1. Extract embeddings from BERT layers:
   ```python
   outputs = model(input_ids, output_hidden_states=True)
   embeddings = outputs.hidden_states
   ```
2. Compare embeddings from different layers:
   - Layer 1 (surface features)
   - Layer 6 (syntactic features)
   - Layer 12 (semantic features)
3. Handle subword tokenization (WordPiece)
4. Aggregate subword embeddings to word level
5. Test on polysemous words (e.g., "bank")
6. Visualize how embeddings change with context
7. Use for downstream task

**Key Insight**:
- Same word gets different embeddings in different contexts
- Different layers capture different linguistic properties

---

## Exercise 6: Word Sense Disambiguation (Advanced)

**Objective**: Disambiguate word meanings using context.

**Problem**: 
Identify the correct sense of polysemous words.

**Tasks**:
1. Implement context-based WSD:
   ```
   Given: "I went to the bank to deposit money"
   Identify: bank = financial institution (not river bank)
   ```
2. Use sense embeddings from WordNet
3. Implement Lesk algorithm:
   - Compare context with sense definitions
   - Select sense with highest overlap
4. Implement neural WSD:
   - Use BERT embeddings
   - Train classifier for sense prediction
5. Evaluate on SemEval WSD task
6. Analyze error patterns

**Baseline Methods**:
- Most Frequent Sense (MFS)
- Lesk algorithm
- Context similarity

---

## Exercise 7: Semantic Similarity Metrics (Intermediate)

**Objective**: Implement and compare similarity measures for text.

**Problem**: 
Measure semantic similarity between texts.

**Tasks**:
1. Implement similarity metrics:
   - Cosine similarity
   - Euclidean distance
   - Manhattan distance
   - Jaccard similarity
2. Implement sentence similarity:
   - Embedding-based (cosine)
   - Edit distance-based
   - BERTScore
3. Create similarity matrix for document collection
4. Implement k-nearest neighbors search
5. Evaluate on STS benchmark
6. Visualize similarity distributions

**Applications**:
- Duplicate detection
- Information retrieval
- Recommendation systems

---

## Exercise 8: Embedding Analogies (Beginner)

**Objective**: Explore and evaluate word analogies using embeddings.

**Problem**: 
Test if embeddings capture semantic relationships.

**Tasks**:
1. Implement analogy solving:
   ```
   "king - man + woman = ?"
   Answer: argmax(cos_sim(v, queen))
   ```
2. Test on standard analogy dataset:
   - Semantic analogies: "Paris:France :: Berlin:Germany"
   - Syntactic analogies: "good:better :: bad:worse"
3. Implement cosine distance method
4. Calculate analogy accuracy
5. Analyze which relationships are captured well
6. Find interesting analogies in domain-specific corpus

**Analogy Types**:
- Geographic relationships
- Gender relationships
- Verb tenses
- Comparative/superlative

---

## Exercise 9: Bias Detection in Embeddings (Advanced)

**Objective**: Detect and analyze social biases in word embeddings.

**Problem**: 
Identify gender, racial, and other biases in embeddings.

**Tasks**:
1. Implement bias tests:
   ```python
   # Gender bias
   gender_bias = cos_sim("programmer", "he") - cos_sim("programmer", "she")
   ```
2. Create bias word pairs:
   - Gender: (he, she), (man, woman)
   - Race/ethnicity
   - Religion
3. Measure bias in word associations:
   - Professions and gender
   - Names and sentiment
4. Implement WEAT (Word Embedding Association Test)
5. Quantify bias scores
6. Propose debiasing methods
7. Evaluate bias-accuracy trade-off

**Debiasing Approaches**:
- Hard debiasing (neutralize and equalize)
- Soft debiasing (reduce but don't eliminate)

---

## Exercise 10: Multilingual Embeddings (Advanced)

**Objective**: Create and align embeddings across languages.

**Problem**: 
Learn shared embedding space for multiple languages.

**Tasks**:
1. Train monolingual embeddings for 2+ languages
2. Implement alignment methods:
   - Supervised (with translation pairs)
   - Unsupervised (Procrustes alignment)
3. Map embeddings to shared space:
   ```
   W = argmin ||W × X - Y||
   where X, Y are aligned word pairs
   ```
4. Test cross-lingual similarity
5. Implement zero-shot cross-lingual transfer
6. Evaluate on MUSE benchmark
7. Visualize aligned embedding spaces

**Applications**:
- Cross-lingual information retrieval
- Machine translation
- Zero-shot learning

---

## Exercise 11: Dynamic Word Embeddings (Advanced)

**Objective**: Track how word meanings change over time.

**Problem**: 
Analyze semantic shift in diachronic corpora.

**Tasks**:
1. Train embeddings on different time periods
2. Align embeddings across time:
   ```python
   # Align 1990s embeddings to 2020s
   align(embeddings_1990, embeddings_2020)
   ```
3. Measure semantic shift:
   ```
   shift = 1 - cos_sim(word_t1, word_t2)
   ```
4. Identify words with largest shifts
5. Analyze shift patterns:
   - Semantic broadening
   - Semantic narrowing
   - Meaning transfer
6. Visualize temporal evolution
7. Test hypotheses about language change

**Examples of Semantic Shift**:
- "gay": happy → homosexual
- "awful": inspiring awe → very bad
- Technology terms gaining new meanings

---

## Exercise 12: Domain-Specific Embeddings (Intermediate)

**Objective**: Train embeddings on specialized domain corpora.

**Problem**: 
Create embeddings for medical, legal, or scientific domains.

**Tasks**:
1. Select domain corpus (e.g., medical papers)
2. Train domain-specific embeddings
3. Compare with general-purpose embeddings:
   - Coverage of domain terms
   - Similarity rankings
   - Downstream task performance
4. Implement domain adaptation:
   - Fine-tune pre-trained embeddings
   - Combine general and domain embeddings
5. Evaluate on domain-specific tasks
6. Analyze domain-specific relationships

**Domain Examples**:
- Medical: disease-symptom relationships
- Legal: case law citations
- Scientific: chemical compounds

---

## Bonus Exercise: Sentence-BERT (Advanced)

**Objective**: Fine-tune BERT for sentence similarity using siamese networks.

**Problem**: 
Create semantically meaningful sentence embeddings with BERT.

**Tasks**:
1. Implement siamese network architecture:
   ```
   Sent1 → BERT → Pool → Embedding1
   Sent2 → BERT → Pool → Embedding2
   Loss = triplet_loss(Embedding1, Embedding2)
   ```
2. Implement pooling strategies:
   - CLS token
   - Mean pooling
   - Max pooling
3. Train with triplet loss:
   ```
   L = max(0, ||anchor - positive|| - ||anchor - negative|| + margin)
   ```
4. Create training data from NLI datasets
5. Fine-tune on semantic similarity
6. Evaluate on STS benchmark
7. Compare with vanilla BERT embeddings

**Result**: Dramatically better sentence embeddings than averaging BERT token embeddings
