# Parsing Exercises

## Exercise 1: CKY Algorithm for Constituency Parsing (Intermediate)

**Objective**: Implement the Cocke-Kasami-Younger (CKY) algorithm for context-free grammar parsing.

**Problem**: Parse sentences using a context-free grammar in Chomsky Normal Form.

**Tasks**:
1. Convert grammar to Chomsky Normal Form (CNF)
2. Implement CKY dynamic programming algorithm
3. Build parse table bottom-up
4. Extract parse tree from table
5. Handle ambiguous parses
6. Visualize parse trees

**Example Grammar**:
```
S → NP VP
NP → Det N | NP PP
VP → V NP | VP PP
PP → P NP
Det → "the" | "a"
N → "cat" | "dog" | "park"
V → "saw" | "chased"
P → "in" | "with"
```

---

## Exercise 2: Transition-Based Dependency Parsing (Advanced)

**Objective**: Implement arc-standard transition-based dependency parser.

**Problem**: Build dependency trees using a stack-based parser.

**Tasks**:
1. Implement parser with stack and buffer
2. Define transitions: SHIFT, LEFT-ARC, RIGHT-ARC
3. Implement oracle for training data creation
4. Train neural model to predict transitions
5. Parse sentences at inference time
6. Evaluate with UAS and LAS scores
7. Visualize dependency trees

**Architecture**:
- Stack: partially built tree
- Buffer: remaining words
- Actions: shift, left-arc, right-arc

---

## Exercise 3: Neural Dependency Parser (Advanced)

**Objective**: Build a neural network-based dependency parser.

**Problem**: Use BiLSTM with attention for dependency parsing.

**Tasks**:
1. Implement BiLSTM encoder
2. Add biaffine attention for arc scoring
3. Implement MST (Maximum Spanning Tree) decoding
4. Train with cross-entropy loss
5. Handle non-projective trees
6. Evaluate on Universal Dependencies
7. Compare with transition-based parser

---

## Exercise 4: Semantic Role Labeling (Advanced)

**Objective**: Implement semantic role labeling to identify predicate-argument structures.

**Problem**: Identify "who did what to whom" in sentences.

**Tasks**:
1. Identify predicates in sentence
2. For each predicate, identify arguments
3. Classify argument roles (Agent, Patient, etc.)
4. Implement BIO tagging scheme
5. Use BERT as encoder
6. Train on PropBank/FrameNet
7. Evaluate with F1 score per role

**Example**:
```
"John gave Mary a book"
Predicate: gave
ARG0 (Agent): John
ARG1 (Theme): a book
ARG2 (Recipient): Mary
```

---

## Exercise 5: Constituency vs Dependency Parsing Comparison (Intermediate)

**Objective**: Compare constituency and dependency parsing approaches.

**Tasks**:
1. Implement both parsers on same data
2. Convert between representations
3. Compare:
   - Parsing complexity
   - Information captured
   - Downstream task performance
4. Analyze strengths and weaknesses
5. Test on different sentence structures
