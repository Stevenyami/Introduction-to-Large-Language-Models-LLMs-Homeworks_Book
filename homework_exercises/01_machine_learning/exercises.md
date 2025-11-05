# Machine Learning Exercises

## Exercise 1: Linear Regression from Scratch (Beginner)

**Objective**: Implement linear regression using gradient descent without using scikit-learn.

**Problem**: 
Given a dataset of house sizes (in square feet) and prices, implement a linear regression model to predict house prices.

**Tasks**:
1. Implement the hypothesis function: h(x) = θ₀ + θ₁x
2. Implement the cost function: J(θ) = (1/2m) Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
3. Implement gradient descent to minimize the cost function
4. Plot the regression line with the data points
5. Calculate R² score to evaluate model performance

**Data Format**:
```python
X = [1000, 1500, 2000, 2500, 3000]  # Square feet
y = [200000, 250000, 300000, 350000, 400000]  # Price in USD
```

**Expected Output**:
- Final parameters θ₀ and θ₁
- Final cost value
- R² score
- Visualization of the fitted line

---

## Exercise 2: Logistic Regression for Binary Classification (Intermediate)

**Objective**: Build a logistic regression classifier to predict whether a text is spam or not.

**Problem**:
Given a dataset of email features, classify emails as spam (1) or not spam (0).

**Tasks**:
1. Implement the sigmoid function: σ(z) = 1 / (1 + e⁻ᶻ)
2. Implement the cost function for logistic regression
3. Implement gradient descent with learning rate α = 0.01
4. Train the model for 1000 iterations
5. Calculate accuracy, precision, recall, and F1-score
6. Plot the decision boundary

**Features** (example):
- Number of capital letters
- Presence of certain keywords
- Email length
- Number of links

**Evaluation Metrics**:
- Accuracy = (TP + TN) / (TP + TN + FP + FN)
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1-Score = 2 × (Precision × Recall) / (Precision + Recall)

---

## Exercise 3: K-Means Clustering (Intermediate)

**Objective**: Implement K-means clustering algorithm to group similar documents.

**Problem**:
Given TF-IDF vectors of documents, cluster them into k groups.

**Tasks**:
1. Initialize k centroids randomly
2. Implement the assignment step (assign each point to nearest centroid)
3. Implement the update step (recalculate centroids)
4. Iterate until convergence or max iterations
5. Visualize clusters using PCA for dimensionality reduction
6. Calculate silhouette score for cluster quality

**Algorithm**:
```
1. Randomly initialize k centroids
2. Repeat until convergence:
   a. Assign each point to the nearest centroid
   b. Update centroids as mean of assigned points
3. Return cluster assignments
```

**Evaluation**:
- Silhouette score: measures how similar an object is to its cluster vs other clusters
- Elbow method: plot distortion vs number of clusters

---

## Exercise 4: Decision Tree Classifier (Intermediate)

**Objective**: Build a decision tree classifier from scratch using information gain.

**Problem**:
Create a decision tree to classify text sentiment (positive/negative/neutral).

**Tasks**:
1. Implement entropy calculation: H(S) = -Σ p(x) log₂ p(x)
2. Implement information gain: IG(S, A) = H(S) - Σ |Sᵥ|/|S| × H(Sᵥ)
3. Implement the tree-building algorithm (recursive splitting)
4. Implement prediction function
5. Prune the tree to avoid overfitting
6. Visualize the decision tree

**Splitting Criteria**:
- Use information gain to select best feature at each node
- Stop when: maximum depth reached, minimum samples per leaf, or pure node

---

## Exercise 5: Cross-Validation and Model Selection (Advanced)

**Objective**: Implement k-fold cross-validation to compare different ML models.

**Problem**:
Compare the performance of multiple classifiers on a text classification task.

**Tasks**:
1. Implement k-fold cross-validation (k=5)
2. Test at least 3 different models (e.g., Logistic Regression, SVM, Random Forest)
3. Calculate mean and standard deviation of scores across folds
4. Perform hyperparameter tuning using grid search
5. Select the best model based on validation scores
6. Test the final model on a held-out test set

**Metrics to Report**:
- Mean accuracy across folds
- Standard deviation of accuracy
- Precision, Recall, F1-Score
- Confusion matrix
- ROC curve and AUC score

---

## Exercise 6: Feature Engineering for NLP (Advanced)

**Objective**: Extract and engineer features from text data for classification.

**Problem**:
Given raw text data, create a comprehensive feature set for sentiment analysis.

**Tasks**:
1. Implement text preprocessing (lowercasing, removing punctuation, tokenization)
2. Create bag-of-words features
3. Implement TF-IDF transformation
4. Extract n-gram features (bigrams and trigrams)
5. Add custom features:
   - Text length
   - Average word length
   - Presence of punctuation/emojis
   - Sentiment lexicon scores
6. Compare model performance with different feature sets

**Feature Sets to Compare**:
- Bag-of-words only
- TF-IDF only
- TF-IDF + n-grams
- TF-IDF + n-grams + custom features

---

## Exercise 7: Regularization and Overfitting (Advanced)

**Objective**: Understand and implement L1 (Lasso) and L2 (Ridge) regularization.

**Problem**:
Fit a polynomial regression model with regularization to prevent overfitting.

**Tasks**:
1. Generate synthetic data with noise
2. Fit polynomial models of different degrees (1 to 10)
3. Implement cost function with L2 regularization: J(θ) = MSE + λΣθ²
4. Implement cost function with L1 regularization: J(θ) = MSE + λΣ|θ|
5. Compare models with and without regularization
6. Plot learning curves (training vs validation error)
7. Select optimal regularization parameter λ using validation set

**Comparison Metrics**:
- Training error vs validation error
- Model complexity (number of parameters)
- Bias-variance tradeoff visualization

---

## Exercise 8: Ensemble Methods (Advanced)

**Objective**: Implement bagging and boosting ensemble techniques.

**Problem**:
Create an ensemble model for text classification that combines multiple weak learners.

**Tasks**:
1. Implement bootstrap sampling for bagging
2. Create a bagging ensemble with decision trees
3. Implement AdaBoost algorithm:
   - Train weak learner
   - Calculate weighted error
   - Update sample weights
   - Combine weak learners
4. Compare single model vs ensemble performance
5. Visualize the improvement with number of estimators

**AdaBoost Algorithm**:
```
1. Initialize sample weights w_i = 1/N
2. For t = 1 to T:
   a. Train weak learner h_t on weighted samples
   b. Calculate error: ε_t = Σ w_i × I(h_t(x_i) ≠ y_i)
   c. Calculate learner weight: α_t = 0.5 × ln((1-ε_t)/ε_t)
   d. Update sample weights: w_i = w_i × exp(-α_t × y_i × h_t(x_i))
   e. Normalize weights
3. Final prediction: H(x) = sign(Σ α_t × h_t(x))
```

---

## Bonus Exercise: Support Vector Machines (SVM) (Advanced)

**Objective**: Understand and implement a basic SVM classifier.

**Problem**:
Implement a linear SVM for binary text classification.

**Tasks**:
1. Understand the SVM objective: maximize margin between classes
2. Implement the hinge loss function: L = max(0, 1 - y × (w·x + b))
3. Train using gradient descent or SMO algorithm
4. Implement kernel trick for non-linear classification (RBF kernel)
5. Tune hyperparameters (C parameter and kernel parameters)
6. Visualize support vectors and decision boundary

**Key Concepts**:
- Margin maximization
- Support vectors
- Kernel trick
- Soft margin (C parameter)
