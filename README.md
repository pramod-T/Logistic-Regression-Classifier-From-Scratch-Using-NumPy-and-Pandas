# Logistic Regression Classifier From Scratch Using Python Numpy and Pandas
Classic Logistic Regression is a Supervised Machine learning algorithm used for binary classification, where the goal is to predict the probability of an outcome belonging to one of two classes. Unlike linear regression, which predicts continuous values, 
logistic regression models the **log-odds of the probability** as a linear combination of input features and then maps it through a **sigmoid function** to ensure 
the output lies between 0 and 1. The model is trained by **minimizing the binary cross-entropy (log loss)** using gradient descent, and predictions are made by 
applying a threshold to the probability. It is widely used due to its simplicity, interpretability, and effectiveness in many classification tasks.

This project include :
- building a classifier using gradient descent
- sigmoid activation
- evaluation with standard metrics like accuracy, precision, recall, and F1-score.

This also includes explanations of the **logit function**, **sigmoid function**, which is useful for learning logistic regression fundamentals.

# Background / Theory
### Logit(p) - log of odds (how much more likely success is than failure). 

Linear relationship:
- We want a model that relates features x linearly to the target(y).
- But probabilities p∈(0,1) can’t be modeled linearly — they’re bounded.
- So we apply a transformation that stretches probabilities into the real line (−∞,∞)
- Applying Log to the Odds Makes it Symmetric: if odds = 0.5 → log-odds = -0.693; if odds = 2 → log-odds = 0.693.​


$$
\text{logit(p)} = \frac{p}{1-p}
$$

Here:
- Probability of positive class: p = P(y=1∣X).
- Probability of negative class: 1−p = P(y=0∣X).

It Makes the relationship linear in features:

$$
\text{logit}(p) = \log\left(\frac{p}{1-p}\right) = z = \beta_0 + \beta_1 x_1 + \dots + \beta_n x_n
$$

### Sigmoid Function: 
This function This allows us to convert linear log-odds back into valid probabilities.

Deriving sigmod function:

$$
\begin{align}
\log\frac{p}{1-p} = z \\
\frac{p}{1-p} = e^z \quad \text{(exponentiate both sides)} \\
p = \frac{e^z}{1+e^z} = \frac{1}{1+e^{-z}}
\end{align}
$$

- This is the sigmoid function σ(z).
- Maps any real number z∈(−∞,∞) to a probability p∈(0,1)​.

## Sigmoid() fit() and predict() functions

```
def sigmoid(self,x):
  return 1 / (1 + np.exp(-x))

def fit(self, X, Y):
  m, n = X.shape
  self.weights = np.zeros(n)
  for i in range(self.epochs):
      z = np.dot(X, self.weights)
      p = self.sigmoid(z)
      gradient = X.T.dot(p - Y) / m
      self.weights -= self.lr * gradient

def predict(self,X_test):
  predictions = []
  for index, row in X_test.iterrows():
      z = np.dot(row, self.weights)
      p = self.sigmoid(z)
      predictions.append(0 if p < 0.5 else 1)
  return np.array(predictions)
    
```

## Binary Classification

Map the probabilities into classes.

prediction class y = { 0 if p < 0.5 else 1 } 

## Performance

- Accuracy = 0.706
- Precision = 0.568
- Recall	= 0.764
- F1-score	= 0.651
- Confusion matrix:
[[42 32]
 [13 66]]

 Model catches most positives (high recall) but has moderate precision.

 ## Notes / Tips

- Adjust learning rate and epochs for convergence.

- Standardizing features may improve stability.




