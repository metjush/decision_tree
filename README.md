# Classification Decision Tree

This repository is a basic implementation of a decision tree algorithm for supervised classification learning. It implements a basic tree classifier, as well as a wrapper for tree bagging (bootstrap aggregating).

## Basic usage

### Single Tree

The `ClassificationTree` object can be created with one basic parameter: `depth_limit`. This restrict tree depth to prevent over-fitting.

```python
from ClassTree import *
tree = ClassificationTree(depth_limit = 10)
```

The public methods available with this object are `train()`, `predict()`, `evaluate()` and `cross_val()`. The tree can be learned with the `train` function, supplying the feature dataset `X` and class labels `y` (as numpy arrays):

```python
from ClassTree import *
tree = ClassificationTree(depth_limit = 10)

tree.train(X,y)
```

Upon training, you can use the `predict` function to use the learned decision rules to classify an unlabeled dataset `X`. This function will raise an error if you call it before training. It returns a one-dimensional numpy array of class predictions.

```python
from ClassTree import *
tree = ClassificationTree(depth_limit = 10)

tree.train(X,y)

predictions = tree.predict(X)
```

An alternative to the `predict` function is the `evaluate` function that runs the prediction and then evaluates the predictions based on supplied ground truth. The default scoring rule is the _F1-score_. Other options include simple _classification accuracy_ and the _Matthews correlation coefficient_. It returns a float.

```python
from ClassTree import *
tree = ClassificationTree(depth_limit = 10)

tree.train(X,y)

predictions = tree.predict(X)

score_f1 = tree.evaluate(X,y)
score_accuracy = tree.evaluate(X,y,method='acc')
score_matt = tree.evaluate(X,y,method= 'matthews')
```
