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

tree.train(X, y)
```

Upon training, you can use the `predict` function to use the learned decision rules to classify an unlabeled dataset `X`. This function will raise an error if you call it before training. It returns a one-dimensional numpy array of class predictions.

```python
from ClassTree import *
tree = ClassificationTree(depth_limit = 10)

tree.train(X, y)

predictions = tree.predict(X)
```

An alternative to the `predict` function is the `evaluate` function that runs the prediction and then evaluates the predictions based on supplied ground truth. The default scoring rule is the _F1-score_. Other options include simple _classification accuracy_ and the _Matthews correlation coefficient_. It returns a float.

```python
from ClassTree import *
tree = ClassificationTree(depth_limit = 10)

tree.train(X, y)

predictions = tree.predict(X)

score_f1 = tree.evaluate(X, y)
score_accuracy = tree.evaluate(X, y, method = 'acc')
score_matt = tree.evaluate(X, y, method = 'matthews')
```

Instead of straight-forward training, you can also use cross-validation with the `cross_val` function. You can specify the number of "folds" (the number of validation trainings, defaults to 1) and the fraction of the supplied dataset that should be left out as validation set (defaults to 0.3), as well as the scoring method (defaults to F1). It returns an array of scores for each cross-validation fold.

```python
from ClassTree import *
tree = ClassificationTree(depth_limit = 10)

cross_val_scores = tree.cross_val(X, y, split = 0.3, method = 'f1', folds = 5)
```

### Bagged Forest

The `TreeBagger` object implements a wrapper for growing a "forest" of "bagged" trees. _Bagging_ refers to _bootstrap aggregating_, where for a specified number of iterations, a new tree is grown with a bootstrapped subsample (with repetition) of the supplied dataset. The class `init`s with parameters that specify the number of trees (`n_trees`), the depth limit of each tree (`depth_limit`) and the fraction of the dataset that should be used as a size of each bootstrap sample (`sample_fraction`).

```python
from ClassTreeBagging import *
bag = TreeBagger(n_trees=50, depth_limit = 10, sample_fraction=0.75)
```

The public methods are the same as those for the simple tree, with the exception of cross-validation, which isn't currently implemented for the Bagged Forest. 

