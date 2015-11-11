__author__ = "metjush"

# An example file for the decision_tree repository, using datasets from scikit-learn
# to demonstrate classification with a single tree, bagged forest and random forest.
# If you just want to see if the package works, run this file.

# Importing all requirements

import numpy as np
from ClassTree import ClassificationTree
from ClassTreeBagging import TreeBagger
from ClassForest import RandomForest

# Create the classifier objects
tree = ClassificationTree()
bag = TreeBagger(n_trees=50)
forest = RandomForest(n_trees=50)

# Get datasets from scikit-learn
from sklearn.datasets import load_iris # iris classification

# Save to arrays
iris = load_iris()

X_iris = iris.data
y_iris = iris.target

# Train classifiers with Iris data

# Simple tree training
tree.train(X_iris, y_iris)
print("Accuracy of the simple tree on iris dataset is %f" % tree.evaluate(X_iris, y_iris))
tree.describe()

#write to json
js = tree.to_json("iris_json.json")

#load from json
tree = tree.from_json("iris_json.json")

#check it is the same
tree.describe()
print("Accuracy of the reloaded tree on iris dataset is %f" % tree.evaluate(X_iris, y_iris))

# Cross validation of a tree
tree.cross_val(X_iris, y_iris, folds=5)

# Tree bag
bag.train(X_iris, y_iris)
print("Accuracy of the bagged forest on iris dataset is %f" % bag.evaluate(X_iris, y_iris))

# Cross validation of a tree bag
bag.cross_val(X_iris, y_iris, folds=5)

# Random forest
forest.train(X_iris, y_iris)
print("Accuracy of the random forest on iris dataset is %f" % forest.evaluate(X_iris, y_iris))

