__author__ = 'metjush'

# Implementation of Classification Tree Bagging
# ============================================
# This Bag of Trees is built on the Classification Tree object implemented in classtree.py
#
# It uses bootstrap aggregating to grow the forest
#
# The primary parameters to input are the number of trees to grow,
#  the number of examples to select for each iteration, and the maximum depth of each forest

import numpy as np
from classtree import ClassificationTree


class TreeBagger:

    def __init__(self, n_trees=50, depth_limit=None, sample_fraction=0.75):
        self.n_trees = n_trees
        self.depth_limit = depth_limit if depth_limit in set({int, float, np.int64, np.float64}) else np.inf
        self.fraction = sample_fraction
        self.trees = [0]*n_trees

    def train(self, X, y):
        #check that X,y are good
        indices = np.arange(len(X))
        #determine the size of the bootstrap sample
        strapsize = np.int(len(X)*self.fraction)
        for t in range(self.n_trees):
            #creat a new classification tree
            tree = ClassificationTree(depth_limit=self.depth_limit)
            #bootstrap a sample
            bootstrap = np.random.choice(indices, strapsize)
            Xstrap = X[bootstrap,:]
            ystrap = y[bootstrap]
            #train the t-th tree with the strapped sample
            tree.train(Xstrap,ystrap)
            self.trees[t] = tree
        print("%d trees grown" % self.n_trees)

    def predict(self, X):
        return 0

    def evaluate(self, X, y):
        return 0






