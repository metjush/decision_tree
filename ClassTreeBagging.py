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
from ClassTree import ClassificationTree
from scipy import stats
import warnings


class TreeBagger:

    def __init__(self, n_trees=50, depth_limit=None, sample_fraction=0.75):
        self.n_trees = n_trees
        self.depth_limit = depth_limit if depth_limit in set({int, float, np.int64, np.float64}) else np.inf
        self.fraction = sample_fraction
        self.trees = [0]*n_trees

    def train(self, X, y):
        #TODO: check that X,y are good
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
        #get predictions from each tree
        #combine predictions into one matrix
        #get the mode of predictions for each sample
        prediction_matrix = np.zeros((len(X), self.n_trees))
        for t in range(self.n_trees):
            pred = self.trees[t].predict(X)
            prediction_matrix[:,t] = pred
        final_vote = stats.mode(prediction_matrix, axis=1)[0]

        return final_vote.flatten()

    def evaluate(self, X, y, method='f1'):
        yhat = self.predict(X)
        accurate = y == yhat
        positive = np.sum(y == 1)
        hatpositive = np.sum(yhat == 1)
        tp = np.sum(yhat[accurate] == 1)

        #F1 score
        if method == 'f1':
            recall = 1.*tp/positive if positive > 0 else 0.
            precision = 1.*tp/hatpositive if hatpositive > 0 else 0.
            f1 = (2.*precision*recall)/(precision+recall) if (precision+recall) > 0 else 0.
            return f1
        #simple accuracy measure
        elif method == 'acc':
            return (1.*np.sum(accurate))/len(yhat)
        #matthews correlation coefficient
        elif method == 'matthews':
            tn = np.sum(yhat[accurate] == 0)
            fp = np.sum(yhat[np.invert(accurate)] == 1)
            fn = np.sum(yhat[np.invert(accurate)] == 0)
            denominator = np.sqrt( (tp+fp)*(tp+fn)*(tn+fp)*(tn*fn) )
            mat = 1.*((tp*tn)-(fp*fn)) / denominator if denominator > 0 else 0.
            return mat
        else:
            warnings.warn("Wrong evaluation method specified, defaulting to F1 score", RuntimeWarning)
            return self.evaluate(X,y)

