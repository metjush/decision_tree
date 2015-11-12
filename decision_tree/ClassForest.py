__author__ = 'metjush'

# Implementation of Classification Random Forest
# ============================================
# This Random Forest is built on the Classification Tree object implemented in ClassTree.py
#
# It uses bootstrap aggregating and feature subsetting to grow the forest
#
# The primary parameters to input are the number of trees to grow,
#  the number of examples to select for each iteration, and the maximum depth of each forest
# In contrast to the Bagged Forest implemented in ClassTreeBagging.py, the Random Forest implementation
#  also subsets features with each tree grown
# Works best with a large number of features. If n < 9, all features are used in all trees.

import numpy as np
from ClassTree import ClassificationTree
from scipy import stats
import warnings


class RandomForest:

    def __init__(self, n_trees=50, depth_limit=None, sample_fraction=0.75, impurity="gini"):
        self.n_trees = n_trees
        self.depth_limit = depth_limit if depth_limit in set({int, float, np.int64, np.float64}) else np.inf
        self.fraction = sample_fraction
        self.trees = [[]]*n_trees
        self.trained = False
        self.impurity = impurity

    def __untrain(self):
        self.trained = False
        self.trees = [[]]*self.n_trees
        print("Retraining")

    #__numpify() takes a regular python list and turns it into a numpy array
    def __numpify(self, array):
        numpied = np.array(array)
        if numpied.dtype in ['int64', 'float64']:
            return numpied
        else:
            return False

    def train(self, X, y):
        # check dimensions
        if not len(X) == len(y):
            raise IndexError("The number of samples in X and y do not match")
        # check if X and y are numpy arrays
        if type(X) is not np.ndarray:
            X = self.__numpify(X)
            if not X:
                raise TypeError("input dataset X is not a valid numeric array")
        if type(y) is not np.ndarray:
            y = self.__numpify(y)
            if not y:
                raise TypeError("input label vector y is not a valid numeric array")

        # check if trained
        if self.trained:
            self.__untrain()

        indices = np.arange(len(X))
        # determine the size of the bootstrap sample
        strapsize = np.int(len(X)*self.fraction)
        features = np.arange(X.shape[1])
        # determine the number of features to subsample each iteration
        # using the sqrt(n) rule of thumb if n > 10
        subsize = np.ceil(np.sqrt(X.shape[1])).astype(np.int) if X.shape[1] >= 9 else X.shape[1]

        # start growing the tree
        for t in xrange(self.n_trees):
            # creat a new classification tree
            tree = ClassificationTree(depth_limit=self.depth_limit, impurity=self.impurity)
            # bootstrap a sample
            bootstrap = np.random.choice(indices, strapsize)
            subfeature = np.random.choice(features, subsize, replace=False) #features are not sampled with replacement
            Xstrap = X[bootstrap,:][:,subfeature]
            ystrap = y[bootstrap]
            # train the t-th tree with the strapped sample
            tree.train(Xstrap,ystrap)
            # for each tree, need to save which features to use
            self.trees[t] = [tree, subfeature]
        self.trained = True
        print("%d trees grown" % self.n_trees)

    def predict(self, X):
        if not self.trained:
            raise RuntimeError("The random forest classifier hasn't been trained yet")
        # the exception to the bagged forest is the subsetting of features,
        # which needs to be account for in prediction/evaluation too
        prediction_matrix = np.zeros((len(X), self.n_trees))
        for t in xrange(self.n_trees):
            tree = self.trees[t][0]
            subX = X[:,self.trees[t][1]]
            pred = tree.predict(subX)
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
