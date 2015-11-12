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

    def __init__(self, n_trees=50, depth_limit=None, sample_fraction=0.75, impurity="gini"):
        self.n_trees = n_trees
        self.depth_limit = depth_limit if depth_limit in set({int, float, np.int64, np.float64}) else np.inf
        self.fraction = sample_fraction
        self.trees = [0]*n_trees
        self.trained = False
        self.impurity = impurity

    def __untrain(self):
        self.trained = False
        self.trees = [0]*self.n_trees
        print("Retraining")

    #__numpify() takes a regular python list and turns it into a numpy array
    def __numpify(self, array):
        numpied = np.array(array)
        if numpied.dtype in ['int64', 'float64']:
            return numpied
        else:
            return False

    # train() trains the Bagged Forest with input numpy arrays X and y
    def train(self, X, y):
        #check dimensions
        if not len(X) == len(y):
            raise IndexError("The number of samples in X and y do not match")
        #check if X and y are numpy arrays
        if type(X) is not np.ndarray:
            X = self.__numpify(X)
            if not X:
                raise TypeError("input dataset X is not a valid numeric array")
        if type(y) is not np.ndarray:
            y = self.__numpify(y)
            if not y:
                raise TypeError("input label vector y is not a valid numeric array")

        #check if trained
        if self.trained:
            self.__untrain()

        indices = np.arange(len(X))
        #determine the size of the bootstrap sample
        strapsize = np.int(len(X)*self.fraction)
        for t in xrange(self.n_trees):
            #creat a new classification tree
            tree = ClassificationTree(depth_limit=self.depth_limit, impurity=self.impurity)
            #bootstrap a sample
            bootstrap = np.random.choice(indices, strapsize)
            Xstrap = X[bootstrap,:]
            ystrap = y[bootstrap]
            #train the t-th tree with the strapped sample
            tree.train(Xstrap,ystrap)
            self.trees[t] = tree
        self.trained = True
        print("%d trees grown" % self.n_trees)

    # predict() uses a trained Bagged Forest to predict labels for a supplied numpy array X
    # returns a one-dimensional vector of predictions, which is selected by a plurality
    # vote from all the bagged trees
    def predict(self, X):
        if not self.trained:
            raise RuntimeError("The bagged forest classifier hasn't been trained yet")
        #get predictions from each tree
        #combine predictions into one matrix
        #get the mode of predictions for each sample
        prediction_matrix = np.zeros((len(X), self.n_trees))
        for t in xrange(self.n_trees):
            pred = self.trees[t].predict(X)
            prediction_matrix[:,t] = pred
        final_vote = stats.mode(prediction_matrix, axis=1)[0]

        return final_vote.flatten()

    # evaluate() is built on top of predict() to also score the generated prediction
    # the methods are the same as with the individual tree
    # the default method is the F1 score
    # alternatives are classification accuracy and Matthews correlation coefficient
    def evaluate(self, X, y, method = 'f1'):
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

    # cross_val() implements cross validation for training Bagged Forests
    # for each fold (default = 1), it splits the input dataset X,y by the
    # split parameter (default = 0.3), trains the Bag on the training split
    # and evaluates it on the cross-val split, using the provided method
    # (defaults to F1)
    def cross_val(self, X, y, split = 0.3, method = 'f1', folds = 1):
        indices = np.arange(len(X))
        set_ind = set(indices)
        size = np.int(len(X)*(1-split))
        scores = np.zeros(folds)
        for f in xrange(folds):
            train = np.random.choice(indices, size, replace=False)
            set_train = set(train)
            set_test = list(set_ind.difference(set_train))
            Xtrain = X[train, :]
            ytrain = y[train]
            Xtest = X[set_test, :]
            ytest = y[set_test]
            self.train(Xtrain,ytrain)
            scores[f] = self.evaluate(Xtest, ytest, method)
            print(scores[f])
        return scores

