__author__ = 'metjush'

# Node Class is the basic building block of the Classification Tree
# it implements decision rules and final assignment to classes

import numpy as np


class Node:

    # the object is initialized by telling the node along which feature it is splitting
    # what the split threshold is, what the parent node of this node is,
    # what level (starting with 0 as the root node) of the tree this is,
    # and what entropy was at this node before splitting
    def __init__(self, feature=None, threshold=None, parent=None, level=0, entropy=0.):
        #check arguments
        numbers = set({int, float, np.float64, np.int64})
        integers = set({int, np.int64})
        if (not isinstance(parent, Node)) and (parent is not None):
            raise TypeError("parent has to be a Node instance or set to None")
        if not type(level) in integers:
            raise TypeError("Index of level has to be an integer")
        if not type(entropy) in numbers:
            raise TypeError("Entropy has to be a numerical value")

        self.parent = parent
        self.entropy = entropy
        self.level = level

        if threshold is None:
            #if there is no threshold, it means this is a terminal node, assigning all to one class
            self.feature = None
            self.threshold = None
            self.outcome = [0]
        else:
            #check branch node arguments
            if not type(threshold) in numbers:
                raise TypeError("Threshold has to be a numerical value")
            if not type(feature) in integers:
                raise TypeError("Index of feature has to be an integer")

            self.feature = feature
            self.threshold = threshold
            #outcome of a node are the two child nodes
            # held in a list [Below, Above]
            self.outcome = [0,0]
        #determine if this node is terminal
        self.terminal = self.feature is None

    def make_terminal(self, class_label):
        self.feature = None
        self.threshold = None
        self.outcome = [class_label]
        self.terminal = True

    def decide(self, value):
        #if this is a terminal node, there is only one outcome
        if self.terminal:
            return self.outcome[0]
        #move an incoming value down either of the two leaves
        if value < self.threshold:
            return self.outcome[0]
        else:
            return self.outcome[1]

    def describe(self):
        if self.terminal:
            print("This is a terminal node that assigns class %d to samples that come here. \n Its level is %d and entropy is %f. \n" % (self.outcome[0], self.level, self.entropy))
        else:
            print("This is a branch node that splits on feature %d and at threshold %f. The level is %d and entropy is %f. \n" % (self.feature, self.threshold, self.level, self.entropy))

