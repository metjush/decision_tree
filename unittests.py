__author__ = 'metjush'

from classtree import *
import numpy as np

# This code runs unit tests on all class methods of the ClassificationTree and Node classes

# Node class tests
# First test: initialize a node
def node_test_init(feature, threshold, parent, level, entropy):
    n = Node(feature, threshold, parent, level, entropy)
    print("Entropy of node is %f" % n.entropy)
    print("Level of node is %f" % n.level)
    if n.terminal:
        print("Node is terminal and the result class is %d" % n.outcome[0])
    else:
        print("Node splits the sample at threshold %f into outcomes %d and %d" % (n.threshold, n.outcome[0], n.outcome[1]))
    return n

#second test: describe a node
def node_test_describe(node):
    node.describe()
    return node

#third test: make the node decide
def node_test_decide(node, value):
    outcome = node.decide(value)
    return outcome

#fourth test: make the node terminal
def node_test_terminal(node,class_label):
    node.make_terminal(class_label)
    return node

## Run the node tests
# Initialization tests
# OK branch node
branch = node_test_init(1, 0.5, None, 0, 0.4)
# OK terminal node
term = node_test_init(None, None, branch, 1, 0.)
#wrong threshold
wrong_t = node_test_init(2, "what", branch, 1, 1.3)
#wrong feature
wrong_f = node_test_init("whatever", 0.3, branch, 2, 0.3)
#wrong parent
wrong_p = node_test_init(0,0., 13, 1, 0.0)





