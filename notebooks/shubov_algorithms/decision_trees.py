from .base_interface import BaseClass
from .helpers import *
import numpy as np


def gini(p):
    """
    p: class frequencies as numpy array with np.sum(p)=1
    returns: impurity according to gini criterion
    """
    return 1. - np.sum(p ** 2)


def entropy(p):
    """
    p: class frequencies as numpy array with np.sum(p)=1
    returns: impurity according to entropy criterion
    """
    idx = np.where(p == 0.)  # consider 0*log(0) as 0
    p[idx] = 1.
    r = p * np.log2(p)
    return -np.sum(r)


def misclass(p):
    """
    p: class frequencies as numpy array with np.sum(p)=1
    returns: impurity according to misclassification rate
    """
    return 1 - np.max(p)


def get_split_attribute(X, y, attributes, impurity, verbose=0):
    """
    :param X: data matrix n rows, d columns
    :param y: vector with n rows, 1 column containing the target concept
    :param attributes: A dictionary mapping an attribute's index to the attribute's domain
    :param impurity: impurity function of the form impurity(p_1....p_k) with k=|y.unique|
    :param verbose: numerical level of verbosity
    :returns: (1) idx of attribute with maximum impurity reduction and (2) impurity reduction
    """

    n, d = X.shape

    ir = [0.] * d
    for a_i in attributes.keys():
        ir[a_i] = impurity_reduction(X, a_i, y, impurity, verbose)
    if verbose:
        print("Impurity reduction for class attribute (ordered by attributes)", (pp_float_list(ir)))
    b_a_i = np.argmax(ir)
    return b_a_i, ir[b_a_i]


def impurity_reduction(X, a_i, y, impurity, verbose=0):
    """ For more readable code no assertion is being checked

    :param X: data matrix n rows, d columns
    :param a_i: column index of the attribute to evaluate the impurity reduction for
    :param y: concept vector with n rows and 1 column
    :param impurity: impurity function of the form impurity(p_1....p_k) with k=|X[a].unique|
    :param verbose: numerical level of verbosity
    :returns: impurity reduction

    """

    n, d = float(X.shape[0]), float(X.shape[1])

    y_v = np.unique(y)

    # Compute relative frequency of each class in X
    p = (1. / n) * np.array([np.sum(y == c) for c in y_v])
    # ..and corresponding impurity l(D)
    h_p = impurity(p)

    if verbose:
        print("\t Impurity %0.3f: %s" % (h_p, pp_float_list(p)))

    a_v = np.unique(X[:, a_i])

    # Create and evaluate splitting of X induced by attribute a_i
    # We assume nominal features and perform m-ary splitting
    h_pa = []
    for a_vv in a_v:
        mask_a = X[:, a_i] == a_vv
        n_a = float(mask_a.sum())

        # Compute relative frequency of each class in X[mask_a]
        pa = (1. / n_a) * np.array([np.sum(y[mask_a] == c) for c in y_v])
        h_pa.append((n_a / n) * impurity(pa))
        if verbose:
            print("\t\t Impurity %0.3f for attribute %d with value %s: " % (h_pa[-1], a_i, a_vv), pp_float_list(pa))

    ir = h_p - np.sum(h_pa)
    if verbose:
        print("\t Estimated reduction %0.3f" % ir)
    return ir


class DecisionNode(object):
    node_id = 0

    def __init__(self, attr=-1, children=None, label=None):
        self.attr = attr
        self.children = children
        self.label = label
        self.id = DecisionNode.node_id
        DecisionNode.node_id += 1


class DecisionTreeID3(BaseClass, object):
    def __init__(self, criterion, verbose=0):
        """
        :param criterion: The function to assess the quality of a split
        :param verbose: numerical level of verbosity
        """
        self.criterion = criterion
        self.root = None
        self.verbose = verbose

    def learn(self, features, targets):
        self.root = self._learn(features, targets, attributes=None)
        return self

    def _learn(self, features, targets, attributes=None):
        """
        :param features: numpy array of shape (N, d) with N being the number of samples and d being the number of
            feature dimensions
        :param targets: numpy array of shape (N, 1) with N being the number of samples as in the
            provided features and 1 being the number of target dimensions
        :param attributes: a dictionary mapping an attribute's index to the attribute's domain
        :return: a new decision node
        """
        # Set up temporary variables
        n, d = features.shape
        if attributes is None:
            attributes = {a_i: np.unique(features[:, a_i]) for a_i in range(d)}
        depth = d - len(attributes) + 1

        # if len(X) == 0: return DecisionNode()

        label, f_is_pure = most_common_class(targets)
        # Stop criterion 1: Node is pure -> create leaf node
        if f_is_pure:
            if self.verbose:
                print("\t\t Leaf Node with label %s due to purity." % label)
            return DecisionNode(label=label)

        # Stop criterion 2: Exhausted attributes -> create leaf node
        if len(attributes) == 0:
            if self.verbose:
                print("\t\t Leaf Node with label %s due to exhausted attributes." % label)
            return DecisionNode(label=label)

        # Get attribute with maximum impurity reduction
        a_i, a_ig = get_split_attribute(features, targets, attributes, self.criterion, verbose=self.verbose)
        if self.verbose:
            print("Level %d: Choosing attribute %d out of %s with gain %f" % (depth, a_i, attributes.keys(), a_ig))

        values = attributes.pop(a_i)
        splits = [features[:, a_i] == v for v in values]
        branches = {}

        for v, split in zip(values, splits):
            if not np.any(split):
                if self.verbose:
                    print("Level %d: Empty split for value %s of attribute %d" % (depth, v, a_i))
                branches[v] = DecisionNode(label=label)
            else:
                if self.verbose:
                    print("Level %d: Recursion for value %s of attribute %d" % (depth, v, a_i))
                branches[v] = self._learn(features[split, :], targets[split], attributes=attributes)

        attributes[a_i] = values
        return DecisionNode(attr=a_i, children=branches, label=label)

    def infer(self, features):
        def _infer(feature, node):
            """
            :param feature:
            :param node: node of the (sub)tree
            :return: predicted value for the provided feat
            """
            if not node.children:
                return node.label
            else:
                v = feature[node.attr]
                child_node = node.children[v]
                return _infer(feature, child_node)

        return [_infer(feature, self.root) for feature in features]

    def print_tree(self, ai2an_map, ai2aiv2aivn_map):
        """
        :param ai2an_map: list of attribute names
        :param ai2aiv2aivn_map: list of lists of attribute values,
            i.e. a value, encoded as integer 2, of attribute with index 3 has name ai2aiv2aivn_map[3][2]
        """

        def _print(node, test="", level=0):
            """
            :param node: node of the (sub)tree
            :param test: string specifying the test that yielded the node 'node'
            :param level: current level of the tree
            """

            prefix = ""
            prefix += " " * level
            prefix += " |--(%s):" % test
            if not node.children:
                print("%s assign label %s" % (prefix, ai2aiv2aivn_map[6][node.label]))
            else:
                print("%s test attribute %s" % (prefix, ai2an_map[node.attr]))
                for v, child_node in node.children.items():
                    an = ai2an_map[node.attr]
                    vn = ai2aiv2aivn_map[node.attr][v]
                    _print(child_node, "%s=%s" % (an, vn), level + 1)

        return _print(self.root)

    def depth(self):
        """
        :returns: depth of the tree
        """

        def _depth(node):
            """
            :param node: node of the (sub)tree
            :returns: depth of the provided node
            """
            if not node.children:
                return 0
            else:
                return 1 + max([_depth(child_node) for child_node in node.children.values()])

        return _depth(self.root)
