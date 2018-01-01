import copy

import tools.tools as tools
from node import Node, Leaf, Question

def reduced_error_pruning(current_node, X, y):
    """ X, y are validation sets """
    if isinstance(current_node, Leaf):
        return current_node

    current_accuracy = current_node.get_accuracy(X, y)
    node_copy = copy.deepcopy(current_node)
    majority_label = tools.get_majority_vote(y)
    current_node = Leaf(majority_label)
    new_accuracy = current_node.get_accuracy(X, y)
    if new_accuracy >= current_accuracy:
        return current_node
    else:
        current_node = node_copy
        # Son management
        node_sons = copy.deepcopy(current_node.sons)
        current_node.sons = []
        for question, son in node_sons:
            # If the node is a leaf, nothing to do
            if isinstance(son, Leaf):
                current_node.add_son(question, son)
            current_node.add_son(
                question, reduced_error_pruning(son, X, y))
        return current_node
