import numpy as np

from node import Leaf
import tools.tools as tools
import pruning_algorithms.pruning as pruning

class DecisionTree:
    def __init__(self, split_function):
        self.split_function = split_function
        self.tree = None
        
    def predict(self, X):
        node = self.tree
        predictions = []
        for record_idx, _ in X.iterrows():
            if isinstance(node, Leaf):
                predictions.append(node.label)
            else: 
                predictions.append(node.predict(X.loc[record_idx]))
        return predictions

    def score(self, X, y):
        if len(y) == 0:
            print('Warning: the data is empty (size = 0)')
            return 0
        return self.tree.get_accuracy(X, y)

    def prune(self, X, y, metric='rep'):
        if metric == 'rep':
            self.tree = pruning.reduced_error_pruning(self.tree, X, y)
    
    def select_attribute(self, X, y, attribute_list,
            metric='naive'):
        """
        @return
            - Continuous data -> (best_attribute, split_candidate)
            - categorical data -> (best_attribute, None)
        """
        # If only one attribute is left and the data is continuous
        # than return the attribute
        if len(attribute_list) == 1 and \
                not tools.is_continuous(X[attribute_list[0]]):
            return (attribute_list[0], None)
        split_list = []
        for attribute in attribute_list:
            # TODO: split_functions should have a default metric argument
            split_score, candidate_split = \
                    self.split_function(X[attribute], y, metric)
            split_list.append(
                    (attribute, split_score, candidate_split))
        # Get best attribute item
        best_attribute_item = max(split_list, key=lambda x: x[1])
        return best_attribute_item[0], best_attribute_item[2]
        
    def split_categorical_data(self, X, attribute): 
        branch_data_list = []; attribute_value_list = []
        for attribute_val, branch_data in X.groupby(X[attribute]):
            attribute_value_list.append(attribute_val)
            branch_data_list.append(branch_data)
        return branch_data_list, attribute_value_list

    def get_clean_branchs(self, data_list, val_list, y):
        """ Joins together the Leaf branchs with the same label
        Eg:
            - Input:
                - a: val = 0, label = 0
                - b: val = 1, label = 0
                - c: val = 2, label = 1
            - Output:
                - a: val = [0,1], label = 0
                - c: val = [2], label = 1
        @return:
            a list of tuple (X, y, attribute_value) for each branch
        """
        branch_list = []
        branch_dict = {}
        for branch, attribute_val in zip(data_list, val_list):
            branch_labels = y.loc[branch.index]
            item = (branch, branch_labels, [attribute_val])
            labels = list(set(branch_labels))
            if len(labels) == 1:
                # Leaf branch:
                label = labels[0]
                if label in branch_dict:
                    # Label already exists, we update only
                    # the attribute value, X is not important because
                    # it's a leaf node
                    aux = branch_dict[label]
                    aux[2].append(attribute_val)
                    branch_dict[label] = aux
                else:
                    branch_dict[label] = item
            else:
                branch_list.append(item)
        branch_list.extend(branch_dict.values())
        return branch_list

    def clean_tree(self, node):
        if isinstance(node, Leaf):
            return node
        for question, son in node.sons:
            pass

