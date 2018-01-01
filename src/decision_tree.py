import numpy as np
import pandas as pd
import operator
import copy

from node import Node, Leaf, Question
import tools.tools as tools
import pruning_algorithms.pruning as pruning

class DecisionTree:
    def __init__(self, split_function):
        self.split_function = split_function
        self.tree = None
        
    def fit(self, X, y,
            prune=False, metric='rep', X_val=None, y_val=None):
        self.attribute_list = list(X.columns)
        self.tree = self.__build_tree(X, y, self.attribute_list.copy())
        if prune:
            if X_val is None or y_val is None:
                raise ValueError("To prune the tree, you need \
                                  to give the validation set")
            else:
                self.prune(X_val, y_val, metric)


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
        return self.tree.get_accuracy(X, y)
        """
        if y.shape[0] == 0:
            return 0
        y = np.array(y)
        predictions = np.array(self.predict(X))
        return (predictions == y).sum() / y.shape[0]
        """

    def prune(self, X, y, metric='rep'):
        if metric == 'rep':
            self.tree = pruning.reduced_error_pruning(self.tree, X, y)

    def __build_tree(self, X, y, attribute_list):
        # Only one class left
        left_classes = set(y)
        if len(left_classes) == 1:
            label = list(left_classes)[0]
            return Leaf(label)
        elif not attribute_list:
            # No attribute left to split on, Get most probable class
            label = tools.get_majority_vote(y)
            return Leaf(label)
        else:
            best_attribute, candidate_split = self.__select_attribute(
                    X, y, attribute_list, metric='naive')
            # Create node
            root = Node(best_attribute)
            # Update the attribute list
            attribute_list.remove(best_attribute)
            if candidate_split is not None:
                # Continuous data
                #TODO: if mid is not used? remove it?
                root.mid = candidate_split    
                # Get less_or_equal and greater_than branchs
                le_branch, gt_branch = self.__split_continuous_data(
                        X, best_attribute, candidate_split)
                # Get branchs labels
                y_le = y.loc[le_branch.index]
                y_gt = y.loc[gt_branch.index]
                # Add branchs
                # Less or equal branch
                root.add_son(Question(
                   best_attribute, [candidate_split], operator.le),
                   self.__build_tree(le_branch, y_le, attribute_list))
                # Greater than branch
                root.add_son(Question(
                   best_attribute, [candidate_split], operator.gt),
                   self.__build_tree(gt_branch, y_gt, attribute_list))
            else:
                # Categorical data
                # get the branchs and their attribute_value
                data_list, val_list = self.__split_categorical_data(
                        X, best_attribute)
                branch_data = self.__get_clean_branchs(data_list, val_list, y)
                # Add branchs to the node
                for branch, labels, attribute_val_list in branch_data:
                    root.add_son(Question(
                        best_attribute, attribute_val_list, operator.eq),
                        self.__build_tree(branch, labels, attribute_list))
            # Return node
            return root

    def __get_clean_branchs(self, data_list, val_list, y):
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
        

    def __select_attribute(self, X, y, attribute_list,
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
        

    def __split_continuous_data(self, X, attribute, candidate_split):
        le_branch = X[operator.le(X[attribute], candidate_split)]
        gt_branch = X[operator.gt(X[attribute], candidate_split)]
        return le_branch, gt_branch

    def __split_categorical_data(self, X, attribute): 
        branch_data_list = []; attribute_value_list = []
        for attribute_val, branch_data in X.groupby(X[attribute]):
            attribute_value_list.append(attribute_val)
            branch_data_list.append(branch_data)
        return branch_data_list, attribute_value_list

