import numpy as np
import pandas as pd
import operator

from decision_tree import DecisionTree
from node import Node, Leaf, Question, QuestionClassic
import tools.tools as tools
import split_metrics.gini_impurity as gini
import pruning_algorithms.pruning as pruning

class DecisionTreeClassic(DecisionTree):
    def __init__(self, split_function=gini.get_gini_split, stop_threshold=0.9):
        DecisionTree.__init__(self, split_function)
        self.stop_threshold = stop_threshold

    def fit(self, X, y,
            prune=False, metric='rep', X_val=None, y_val=None):
        self.attribute_list = list(X.columns)
        self.tree = self.__build_tree(X, y, self.attribute_list.copy())
        if prune:
            if X_val is None or y_val is None:
                raise ValueError("To prune the tree, you need \
                                  to give the validation set")
            else:
                super(DecisionTreeClassic, self).prune(X_val, y_val, metric)
    
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
            # This condition is stated here because of a bug in pandas when
            # using valu_counts
            try:
                percent = y.value_counts()[0] / len(y)
                if percent >= self.stop_threshold:
                    label = tools.get_majority_vote(y)
                    return Leaf(label)
            except Exception as e:
                print(str(e))
            # Normal condition
            base_class = super(DecisionTreeClassic, self)
            best_attribute, candidate_split = base_class.select_attribute(
                    X, y, attribute_list, metric='naive')
            # Create node
            root = Node()
            # Update the attribute list
            attribute_list.remove(best_attribute)
            if candidate_split is not None:
                # Continuous data
                # Get less_or_equal and greater_than branchs
                le_branch, gt_branch = self.__split_continuous_data(
                        X, best_attribute, candidate_split)
                # Get branchs labels
                y_le = y.loc[le_branch.index]
                y_gt = y.loc[gt_branch.index]
                # Add branchs
                # Less or equal branch
                root.add_son(QuestionClassic(
                   best_attribute, [candidate_split], operator.le),
                   self.__build_tree(le_branch, y_le, attribute_list))
                # Greater than branch
                root.add_son(QuestionClassic(
                   best_attribute, [candidate_split], operator.gt),
                   self.__build_tree(gt_branch, y_gt, attribute_list))
            else:
                # Categorical data
                # get the branchs and their attribute_value
                data_list, val_list = self.split_categorical_data(
                        X, best_attribute)
                branch_data = self.get_clean_branchs(data_list, val_list, y)
                # Add branchs to the node
                for branch, labels, attribute_val_list in branch_data:
                    root.add_son(QuestionClassic(
                        best_attribute, attribute_val_list, operator.eq),
                        self.__build_tree(branch, labels, attribute_list))
            # Return node
            return root

    def __split_continuous_data(self, X, attribute, candidate_split):
        le_branch = X[operator.le(X[attribute], candidate_split)]
        gt_branch = X[operator.gt(X[attribute], candidate_split)]
        return le_branch, gt_branch


