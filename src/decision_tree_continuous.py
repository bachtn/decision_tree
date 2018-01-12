import numpy as np
import pandas as pd
from sklearn import svm

from decision_tree import DecisionTree
from node import Node, Leaf, Question
import tools.tools as tools
import split_metrics.gini_impurity as gini

class DecisionTreeContinuous(DecisionTree):
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
                super(DecisionTreeContinuous, self).prune(X_val, y_val, metric)

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
            base_class = super(DecisionTreeContinuous, self)
            best_attribute, is_continuous = base_class.select_attribute(
                    X, y, attribute_list, metric='naive')
            # Create node
            root = Node()
            # Update the attribute list
            attribute_list.remove(best_attribute)
            if is_continuous:
                # Continuous data -> Train an SVM classifier
                clf, X_list, y_list, class_list = self.__split_continuous_data(X, y)
                for data, labels, c in zip (X_list, y_list, class_list):
                    root.add_son(Question(is_continuous, best_attribute, c, clf),
                        Leaf(c))
                    #root.add_son(Question(is_continuous, best_attribute, c, clf),
                    #    self.__build_tree(data, labels, attribute_list))
            else:
                # Categorical data
                # get the branchs and their attribute_value
                data_list, val_list = base_class.split_categorical_data(
                        X, best_attribute)
                branch_data = base_class.get_clean_branchs(data_list, val_list, y)
                # Add branchs to the node
                for data, labels, attribute_val_list in branch_data:
                    root.add_son(Question(
                        is_continuous, best_attribute, attribute_val_list),
                        self.__build_tree(data, labels, attribute_list))
            # Return node
            return root

    def __split_continuous_data(self, X, y):
        clf = svm.LinearSVC()
        clf.fit(X, list(y))
        predicted_classes = clf.predict(X)
        X_list = []; y_list = []; class_list = []
        for c in np.unique(predicted_classes):
            sample_idxs = np.where(predicted_classes == c)
            # we have the index of the sample in the array not in the
            # dataframe -> use iloc instead of loc
            X_list.append(X.iloc[sample_idxs])
            y_list.append(y.iloc[sample_idxs])
            class_list.append(c)
        return clf, X_list, y_list, class_list
