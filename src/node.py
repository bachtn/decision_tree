import numpy as np
import operator

import tools as tools

class Node:
    def __init__(self, attribute):
        """
        - categorical-data -> mid = None
        - continuous-data -> mid = candidate-split
        """
        self.attribute = attribute
        self.mid = None
        self.sons = []

    def add_son(self, question, node):
        self.sons.append((question, node))

    def predict(self, X):
        for question, son in self.sons:
            if question.match(X):
                if isinstance(son, Leaf):
                    return son.label
                else:
                    return son.predict(X)

    def get_accuracy(self, X, y):
        if y.shape[0] == 0:
            return 0
        y = np.array(y)
        predictions = np.array(self.get_predictions(X))
        return (predictions == y).sum() / len(y)

    def get_predictions(self, X):
        predictions = []
        for record_idx, _ in X.iterrows():
            if isinstance(self, Leaf):
                predictions.append(self.label)
            else:
                predictions.append(self.predict(X.loc[record_idx]))
        return predictions

class Leaf:
    def __init__(self, label):
        self.label = label

    def get_accuracy(self, X, y):
        predictions = np.array([self.label] * len(y))
        return (predictions == y).sum() / len(y)
        
    
    def __repr__(self):
        return "--> Predicted value : {}".format(self.label)


class Question:
    def __init__(self, attribute, value_list, op):
        """ If the data is continuous than value_list will contain
        only one element otherwise one or multiple element """
        self.attribute = attribute
        self.value_list = value_list
        self.op = op

    def match(self, X):
        val = X[self.attribute]
        if self.op == operator.eq:
            # Categorical data
            return val in self.value_list
        else:
            # Continuous data
            return self.op(val, self.value_list[0])

    def __repr__(self):
        rep =''
        if self.op == operator.eq:
            if len(self.value_list) == 1:
                rep = str(self.value_list[0])
            else:
                rep = "["
                for idx, val in enumerate(self.value_list):
                    aux = ', ' if idx < len(self.value_list) - 1 else ''
                    if tools.tools.is_numeric(val):
                        rep += str(round(float(val), 2)) + aux
                    else:
                        rep += str(val) + aux
                rep += "]"
        else:
            val = round(float(self.value_list[0]), 2)
            x = "<=" if self.op == operator.le else ">"
            rep = str(self.attribute) + " " + x + " " + str(val)
        return rep
