import numpy as np
import operator

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
    def __init__(self, attribute, value, op):
        self.attribute = attribute
        self.value = value
        self.op = op

    def match(self, data):
        val = data[self.attribute]
        return self.op(val, self.value)

    def __repr__(self):
        rep =''
        if self.op == operator.eq:
            rep = "{} == {} ?".format(self.attribute,
                    round(self.value, 2))
        else:
            x = "<=" if self.op == operator.le else ">"
            rep = "%.2f %s %.2f" % (self.attribute, x,
                round(self.value, 2))
        return rep
