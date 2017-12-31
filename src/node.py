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

    def predict(self, data):
        for question, son in self.sons:
            if question.match(data):
                if isinstance(son, Leaf):
                    return son.label
                else:
                    return son.predict(data)

class Leaf:
    def __init__(self, label):
        self.label = label

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
            rep = "{} == {} ?".format(self.attribute, self.value)
        else:
            x = "<=" if self.op == operator.le else ">"
            rep = "%.2f %s %.2f" % (self.attribute, x, self.value)
        return rep
