import pandas as pd
import numpy as np
from graphviz import Digraph
import uuid
from numbers import Number

from node import Node, Leaf, Question

def train_test_split(X, y, train_percent, test_percent,
        display_info=False):
    # offsets
    n = X.shape[0]
    train_offset = int(train_percent*len(X))
    test_offset = int(len(X) - test_percent*len(X))
    # X
    X_train, X_validation, X_test = \
            np.split(X.sample(frac=1), [train_offset, test_offset])
    # y
    y_train = y.loc[X_train.index]
    y_validation = y.loc[X_validation.index]
    y_test = y.loc[X_test.index]
    
    if display_info:
        print("Data : ", len(X), ", Train : ", len(y_train),
                ", Test : ", len(y_test), ", Validation : ",
                len(y_validation))
    
    return (X_train, y_train), (X_validation, y_validation), \
            (X_test, y_test)

def get_majority_vote(target_vector):
    """ Return the class that occurs the most """
    return target_vector.value_counts().idxmax()

def is_continuous(data_vector, numeric_threshold=4):
    """ Returns True if the `data_vector` contains continuous values
    @param data_vector (dataframe column)
    @param numeric_threshold (int) the threshold value to detect
        if a numeric column is continuous or categorical
        
    Note: if a column contains numerical data, we check
        its different possible values, if the number is greater than
        the `numeric_threshold` than the column values are
        continuous otherwise they are categorical.
        
        Problems: If we classify a column as categorical and we have
            some values that are not present in the training data,
            we can never classify the rows corresponding to these
            values because we do not have a branch corresponding
            to these values in the constructed decision tree.
    """
    nb_unique_values = len(data_vector.value_counts())
    return (is_numeric(data_vector.iloc[0]) and
            nb_unique_values > numeric_threshold)

def generate_tree_graph(tree, labels,
        graph_name="decision_tree_train", display=False):
    dot = Digraph()
    root_node_id = uuid.uuid4()
    label_color_dict = get_label_colors(labels)
    generate_tree_graph_aux(dot, tree, root_node_id, label_color_dict)
    if display:
        dot.render('tree_graph/' + graph_name, view=True)
    else:
        dot.render('tree_graph/' + graph_name, view=False)
    return dot



def generate_tree_graph_aux(dot, node, root_node_id,
        label_color_dict):
    if isinstance(node, Leaf):
        color = label_color_dict[node.label]
        dot.node(str(uuid.uuid4()), str(node.label),
                style="filled", fillcolor=color, color=color)
    else:
        dot.node(str(root_node_id), node.attribute)
        for question, son in node.sons:
            node_id = uuid.uuid4()
            if isinstance(son, Leaf):
                color = label_color_dict[son.label]
                dot.node(str(node_id), str(son.label),
                        style="filled", fillcolor=color, color=color)
                dot.edge(str(root_node_id), str(node_id),
                        label=str(question.value))
            else:
                generate_tree_graph_aux(dot, son, node_id,
                        label_color_dict)
                dot.edge(str(root_node_id), str(node_id),
                        label=str(question.value))

def get_label_colors(labels):
    node_colors = ['aquamarine', 'bisque', 'azure3', 'brown1',
        'burlywood1', 'cadetblue3', 'chartreuse', 'chocolate1',
        'coral', 'cornflowerblue', 'darkgoldenrod1', 'darkgreen',
        'darkolivegreen1', 'darkorange1', 'darkturquoise',
        'deeppink', 'dimgray', 'gold3', 'gray42', 'greenyellow']
    label_dict = {}
    for idx, label in enumerate(labels):
        label_dict[label] = node_colors[idx] \
                if idx < len(node_colors) else node_colors[0]
    return label_dict


def is_numeric(value):    
    return isinstance(value, Number)

def get_dataset(dataset_name, datasets_dict):
    """ Returns the dataframe, name of the label column
    and a list with the column name of the attributes.
    
    @param dataset_name (str): the name of the dataset
        in the previously defined dict (dataset_dict)
    
    @param dataset_dict (dict): A dictionary with all
        the datasets informations.    
    """
    dataset_info = datasets_dict[dataset_name]
    path = '../datasets/' + dataset_info['filename']
    
    data = pd.read_csv(path, delimiter=dataset_info['delimiter'])
    label = dataset_info['label']
    attribute_list = list(data.columns)
    attribute_list.remove(label)
    return data, label, attribute_list

class ObjectView():
    """
    Used to access dictionary items as object attributes
    d = {'a': 1, 'b': 2}
    o = objectview(d)
    assert o.a == 1
    """
    def __init__(self, d):
        """d : dictionary"""
        self.__dict__ = d

def print_tree(node, spacing=""):
    if isinstance(node, Leaf):
        print(spacing, node, "\n")
        return
    
    for question, son in node.sons:
        if question:
            print(spacing, question)
        print_tree(son, " ----")
