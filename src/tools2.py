from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from decision_tree_classic import DecisionTreeClassic
from decision_tree_continuous import DecisionTreeContinuous
from split_metrics  import gini_impurity as gini
import tools.tools as tools

dataset_dict = {
    'tennis': {'label': 'PlayTennis', 'filename': 'play_tennis.csv'},
    'mushrooms' : {'label': 'class', 'filename': 'mushrooms.csv'},
    'iris' : {'label': 'class', 'filename': 'iris.csv'},
    'spambase' : {'label': 'spam', 'filename': 'spambase.csv'},
    'magic_telescope' : {'label': 'class', 'filename': 'magic.csv'},
    'pima' : {'label': 'class', 'filename': 'pima.csv'}
}

def test_tree(dtype='classic', dataset='iris', train_percent=.6, test_percent=.2,
              split_function=gini.get_gini_split, prune=False,
              stop_threshold=0.9, nb_features=10):
    # Set up data
    data, label, attribute_list = tools.get_dataset(dataset, dataset_dict,
            nb_features)
    train, validation, test = tools.train_test_split(
        data[attribute_list], data[label],
        train_percent=train_percent, test_percent=test_percent)
    # DTree
    decision_tree = DecisionTreeClassic(split_function, stop_threshold) if dtype == 'classic' \
        else DecisionTreeContinuous(split_function, stop_threshold)
    if prune:
        decision_tree.fit(train[0], train[1],
                prune=True, X_val=test[0], y_val=test[1])
    else:
        decision_tree.fit(train[0], train[1])
    print("Train score : ", decision_tree.score(train[0], train[1]))
    print("Test score : ", decision_tree.score(test[0], test[1]))
    graph = tools.generate_tree_graph(decision_tree.tree, data[label].unique())
    return decision_tree.tree, graph

def benchmark_models(models, dataset_dict, nb_folds=5, dataset_name='iris'):
    data, label, attribute_list = tools.get_dataset(dataset_name, dataset_dict)
    data = tools.data_encoder(data)
        
    nb_models = len(models) 
    train_acc = np.zeros((nb_models, nb_folds))
    test_acc = np.zeros((nb_models, nb_folds))

    k_idx = 0
    kf = KFold(n_splits=nb_folds, shuffle=True, random_state=48)
    for train_idxs, test_idxs in kf.split(data):
        # Set up data
        X_train, X_test = data.iloc[train_idxs], data.iloc[test_idxs]
        y_train, y_test = X_train[label], X_test[label]
        X_train, X_test = X_train[attribute_list], X_test[attribute_list]
        # Train models
        model_idx = 0
        for model in models:
            if model_idx == 1:
                model.fit(X_train, y_train)
                y_train, y_test = np.array(list(y_train)), np.array(list(y_test))
            else:
                model.fit(X_train, y_train)
            train_acc[model_idx, k_idx] = model.score(X_train, y_train)
            test_acc[model_idx, k_idx] = model.score(X_test, y_test)
            model_idx += 1

        k_idx += 1
    return train_acc, test_acc

def custom_plot(models_dict, y_values, title):
    x_val = list(range(1, len(y_values[1]) + 1))
    for model_idx in models_dict.keys():
        plt.ylim(ymin=0., ymax=1.1)
        plt.plot(x_val, y_values[model_idx], label=models_dict[model_idx])
        plt.xticks(x_val)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.ylabel('Accuracy')
        plt.xlabel('fold index')
        plt.title(title, y=1.08, fontweight="bold")
    plt.grid()
    plt.show()

