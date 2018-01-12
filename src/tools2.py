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
              split_function=gini.get_gini_split, nb_features=10):
    # Set up data
    data, label, attribute_list = tools.get_dataset(dataset, dataset_dict,
            nb_features)
    train, validation, test = tools.train_test_split(
        data[attribute_list], data[label],
        train_percent=train_percent, test_percent=test_percent)
    # DTree
    decision_tree = DecisionTreeClassic(split_function) if dtype == 'classic' \
        else DecisionTreeContinuous(split_function)    
    decision_tree.fit(train[0], train[1])
    print("Train score : ", decision_tree.score(train[0], train[1]))
    print("Test score : ", decision_tree.score(test[0], test[1]))
    graph = tools.generate_tree_graph(decision_tree.tree, data[label].unique())
    return decision_tree.tree, graph
