import tools.tools as tools
import split_metrics.entropy as entropy
import split_metrics.gini_impurity as gini
from decision_tree import DecisionTree

# All datasets should be in the dataset folder
dataset_dict = {
    'tennis': {'label': 'PlayTennis', 'filename': 'play_tennis.csv'},
    'mushrooms' : {'label': 'class', 'filename': 'mushrooms.csv'},
    'iris' : {'label': 'class', 'filename': 'iris.csv'},
    'spambase' : {'label': 'spam', 'filename': 'spambase.csv'},
    'magic_telescope' : {'label': 'class', 'filename': 'magic.csv'},
    'pima' : {'label': 'class', 'filename': 'pima.csv'}
}

if __name__=='__main__':
    dataset = 'mushrooms'
    data, label, attribute_list = \
            tools.get_dataset(dataset, dataset_dict)
    labels = data[label].unique()
    train, validation, test = tools.train_test_split(
            data[attribute_list], data[label], .6, .2)
    decision_tree = DecisionTree(gini.get_gini_split)
    decision_tree.fit(train[0], train[1])
    print("Score before pruning : ", decision_tree.score(test[0], test[1]))
    tools.generate_tree_graph(decision_tree.tree, labels, display=True)
    decision_tree.prune(validation[0], validation[1])
    print("Score after pruning : ", decision_tree.score(test[0], test[1]))
    tools.generate_tree_graph(decision_tree.tree, labels,
            graph_name="decision_tree_prune")
