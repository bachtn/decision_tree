import tools.tools as tools
import split_metrics.entropy as entropy
import split_metrics.gini_impurity as gini
from decision_tree import DecisionTree

# All datasets should be in the dataset folder
dataset_dict = {
    'tennis': {'label': 'PlayTennis', 'filename': 'play_tennis.csv',
        'delimiter': ','},
    'tennis_continuous': {'label': 'PlayTennis',
        'filename': 'tennis_continuous.csv', 'delimiter': ','},
    'mushrooms' : {'label': 'class', 'filename': 'mushrooms.csv',
        'delimiter': ','},
    'iris' : {'label': 'class', 'filename': 'iris.csv',
        'delimiter': ','},
    'chess' : {'label': 'result', 'filename': 'chess.csv',
        'delimiter': ','},
    'abalone' : {'label': 'Rings', 'filename': 'abalone.csv',
        'delimiter':','},
    'mammal' : {'label': 'is_mammal', 'filename': 'mammal.csv',
        'delimiter':','},
    'income' : {'label': 'income', 'filename': 'adult.csv',
        'delimiter': ','}
}

if __name__=='__main__':
    dataset = 'iris'
    data, label, attribute_list = \
            tools.get_dataset(dataset, dataset_dict)
    train, validation, test = tools.train_test_split(
            data[attribute_list], data[label], .8, .2)
    decision_tree = DecisionTree(gini.get_gini_split)
    decision_tree.fit(train[0], train[1])
    print("Score : ", decision_tree.score(test[0], test[1]))
    tools.generate_tree_graph(decision_tree.tree, display=True)
