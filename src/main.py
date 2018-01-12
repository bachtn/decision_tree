import tools.tools as tools
import split_metrics.entropy as entropy
import split_metrics.gini_impurity as gini
from decision_tree_classic import DecisionTreeClassic
from decision_tree_continuous import DecisionTreeContinuous
import tools2

# All datasets should be in the dataset folder
dataset_dict = {
        'tennis': {'label': 'PlayTennis', 'filename': 'play_tennis.csv'},
        'mushrooms' : {'label': 'class', 'filename': 'mushrooms.csv'},
        'iris' : {'label': 'class', 'filename': 'iris.csv'},
        'spambase' : {'label': 'spam', 'filename': 'spambase.csv'},
        'magic_telescope' : {'label': 'class', 'filename': 'magic.csv'},
        'pima' : {'label': 'class', 'filename': 'pima.csv'}
        }


from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

models = [DecisionTreeClassic(gini.get_gini_split),
        DecisionTreeContinuous(gini.get_gini_split),
        DecisionTreeClassifier(), svm.SVC(),
        GaussianNB(),
        LinearDiscriminantAnalysis(n_components=2)]

if __name__=='__main__':
    train_acc, test_acc = tools2.benchmark_models(models, dataset_dict,
            dataset_name='iris', nb_folds=10)
    #test_tree(dtype='continuous', dataset='iris')
    continuous = False; train_percent = .8; test_percent = .2
    split_function = gini.get_gini_split; stop_threshold = 0.9
    dataset = 'iris'
    data, label, attribute_list = \
            tools.get_dataset(dataset, dataset_dict)
    labels = data[label].unique()
    train, validation, test = tools.train_test_split(
            data[attribute_list], data[label], train_percent, test_percent)
    decision_tree = DecisionTreeContinuous(split_function) if continuous \
            else DecisionTreeClassic(split_function)
    decision_tree.fit(train[0], train[1])
    print("Score before pruning : ", decision_tree.score(test[0], test[1]))
    tools.generate_tree_graph(decision_tree.tree, labels, display=True)
    decision_tree.prune(validation[0], validation[1])
    print("Score after pruning : ", decision_tree.score(test[0], test[1]))
    tools.generate_tree_graph(decision_tree.tree, labels,
            graph_name="decision_tree_prune")

