import tools.tools as tools
import split_metrics.entropy as entropy
import split_metrics.gini_impurity as gini

# All datasets should be in the dataset folder
dataset_dict = {
    'tennis': {'label': 'PlayTennis', 'filename': 'play_tennis.csv', 'delimiter': ','},
    'tennis_continuous': {'label': 'PlayTennis', 'filename': 'tennis_continuous.csv', 'delimiter': ','},
    'mushrooms' : {'label': 'class', 'filename': 'mushrooms.csv', 'delimiter': ','},
    'iris' : {'label': 'class', 'filename': 'iris.csv', 'delimiter': ','},
    'chess' : {'label': 'result', 'filename': 'chess.csv', 'delimiter': ','},
    'abalone' : {'label': 'Rings', 'filename': 'abalone.csv', 'delimiter':','},
    'mammal' : {'label': 'is_mammal', 'filename': 'mammal.csv', 'delimiter':','},
    'income' : {'label': 'income', 'filename': 'adult.csv', 'delimiter': ','}
}

def test_gini_impurity():
    for dataset_name in ['mushrooms', 'iris']:
        data, label, attribute_list = tools.get_dataset(dataset_name, dataset_dict)
        attribute_vector = data[attribute_list[0]]
        target_vector = data[label]
        print(gini.get_gini_split(target_vector, attribute_vector))



def test_entropy():
    data, label, attribute_list = tools.get_dataset('mushrooms', dataset_dict)
    attribute_vector = data[attribute_list[0]]
    target_vector = data[label]
    information_gain = entropy.get_information_gain(attribute_vector, target_vector)
    print(information_gain)
    assert(information_gain == 0.0487967019354)
    information_gain_ratio = entropy.get_information_gain_ratio(attribute_vector, target_vector)
    assertEqual(information_gain_ratio == 0.0295220698859)

if __name__=='__main__':
    data, label, attribute_list = tools.get_dataset('mushrooms', dataset_dict)
    attribute_vector = data[attribute_list[0]]
    target_vector = data[label]
    print("Testing Entropy :")
    #test_entropy()
    print("Testing Gini impurity :")
    test_gini_impurity()
