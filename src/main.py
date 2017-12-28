import tools.tools as tools
import split_metrics.entropy as entropy

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


def test_entropy():
    data, label, attribute_list = tools.get_dataset('mushrooms', dataset_dict)
    attribute_vector = data[attribute_list[0]]
    target_vector = data[label]
    print(entropy.get_information_gain(attribute_vector, target_vector))
    print(entropy.get_information_gain_ratio(attribute_vector, target_vector))

if __name__=='__main__':
    data, label, attribute_list = tools.get_dataset('mushrooms', dataset_dict)
    attribute_vector = data[attribute_list[0]]
    target_vector = data[label]
    test_entropy()
