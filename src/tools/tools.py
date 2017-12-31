import pandas as pd
from numbers import Number

def is_numeric(value):    
    return isinstance(value, Number)

def is_continuous(data_vector, numeric_threshold=5):
    """ Returns True if the `data_vector` contains continuous values
    @param data_vector (dataframe column)
    @param numeric_threshold (int) the threshold value to detect
        if a numeric column is continuous or categorical
        
    Note: if a column contains numerical data, we check its different
        possible values, if the number is greater than
        the `numeric_threshold` than the column values are continuous
        otherwise they are categorical.
        
        Problems: If we classify a column as categorical and we have
            some values that are not present in the training data,
            we can never classify the rows corresponding to these values
            because we do not have a branch corresponding to these values
            in the constructed decision tree.
    """
    return (is_numeric(data_vector.iloc[0]) and 
            len(data_vector.value_counts()) > numeric_threshold)


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
