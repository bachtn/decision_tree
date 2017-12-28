import numpy as np
import pandas as pd
import math

from tools.tools import is_continuous

def get_vector_entropy(data_vector):
    """ returns the entropy the vector
    @param data_vector (dataframe column)
    @return float
    Note: if the vetor_data contains continous data than a warning is raised
    """
    # Check that the values are categorical
    # Todo take into account the categorical values with numbers (1, 2, 4 ...)
    if is_continuous(data_vector):
        raise Warning('Entropy can be computed only for categorical data vectors.')
    
    vector_entropy = 0
    n = len(data_vector)
    for nb_occur in data_vector.value_counts():        
        prob =  nb_occur / n
        vector_entropy -= prob * math.log2(prob)
    return vector_entropy

def get_partition_entropy(attribute_vector, target_vector):
    """ Returns the partition entropy of the attribute
    @param target_vector (dataframe column) the target vector (labels of dataset)
    @param attribute_vector (dataframe column) the candidate attribute vector
    """
    # Todo take into account the categorical values with numbers (1, 2, 4 ...)
    if is_continuous(attribute_vector):
        raise Warning('Entropy can be computed only for categorical data vectors.')
    
    partition_entropy = 0
    n = len(attribute_vector)
    for value, nb_occur in attribute_vector.value_counts().items():
        prob =  nb_occur / n
        val_target_vector = target_vector[attribute_vector == value]
        partition_entropy += prob * get_vector_entropy(val_target_vector)
    return partition_entropy

def get_information_gain(attribute_vector, target_vector):
    """ Returns the information gain if we choose to split the data
    on the candidate attribute.
    
    @param target_vector (dataframe column) the target vector (labels of dataset)
    @param attribute_vector (dataframe column) the candidate attribute vector
    """
    target_entropy = get_vector_entropy(target_vector)
    partition_entropy = get_partition_entropy(attribute_vector, target_vector)
    return target_entropy - partition_entropy

def get_information_gain_ratio(attribute_vector, target_vector):
    information_gain = get_information_gain(attribute_vector, target_vector)
    split_entropy = get_vector_entropy(attribute_vector)
    return information_gain / split_entropy
