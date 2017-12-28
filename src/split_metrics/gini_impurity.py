import operator

from tools.tools import is_continuous

def get_gini_split(attribute_vector, target_vector):
    """ Returns the gini split score
    If the data is continuous, then it returns the best split candidate
    with its gini split score
    Otherwise, it returns only the split score
    """
    parent_gini_idx = get_gini_index(target_vector)
    n = len(attribute_vector)
    records_gini_idx = 0
    if is_continuous(attribute_vector):
        candidate_split, gini_split = get_best_candidate_split(attribute_vector, target_vector, parent_gini_idx)
        return gini_split, candidate_split
    else:
        for value, nb_occur in attribute_vector.value_counts().items():
            current_record_gini_idx = get_gini_index(target_vector[attribute_vector == value])
            prob =  nb_occur / n
            records_gini_idx += prob * current_record_gini_idx
    
        gini_split = parent_gini_idx - records_gini_idx
    return gini_split

def get_gini_index(target_vector):
    accuracy = 0
    n = len(target_vector)
    for val_count in target_vector.value_counts():
        prob = val_count / n
        accuracy += prob ** 2
    return 1 - accuracy

def get_split_candidates(values):
    possible_thresholds = []
    values.sort()
    for i in range(len(values) - 1):
        mid = (values[i] + values[i+1]) / 2
        possible_thresholds.append(mid)
    return possible_thresholds


def get_gini_index_continuous(attribute_vector, target_vector, candidate_split):
    records_gini_idx = 0
    n = len(attribute_vector)
    operators = [operator.le, operator.gt]
    for op in operators:
        record_gini_idx = get_gini_index(target_vector[op(attribute_vector, candidate_split)])
        prob = len(attribute_vector[op(attribute_vector, candidate_split)]) / n
        records_gini_idx += prob * record_gini_idx
    return records_gini_idx

def get_best_candidate_split(attribute_vector, target_vector, parent_gini_idx):
    candidates_gini_split = []
    split_candidates = get_split_candidates(attribute_vector.values)
    for candidate in split_candidates:
        gini_idx = get_gini_index_continuous(attribute_vector, target_vector, candidate)
        gini_split = parent_gini_idx - gini_idx
        candidates_gini_split.append(gini_split)
    best_gini_split = max(candidates_gini_split)
    best_candidate_split = split_candidates[candidates_gini_split.index(best_gini_split)]
    return best_candidate_split, best_gini_split
