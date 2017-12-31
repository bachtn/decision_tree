import operator

from tools.tools import is_continuous

def get_gini_index(target_vector):
    accuracy = 0
    n = len(target_vector)
    for val_count in target_vector.value_counts():
        prob = val_count / n
        accuracy += prob ** 2
    return 1 - accuracy

def get_gini_split(attribute_vector, target_vector):
    """ Returns the gini split score
    @return:
        - data = continuous  -> (split_score, split_candidate)
        - data = categorical -> (split_score, None)
    """
    candidate_split = None
    parent_gini_idx = get_gini_index(target_vector)
    if is_continuous(attribute_vector):
        records_gini_idx, candidate_split = get_continuous_gini_index(
                attribute_vector, target_vector, metric='brute_force')
    else:
        records_gini_idx = get_categorical_gini_index(
                attribute_vector, target_vector)
    gini_split = parent_gini_idx - records_gini_idx
    return (gini_split, candidate_split)

def get_categorical_gini_index(attribute_vector, target_vector):
    n = len(attribute_vector)
    records_gini_idx = 0
    for value, nb_occur in attribute_vector.value_counts().items():
        current_record_gini_idx = get_gini_index(target_vector[attribute_vector == value])
        prob =  nb_occur / n
        records_gini_idx += prob * current_record_gini_idx
    return records_gini_idx

def get_continuous_gini_index(attribute_vector, target_vector,
        metric='naive'):
    if metric == 'naive':
        records_gini_idx, candidate_split = naive_gini(
                attribute_vector, target_vector)
    elif metric == 'brute_force':
        records_gini_idx, candidate_split = brute_force_gini(
                attribute_vector, target_vector)
    else:
        records_gini_idx, candidate_split = naive_gini(
                attribute_vector, target_vector)
    return records_gini_idx, candidate_split

def brute_force_gini(attribute_vector, target_vector):
    split_candidates = get_split_candidates(attribute_vector.values)
    candidates_gini_score = []
    for candidate in split_candidates:
        candidate_gini_index = get_candidate_gini_index(
                attribute_vector, target_vector, candidate)
        candidates_gini_score.append(candidate_gini_index)
    best_gini_idx = min(candidates_gini_score)
    best_candidate_split = split_candidates[
            candidates_gini_score.index(best_gini_idx)]
    return best_gini_idx, best_candidate_split

def naive_gini(attribute_vector, target_vector):
    """ Return the median as a candidate_split """
    candidate_split = attribute_vector.median()
    records_gini_idx = get_candidate_gini_index(
            attribute_vector, target_vector, candidate_split)
    return records_gini_idx, candidate_split

def get_candidate_gini_index(attribute_vector, target_vector, candidate_split):
    records_gini_idx = 0
    n = len(attribute_vector)
    operators = [operator.le, operator.gt]
    for op in operators:
        record_gini_idx = get_gini_index(target_vector[op(attribute_vector, candidate_split)])
        prob = len(attribute_vector[op(attribute_vector, candidate_split)]) / n
        records_gini_idx += prob * record_gini_idx
    return records_gini_idx

def get_split_candidates(values):
    """ 
    - [1,5,3]
    - Sort the values -> [1,3,5]
    - For each pair of values, get the mid value -> [2,4]
    """
    split_candidates = []
    values.sort()
    for i in range(len(values) - 1):
        mid = (values[i] + values[i+1]) / 2
        split_candidates.append(mid)
    return split_candidates
