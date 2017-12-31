#!/usr/bin/env python3

import unittest
import numpy as np
import pandas as pd

import split_metrics.gini_impurity as gini
from tools.tools import get_dataset, ObjectView
from main import dataset_dict

class GiniImpurityTest(unittest.TestCase):
    def test_gini_index(self):
        target_vector = pd.Series([0,0,1,1,1,0,1,0,1,1,1,1,1,0])
        correct_gini_index = 1 - (9/14)**2 - (5/14)**2
        gini_index = gini.get_gini_index(target_vector)
        self.assertEqual(gini_index, correct_gini_index)

    def test_gini_split(self):
        # Test continuous data
        continuous_attribute_vector = pd.Series(
                [20,21,23,23,25,29,32,33])
        continous_target_vector = pd.Series([0,0,0,1,0,1,0,1])
        correct_candidate = 24; correct_gini_split = 0.03125
        gini_split, candidate = gini.get_gini_split(
                continuous_attribute_vector, continous_target_vector)
        self.assertEqual((gini_split, candidate),
                (correct_gini_split, correct_candidate))

        # Test categorical data
        categorical_attribute_vector =pd.Series([0,0,0,0,1,1,1,0,2,2])
        categorical_target_vector = pd.Series([0,0,0,0,1,1,1,2,2,2])
        correct_gini_split = 0.34
        gini_split, candidate = gini.get_gini_split(
              categorical_attribute_vector, categorical_target_vector)
        self.assertEqual((round(gini_split,2), candidate),
                (correct_gini_split, None))

        

    def test_categorical_gini_index(self):
        attribute_vector = pd.Series([0,0,0,0,1,1,1,0,2,2])
        target_vector = pd.Series([0,0,0,0,1,1,1,2,2,2])
        correct_gini_idx = 0.32
        gini_idx = gini.get_categorical_gini_index(
                attribute_vector, target_vector)
        self.assertEqual(round(gini_idx, 2), correct_gini_idx)

    def test_continuous_gini_index(self):
        attribute_vector = pd.Series([20,21,23,23,25,29,32,33])
        target_vector = pd.Series([0,0,0,1,0,1,0,1])
        # Test Naive metric
        naive_candidate = 24; naive_gini_score = 7/16
        gini_idx, candidate = gini.get_continuous_gini_index(
                attribute_vector, target_vector, metric='naive')
        self.assertEqual((gini_idx, candidate),
                (naive_gini_score, naive_candidate))
        
        # Test Brute force metric
        # TODO: set correct values
        bf_candidate = 24; bf_gini_score = 7/16
        gini_idx, candidate = gini.get_continuous_gini_index(
              attribute_vector, target_vector, metric='brute_force')
        self.assertEqual((gini_idx, candidate),
                (bf_gini_score, bf_candidate))

    def test_brute_force_gini(self):
        # TODO: test
        pass

    def test_naive_gini(self):
        attribute_vector = pd.Series([20,21,23,23,25,29,32,33])
        target_vector = pd.Series([0,0,0,1,0,1,0,1])
        median_candidate = 24
        median_gini_idx = gini.get_candidate_gini_index(
                attribute_vector, target_vector, median_candidate)
        gini_idx, candidate = gini.naive_gini(
                attribute_vector, target_vector)
        self.assertEqual(candidate, median_candidate)
        self.assertEqual(gini_idx, median_gini_idx)

    def test_candidate_gini_index(self):
        attribute_vector = pd.Series([20,21,23,23,25,29,32,33])
        target_vector = pd.Series([0,0,0,1,0,1,0,1])
        candidate = 24
        correct_gini_index = 7/16
        gini_idx = gini.get_candidate_gini_index(
                attribute_vector,target_vector, candidate)
        self.assertEqual(correct_gini_index, gini_idx)

    def test_split_candidates(self):
        values = [1,5,3]
        candidates = gini.get_split_candidates(values)
        true_candidates = np.array([2.0, 4.0])
        self.assertTrue((candidates == true_candidates).all())

if __name__ == '__main__':
    unittest.main()

