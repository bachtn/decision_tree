#!/usr/bin/env python3

import unittest

import split_metrics.entropy as entropy
from tools.tools import get_dataset, ObjectView
from main import dataset_dict

class EntropyTest(unittest.TestCase):
    def test_information_gain(self):
        information_gain = entropy.get_information_gain(
                self.categorical_vectors.attribute_vector,
                self.categorical_vectors.target_vector)
        self.assertEqual(round(information_gain, 4), 0.2467)

    def test_information_gain_ratio(self):
        info_gain_ratio = entropy.get_information_gain_ratio(
                self.categorical_vectors.attribute_vector,
                self.categorical_vectors.target_vector)
        self.assertEqual(round(info_gain_ratio, 4), 0.1564)

    def test_vector_entropy(self):
        # Test Categorical data
        vector_entropy = entropy.get_vector_entropy(
                self.categorical_vectors.attribute_vector)
        self.assertEqual(round(vector_entropy, 4), 1.5774)
        # Test Continuous data
        with self.assertRaises(ValueError):
            entropy.get_vector_entropy(
                    self.continuous_vectors.attribute_vector)

    def test_partition_entropy(self):
        # Test Categorical data
        partition_entropy = entropy.get_partition_entropy(
                self.categorical_vectors.attribute_vector,
                self.categorical_vectors.target_vector)
        self.assertEqual(round(partition_entropy, 4), 0.6935)
        # Test Continuous data
        with self.assertRaises(ValueError):
            entropy.get_partition_entropy(
                    self.continuous_vectors.attribute_vector,
                    self.continuous_vectors.target_vector)

    @classmethod
    def setUpClass(self):
        self.continuous_vectors = self.__get_vectors('iris')
        self.categorical_vectors = self.__get_vectors('tennis')
    
    @classmethod
    def __get_vectors(self, dataset_name):
        data, label, attribute_list = get_dataset(dataset_name, dataset_dict)
        target_vector = data[label]
        attribute_vector = data[attribute_list[0]]
        return ObjectView(
                {
                    'target_vector' : target_vector,
                    'attribute_vector' : attribute_vector
                })

if __name__ == '__main__':
    unittest.main()
