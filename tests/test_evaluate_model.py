import unittest
import os
import pandas as pd
from scipy.sparse import csr_matrix, load_npz
from MovieRecommender.train_test_model import *

test_sparse_user_item = load_npz("./tests/test_sparse_user_item.npz")
test_train_data, test_test_data, test_users_altered = test_train_split(test_sparse_user_item)
test_als_model, test_user_vecs, test_item_vecs = train_model(test_train_data.T)

return_store_AUC, return_popularity_AUC = evaluate_model(test_train_data, test_users_altered, [csr_matrix(test_user_vecs), csr_matrix(test_item_vecs.T)], test_test_data)

# Define a class in which the tests will run
class TestEvaluateModel(unittest.TestCase):
    
    #test to make sure return scores are floats
    def test_return_format(self):
        self.assertEqual(type(return_store_AUC), float)
        self.assertEqual(type(return_popularity_AUC), float)

if __name__ == '__main__':
    unittest.main()
