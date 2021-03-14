import unittest
import os
import pickle
import pandas as pd
from MovieRecommender.train_test_model import test_train_split
from MovieRecommender.process_data import create_sparse_matrix

testdf = pd.DataFrame({'user_id':[1, 2, 5, 5, 5, 6, 5], 'item_id':[2, 4, 6, 8, 62, 62, 100], 'rating':[1, 0, 3, 4, 5, 2, 1]})
test_sparse_item_user, test_sparse_user_item = create_sparse_matrix(testdf)
test_train_data, test_test_data, test_users_altered = test_train_split(test_sparse_user_item)

# Define a class in which the tests will run
class TestTrainSplit(unittest.TestCase):
    
    #test to make sure drops zero ratings
    def test_dim_drop_zeros(self):
        #should return a csr matrix with dimensions (number of unique users, number of unique items) 
        #after dropping the user/item with a zero rating
        self.assertEqual(test_test_data.get_shape(), (3, 5))
    
    #test to make sure test data only contains ones and zeros
    def test_zeros_ones(self):
        arrtest = test_test_data.toarray()
        for i in arrtest:
            self.assertTrue(max(i) == 1)
            self.assertTrue(min(i) == 0)

    #test to make sure training data has fewer stored values than test data
    def test_train_less_test(self):
        testnnz = test_test_data.getnnz()
        trainnnz = test_train_data.getnnz()
        self.assertTrue(trainnnz < testnnz)

    #test to make sure training data dimensions match test data dimensions
    def test_dim_match(self):
        self.assertEqual(test_test_data.get_shape(), test_train_data.get_shape())

    #test to make sure some users were altered in the creation of the training set
    def test_users_altered(self):
        self.assertIsNotNone(len(test_users_altered))

if __name__ == '__main__': 
        unittest.main()
