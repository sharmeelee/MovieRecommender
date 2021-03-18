import unittest
import os
import pickle
import pandas as pd
from MovieRecommender.process_data import create_sparse_matrix

testdata = {'user_id':[1, 2, 5, 5, 5], 'item_id':[4, 5, 6, 7, 7], 'rating':[7, 1, 3, 4, 5]}
testdf = pd.DataFrame(testdata)
testsparse_item_user, testsparse_user_item = create_sparse_matrix(testdf)

class TestSparseMatrix(unittest.TestCase):

    def test_user_item(self):
        """Check the dimensions of the returned item_user matrix"""
        returnshape = testsparse_item_user.get_shape()
        testshape = ((testdf['item_id'].nunique()),(testdf['user_id'].nunique()))
        self.assertEqual(testshape, returnshape)

    def test_item_user(self):
        """Check the dimensions of the returned user_item matrix"""
        returnshape = testsparse_user_item.get_shape()
        testshape = ((testdf['user_id'].nunique()),(testdf['item_id'].nunique()))
        self.assertEqual(testshape, returnshape)

    def test_return_format1(self):
        """Check to ensure item_user csr matrix is returned"""
        self.assertEqual(testsparse_item_user.getformat(), "csr")
        
    def test_return_format2(self):
        """Check to ensure user_item csr matrix is returned"""
        self.assertEqual(testsparse_user_item.getformat(), "csr")

if __name__ == '__main__':
    unittest.main()
