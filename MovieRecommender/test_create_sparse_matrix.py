import unittest
import os
import pickle
import pandas as pd
from process_data import create_sparse_matrix

testdata = {'user_id':[1, 2, 5, 5, 5], 'item_id':[4, 5, 6, 7, 7], 'rating':[7, 1, 3, 4, 5]}
testdf = pd.DataFrame(testdata)
testsparse_item_user, testsparse_user_item = create_sparse_matrix(testdf)

class TestSparseMatrix(unittest.TestCase):

    #check the dimensions of the returned item_user matrix
    def test_user_item(self):
        returnshape = testsparse_item_user.get_shape()
        testshape = ((testdf['item_id'].nunique()),(testdf['user_id'].nunique()))
        self.assertEqual(testshape, returnshape)

    #check the dimensions of the returned user_item matrix
    def test_item_user(self):
        returnshape = testsparse_user_item.get_shape()
        testshape = ((testdf['user_id'].nunique()),(testdf['item_id'].nunique()))
        self.assertEqual(testshape, returnshape)

    #check to ensure returns csr matrix format
    def test_return_format1(self):
        returnformat = testsparse_item_user.getformat()
        self.assertEqual(returnformat, "csr")
        
    #check to ensure returns csr matrix format
    def test_return_format2(self):
        returnformat = testsparse_user_item.getformat()
        self.assertEqual(returnformat, "csr")

if __name__ == '__main__':
    unittest.main()