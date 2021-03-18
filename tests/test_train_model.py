import unittest
import os
import pickle
import pandas as pd
from scipy.sparse import csr_matrix
import implicit
from MovieRecommender.train_test_model import train_model, main

testdf = pd.DataFrame({'user_id':[1, 2, 5, 7, 9, 12, 45], 'item_id':[2, 4, 6, 8, 62, 92, 100], 'rating':[1, 5, 0, 4, 5, 2, 1]})
testsparse = csr_matrix(testdf)
return_als_model, return_user_vecs, return_item_vecs = train_model(testsparse)
modelfactor = 20

class TestTrainModel(unittest.TestCase):

    def test_return_type(self):
        """Check to make sure return is ALS object"""
        self.assertEqual(type(return_als_model), implicit.als.AlternatingLeastSquares)

    def test_model_factors(self):
        """Check value of return parameters"""
        self.assertEqual(return_als_model.factors, modelfactor)

    def test_user_vecs(self):
        """Check for correct dimensions on returned user vector"""
        returnshape = return_user_vecs.shape
        checkshape = (testsparse.shape[1], modelfactor)
        self.assertEqual(returnshape, checkshape)

    def test_item_vecs(self):
        """Check for correct dimensions on returned item vector"""
        returnshape = return_item_vecs.shape
        checkshape = (testsparse.shape[0], modelfactor)
        self.assertEqual(returnshape, checkshape)
        
    def test_main(self):
        """Check that output directory exists"""
        main()
        self.assertTrue(os.path.exists('./output'))
        
if __name__ == '__main__': 
    unittest.main()
