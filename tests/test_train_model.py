import unittest
import pandas as pd
from scipy.sparse import csr_matrix
import implicit
from MovieRecommender.train_test_model import train_model

testdf = pd.DataFrame({'user_id': [1, 2, 5, 7, 9, 12, 45],
                      'item_id': [2, 4, 6, 8, 62, 92, 100],
                       'rating': [1, 5, 0, 4, 5, 2, 1]})
testsparse = csr_matrix(testdf)
return_als_model, return_user_vecs, return_item_vecs = train_model(testsparse)
modelfactor = 20


# Define a class in which the tests will run
class TestTrainModel(unittest.TestCase):

    # test to make sure return is ALS object
    def test_return_type(self):
        self.assertEqual(type(return_als_model),
                         implicit.als.AlternatingLeastSquares)

    # test to make sure return factors match parameters set at 20
    def test_model_factors(self):
        self.assertEqual(return_als_model.factors, modelfactor)

    # test to make sure returned user vector has correct dimensions
    def test_user_vecs(self):
        returnshape = return_user_vecs.shape
        checkshape = (testsparse.shape[1], modelfactor)
        self.assertEqual(returnshape, checkshape)

    # test to make sure returned item vector has correct dimensions
    def test_item_vecs(self):
        returnshape = return_item_vecs.shape
        checkshape = (testsparse.shape[0], modelfactor)
        self.assertEqual(returnshape, checkshape)


if __name__ == '__main__':
    unittest.main()
