import unittest
import os
import pickle
import pandas as pd
from MovieRecommender.extract_data import read_data_ml100k, get_movies_ratings, main

testdata, testmovies, testnum_users, testnum_items = read_data_ml100k()
return_movies = get_movies_ratings(testmovies)

class TestReadData(unittest.TestCase):
    
    def test_df_read(self):
        self.assertEqual(len(testdata.columns), 4)

    def test_users_type(self):
        self.assertTrue(type(testnum_users) is int)

    def test_items_type(self):
        self.assertTrue(type(testnum_items) is int)
    
    def test_get_df(self):
        self.assertTrue(type(return_movies) is pd.DataFrame)

    def test_df_ratings(self):
        self.assertEqual(len(return_movies.columns), 2)
    
    def test_main(self):
        result = os.system("python3 extract_data.py")
        self.assertTrue(type(result) is int)

if __name__ == '__main__':
    unittest.main()
