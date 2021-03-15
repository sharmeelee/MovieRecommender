import unittest
import os
import pickle
import pandas as pd
from MovieRecommender.process_data import process_data, main

#create a dataframe with the four columns and send it
testdata = {'user_id':[1, 2, 3, 5, 5], 'item_id':[4, 5, 6, 7, 7], 'rating':[1, 2, 3, 4, 5], 'timestamp':[10, 11, 12, 12, 12]}
testdf = pd.DataFrame(testdata)
testdfreturn, test_cust_count, test_item_count = process_data(testdf)

class TestProcessData(unittest.TestCase):

    #test to make sure return has no column titled timestamp
    #check to make sure it comes back without the timestamp column or that it matches columns
    def test_column_drop(self):
        self.assertFalse('timestamp' in testdfreturn)

    def test_cust_counts(self):
        self.assertEqual(test_cust_count, testdfreturn['user_id'].nunique())

    def test_item_counts(self):
        self.assertEqual(test_item_count, testdfreturn['item_id'].nunique())
        
    def test_main(self):
        main()
        self.assertTrue(os.path.exists('./output'))


if __name__ == '__main__':
    unittest.main()
