import unittest
import os
import pickle
import pandas as pd
from MovieRecommender.process_data import process_data, main

testdata = {'user_id':[1, 2, 3, 5, 5], 'item_id':[4, 5, 6, 7, 7], 'rating':[1, 2, 3, 4, 5], 'timestamp':[10, 11, 12, 12, 12]}
testdf = pd.DataFrame(testdata)
testdfreturn, test_cust_count, test_item_count = process_data(testdf)

class TestProcessData(unittest.TestCase):

    def test_column_drop(self):
        """Check that the timestamp column has been removed"""
        self.assertFalse('timestamp' in testdfreturn)

    def test_cust_counts(self):
        """Check return has correct number of unique users"""
        self.assertEqual(test_cust_count, testdfreturn['user_id'].nunique())

    def test_item_counts(self):
        """Check return has correct number of unique items"""
        self.assertEqual(test_item_count, testdfreturn['item_id'].nunique())
        
    def test_main(self):
        """Check that an ouput directory was created"""
        main()
        self.assertTrue(os.path.exists('./output'))
        
    def test_images(self):
        """Check that an image directory was created"""
        self.assertTrue(os.path.exists('./images'))


if __name__ == '__main__':
    unittest.main()
