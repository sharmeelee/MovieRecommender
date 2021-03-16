from MovieRecommender import process_data
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
import random
import implicit
from sklearn import metrics
import os, sys, time 
import csv
import pickle

##########################################################################################   
      
def test_train_split(sparse_user_item):
  def make_train(ratings, pct_test = 0.2):
    test_set = ratings.copy() # Make a copy of the original set to be the test set. 
    test_set[test_set != 0] = 1 # Store the test set as a binary preference matrix
    training_set = ratings.copy() # Make a copy of the original data we can alter as our training set. 
    nonzero_inds = training_set.nonzero() # Find the indices in the ratings data where an interaction exists
    nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1])) # Zip these pairs together of user,item index into list
    random.seed(0) # Set the random seed to zero for reproducibility
    num_samples = int(np.ceil(pct_test*len(nonzero_pairs))) # Round the number of samples needed to the nearest integer
    samples = random.sample(nonzero_pairs, num_samples) # Sample a random number of user-item pairs without replacement
    user_inds = [index[0] for index in samples] # Get the user row indices
    item_inds = [index[1] for index in samples] # Get the item column indices
    training_set[user_inds, item_inds] = 0 # Assign all of the randomly chosen user-item pairs to zero
    training_set.eliminate_zeros() # Get rid of zeros in sparse array storage after update to save space
    return training_set, test_set, list(set(user_inds)) # Output the unique list of user rows that were altered
  train_data, test_data, users_altered = make_train(sparse_user_item, pct_test = 0.2)
  print("Train test split done! ",train_data.shape, test_data.shape)
  return train_data, test_data, users_altered  
  
########################################################################################### 

def train_model(train_data):   
    # Initialize the als model and fit it using the sparse item-user matrix
    als_model = implicit.als.AlternatingLeastSquares(factors=20, 
                                                     regularization=1e-3, 
                                                     iterations=50)
    # Calculate the confidence by multiplying it by our alpha value.
    alpha_val = 40
    data_conf = (train_data * alpha_val).astype('double') # train data is of item-user format
    als_model.fit(data_conf)
    # Get the user and item vectors from our trained model
    user_vecs = als_model.user_factors
    item_vecs = als_model.item_factors
    print("Model trained, user vectors and item vectors shape",user_vecs.shape,item_vecs.shape)
    return als_model,user_vecs,item_vecs
    
########################################################################################### 

def evaluate_model(training_set, altered_users, predictions, test_set):
    def auc_score(predictions, test):
        fpr, tpr, thresholds = metrics.roc_curve(test, predictions)
        return metrics.auc(fpr, tpr)
    
    store_auc = [] # An empty list to store the AUC for each user that had an item removed from the training set
    popularity_auc = [] # To store popular AUC scores
    pop_items = np.array(test_set.sum(axis = 0)).reshape(-1) # Get sum of item iteractions to find most popular
    item_vecs = predictions[1]
    for user in altered_users: # Iterate through each user that had an item altered
      training_row = training_set[user,:].toarray().reshape(-1) # Get the training set row
      zero_inds = np.where(training_row == 0) # Find where the interaction had not yet occurred
      # Get the predicted values based on our user/item vectors
      user_vec = predictions[0][user,:]
      pred = user_vec.dot(item_vecs).toarray()[0,zero_inds].reshape(-1)
      # Get only the items that were originally zero
      # Select all ratings from the MF prediction for this user that originally had no iteraction
      actual = test_set[user,:].toarray()[0,zero_inds].reshape(-1) 
      # Select the binarized yes/no interaction pairs from the original full data
      # that align with the same pairs in training 
      pop = pop_items[zero_inds] # Get the item popularity for our chosen items
      store_auc.append(auc_score(pred, actual)) # Calculate AUC for the given user and store
      popularity_auc.append(auc_score(pop, actual)) # Calculate AUC using most popular and score
      # End users iteration    
    return float('%.3f'%np.mean(store_auc)), float('%.3f'%np.mean(popularity_auc))  
    # Return the mean AUC rounded to three decimal places for both test and popularity benchmarkm  

########################################################################################### 
def main():
    process_data.main()
    sparse_user_item = load_npz("./output/sparse_user_item.npz")
    train_data, test_data, users_altered = test_train_split(sparse_user_item)
    als_model,user_vecs,item_vecs = train_model(train_data.T) # the parameter to trail_model should be item - user matrix
    print("implicit_recomm_auc,popularity_auc",evaluate_model(train_data, users_altered,[csr_matrix(user_vecs), csr_matrix(item_vecs.T)], test_data))
  
    directory = './output'
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save('./output/item_vecs', item_vecs)
    np.save('./output/user_vecs', user_vecs)
  
    with open('./output/als_model', 'wb') as file:  
        pickle.dump(als_model, file)
        
    with open('./output/train_data', 'wb') as train_file:  
        pickle.dump(train_data, train_file)
        
    with open('./output/test_data', 'wb') as test_file:  
        pickle.dump(test_data, test_file)

if __name__ =="__main__":
  main()

