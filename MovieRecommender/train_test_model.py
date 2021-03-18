from MovieRecommender import process_data
import numpy as np
from scipy.sparse import csr_matrix, load_npz
import random
import implicit
from sklearn import metrics
import os
import pickle


def test_train_split(sparse_user_item,pct_test = 0.2):
    '''
    This function will take in the original user-item matrix and "mask" a 
    percentage of the original ratings where a user-item interaction has 
    taken place for use as a test set. The test set will contain all of 
    the original ratings, while the training set replaces the specified 
    percentage of them with a zero in the original ratings matrix. 
    
    parameters: 
    
    sparse_user_item - the original ratings sparse_user_item matrix from 
    which you want to generate a train/test set. Test is just a complete
    copy of the original set. This is in the form of a sparse csr_matrix. 
    
    pct_test - The percentage of user-item interactions where an 
    interaction took place that you want to mask in the training set for 
    later comparison to the test set, which contains all of the original 
    ratings. 
    
    returns:
    
    train_data - The altered version of the original data with a certain 
    percentage of the user-item pairs 
    that originally had interaction set back to zero.
    
    test_data - A copy of the original ratings matrix, unaltered, so it 
    can be used to see how the rank order compares with the actual 
    interactions.
    
    users_altered  - From the randomly selected user-item indices, which 
    user rows were altered in the training data.This will be necessary 
    later when evaluating the performance via AUC.
    '''
    ratings = sparse_user_item
    test_data = ratings.copy() # Make a copy of the original set to be the test set. 
    test_data[test_data != 0] = 1 # Store the test set as a binary preference matrix
    train_data = ratings.copy() # Make a copy of the original data we can alter as our training set. 
    nonzero_inds = train_data.nonzero() # Find the indices in the ratings data where an interaction exists
    nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1])) # Zip these pairs together of user,item index into list
    random.seed(0) # Set the random seed to zero for reproducibility
    num_samples = int(np.ceil(pct_test*len(nonzero_pairs))) # Round the number of samples needed to the nearest integer
    samples = random.sample(nonzero_pairs, num_samples) # Sample a random number of user-item pairs without replacement
    user_inds = [index[0] for index in samples] # Get the user row indices
    item_inds = [index[1] for index in samples] # Get the item column indices
    train_data[user_inds, item_inds] = 0 # Assign all of the randomly chosen user-item pairs to zero
    train_data.eliminate_zeros() # Get rid of zeros in sparse array storage after update to save space
    train_data, test_data, users_altered = train_data, test_data, list(set(user_inds)) # Output the unique list of user rows that were altered
    print("Train test split done! ",train_data.shape, test_data.shape)
    return train_data, test_data, users_altered
	
 
def train_model(train_data):   
    '''
    Implicit weighted ALS taken from Hu, Koren, and Volinsky 2008. Designed
    for alternating least squares and implicit feedback based collaborative
    filtering. 
    
    parameters:
    
    train_data - Our matrix of ratings with shape m x n, where m is the
    number of users and n is the number of items.Should be a sparse csr
    matrix to save space. 
    
    regularization - Used for regularization during alternating least
    squares. Increasing this value may increase bias but decrease
    variance. Default is 0.1. 
    
    alpha_val - The parameter associated with the confidence matrix,
    where Cui = 1 + alpha*Rui. The paper found a default of 40 most
    effective. Decreasing this will decrease the variability in
    confidence between various ratings.
    
    iterations - The number of times to alternate between both user 
    feature vector and item feature vector in alternating least squares.
    More iterations will allow better convergence at the cost of increased 
    computation. We will use 50 iterations for better convergence. 
    
    factors - The number of latent features in the user/item feature vectors.
    Increasing the number of features may overfit but could reduce bias. 
    
    returns:
    
    The model and feature vectors for users and items. The dot product of 
    these feature vectors should give you the expected 
    "rating" at each point in your original matrix.
    
    '''
    
    # Initialize the als model and fit it using the sparse item-user matrix
    als_model = implicit.als.AlternatingLeastSquares(factors=20,
                                                     regularization=1e-3,
                                                     iterations=50)
    # Calculate the confidence by multiplying it by our alpha value.
    alpha_val = 40
    # train data is of item-user format
    data_conf = (train_data * alpha_val).astype('double')
    als_model.fit(data_conf)
    # Get the user and item vectors from our trained model
    user_vecs = als_model.user_factors
    item_vecs = als_model.item_factors
    print("Model trained, user vectors and item vectors shape",
          user_vecs.shape, item_vecs.shape)
    return als_model, user_vecs, item_vecs


def evaluate_model(train_data, users_altered, predictions, test_data):
    '''
    This function will calculate the mean AUC by user for any user that had 
    their user-item matrix altered. 
    
    parameters:
    
    train_data - The training set resulting from make_train, where a certain 
    percentage of the original user/item interactions are reset to zero to 
    hide them from the model 
    
    predictions - The matrix of your predicted ratings for each user/item 
    pair as output from the implicit MF. These should be stored in a list, 
    with user vectors as item zero and item vectors as item one. 
    
    users_altered - The indices of the users where at least one user/item 
    pair was altered from make_train function
    
    test_data - The test set constucted earlier from make_train function    
    
    returns:
    
    The mean AUC (area under the Receiver Operator Characteristic curve) of 
    the test set only on user-item interactions there were originally zero to 
    test ranking ability in addition to the most popular items as a benchmark.
    '''
    def auc_score(predictions, test):
        '''
        This simple function will output the area under the curve using 
        sklearn's metrics. 

        parameters:

        - predictions: your prediction output

        - test: the actual target result you are comparing to

        returns:

        - AUC (area under the Receiver Operating Characterisic curve)
        
        '''
        fpr, tpr, thresholds = metrics.roc_curve(test, predictions)
        return metrics.auc(fpr, tpr)
    # Store AUC for each user that had item removed from training set
    store_auc = []
    # To store popular AUC scores
    popularity_auc = []
    # Get sum of item iteractions to find most popular
    pop_items = np.array(test_data.sum(axis=0)).reshape(-1)
    item_vecs = predictions[1]
    # Iterate through each user that had an item altered
    for user in users_altered:
        # Get the training set row
        training_row = train_data[user, :].toarray().reshape(-1)
        # Find where the interaction had not yet occurred
        zero_inds = np.where(training_row == 0)
        # Get the predicted values based on our user/item vectors
        user_vec = predictions[0][user, :]
        pred = user_vec.dot(item_vecs).toarray()[0, zero_inds].reshape(-1)
        # Get only the items that were originally zero
        # Select ratings from MF prediction for user that had no interaction
        actual = test_data[user, :].toarray()[0, zero_inds].reshape(-1)
        # Select yes/no interaction pairs from the original full data
        # Get the item popularity for our chosen items
        pop = pop_items[zero_inds]
        # Calculate AUC for the given user and store
        store_auc.append(auc_score(pred, actual))
        # Calculate AUC using most popular and score
        popularity_auc.append(auc_score(pop, actual))
        # End users interation
    mean_s_auc = float('%.3f' % np.mean(store_auc))
    mean_p_auc = float('%.3f' % np.mean(popularity_auc))
    return mean_s_auc, mean_p_auc


def main():
    '''
    This function executes the steps sequentially and writes
    the model, train and test data into pickle files to be 
    used later. It also saves the latent feature vectors
    using np.save to be used later
    
    '''
    process_data.main()
    sparse_user_item = load_npz("./output/sparse_user_item.npz")
    train_data, test_data, users_altered = test_train_split(sparse_user_item,pct_test = 0.2)
    # the parameter to trail_model should be item - user matrix
    als_model, user_vecs, item_vecs = train_model(train_data.T)
    print("implicit_recomm_auc,popularity_auc", evaluate_model(train_data,
          users_altered, [csr_matrix(user_vecs), csr_matrix(item_vecs.T)],
          test_data))

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


if __name__ == "__main__":
    main()
