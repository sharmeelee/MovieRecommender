from MovieRecommender import process_data
import numpy as np
from scipy.sparse import csr_matrix, load_npz
import random
import implicit
from sklearn import metrics
import os
import pickle


def test_train_split(sparse_user_item):
    def make_train(ratings, pct_test=0.2):
        # Make a copy of the original set to be the test set.
        test_set = ratings.copy()
        # Store the test set as a binary preference matrix
        test_set[test_set != 0] = 1
        # Make a copy of the original data we can alter as our training set.
        training_set = ratings.copy()
        # Find the indices in the ratings data where an interaction exists
        nonzero_inds = training_set.nonzero()
        # Zip these pairs together of user,item index into list
        nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1]))
        # Set the random seed to zero for reproducibility
        random.seed(0)
        # Round the number of samples needed to the nearest integer
        num_samples = int(np.ceil(pct_test*len(nonzero_pairs)))
        # Sample a random number of user-item pairs without replacement
        samples = random.sample(nonzero_pairs, num_samples)
        # Get the user row indices
        user_inds = [index[0] for index in samples]
        # Get the item column indices
        item_inds = [index[1] for index in samples]
        # Assign all of the randomly chosen user-item pairs to zero
        training_set[user_inds, item_inds] = 0
        # Get rid of zeros in sparse array storage after update to save space
        training_set.eliminate_zeros()
        # Output the unique list of user rows that were altered
        return training_set, test_set, list(set(user_inds))
    train_data, test_data, users_altered = make_train(sparse_user_item,
                                                      pct_test=0.2)
    print("Train test split done! ", train_data.shape, test_data.shape)
    return train_data, test_data, users_altered


def train_model(train_data):
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


def evaluate_model(training_set, altered_users, predictions, test_set):
    def auc_score(predictions, test):
        fpr, tpr, thresholds = metrics.roc_curve(test, predictions)
        return metrics.auc(fpr, tpr)
    # Store AUC for each user that had item removed from training set
    store_auc = []
    # To store popular AUC scores
    popularity_auc = []
    # Get sum of item iteractions to find most popular
    pop_items = np.array(test_set.sum(axis=0)).reshape(-1)
    item_vecs = predictions[1]
    # Iterate through each user that had an item altered
    for user in altered_users:
        # Get the training set row
        training_row = training_set[user, :].toarray().reshape(-1)
        # Find where the interaction had not yet occurred
        zero_inds = np.where(training_row == 0)
        # Get the predicted values based on our user/item vectors
        user_vec = predictions[0][user, :]
        pred = user_vec.dot(item_vecs).toarray()[0, zero_inds].reshape(-1)
        # Get only the items that were originally zero
        # Select ratings from MF prediction for user that had no interaction
        actual = test_set[user, :].toarray()[0, zero_inds].reshape(-1)
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
    process_data.main()
    sparse_user_item = load_npz("./output/sparse_user_item.npz")
    train_data, test_data, users_altered = test_train_split(sparse_user_item)
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
