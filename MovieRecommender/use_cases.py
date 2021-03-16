from MovieRecommender import train_test_model
import pandas as pd
import numpy as np
#import math
import os, sys, time 
from scipy.sparse import csr_matrix, save_npz, load_npz
import pickle
from tabulate import tabulate

##########################################################################################

def get_movies_rated(data,user_id, train_data,movies):
    data_matrix = data.loc[data.rating != 0]
    users = list(np.sort(data_matrix.user_id.unique())) # Get our unique users
    items = list(np.sort(data_matrix.item_id.unique())) # Get our unique movies 
    users_arr = np.array(users) # Array of user IDs from the ratings matrix
    items_arr = np.array(items) # Array of movie IDs from the ratings matrix
    user_ind = np.where(users_arr == user_id)[0][0] # Returns the index row of our user id
    rating_ind = train_data[user_ind,:].nonzero()[1] # Get column indices of rated items
    movie_codes = items_arr[rating_ind] # Get the movie ids for our rated items
    return movies.loc[movies['item_id'].isin(movie_codes),'name'].reset_index(drop=True)  
      
def predict_ratings(predictions,item_vecs,user_id):
    item_vecs = predictions[1]
    user_vec = predictions[0][user_id,:]
    pred = user_vec.dot(item_vecs).toarray()[0].reshape(-1)
    return pred
    
def similar_items(movies,model,movie_list,n_similar=20):
    # Use implicit to get similar items.
    movies.name = movies.name.str.strip()
    item_id = movies.item_id.loc[movies.name.str.lower().isin([s.lower() for s in movie_list])].iloc[0]
    movie_names = []
    similar = model.similar_items(item_id, n_similar)
    # Print the names of similar movies
    for item in similar:
        idx, rating = item
        movie_names.append(movies.name.loc[movies.item_id == idx+1].iloc[0])
    similar = pd.DataFrame({"Similar Movies":movie_names[1:]})
    return similar
            
def recommendations(data,train_data,movies,model,sparse_user_item,user_id):
    # Use the implicit recommender.
    recommended = model.recommend(user_id, sparse_user_item)
    movies_recom = []
    ratings_recom = []
    # Get movie names from ids
    for item in recommended:
        idx, rating = item
        movies_recom.append((movies.name.loc[movies.item_id == idx+1].iloc[0]))
        #ratings_recom.append(rating)
    # Create a dataframe of movie names and scores
    #recommendations = pd.DataFrame({'Movies': movies_recom, 'Rating': ratings_recom})
    movies_rated_by_users = get_movies_rated(data,user_id, train_data, movies)
    minlen  = min(len(movies_recom),len(movies_rated_by_users))
    recommendations = pd.DataFrame({'Recommended Movies': movies_recom[:minlen], 'Movies Rated': movies_rated_by_users[:minlen]})
    return recommendations

########################################################################################### 
def main():
    train_test_model.main()
    movies = pd.read_pickle("./output/movies.pkl")
    sparse_user_item = load_npz("./output/sparse_user_item.npz")
    item_vecs = np.load('./output/item_vecs.npy')
    user_vecs = np.load('./output/user_vecs.npy')
    data = pd.read_pickle("./output/ratings.pkl")
    with open('./output/als_model', 'rb') as file:  
        als_model = pickle.load(file)
    with open('./output/train_data', 'rb') as train_file:  
        train_data = pickle.load(train_file)
    with open('./output/test_data', 'rb') as test_file:  
        test_data = pickle.load(test_file)
  
    print('Number of arguments:', len(sys.argv) - 1, 'arguments.')
    print('Argument List:', str(sys.argv))
    if len(sys.argv) == 2:
        movie_list=[sys.argv[1]]
        n_similar=21
        similar_df = similar_items(movies,als_model,movie_list,n_similar)
        similar_df.index += 1 
        print()
        print("************************** "+str(n_similar - 1) +" MOVIES SIMILAR TO :" +str(movie_list) +"  *****************")
        print()
        print(tabulate(similar_df, tablefmt="pipe", headers="keys"))
        print()
        print("*************************************************************************************") 
    elif len(sys.argv) == 3:
        movie_list=[sys.argv[1]]
        n_similar=int(sys.argv[2]) + 1
        similar_df = similar_items(movies,als_model,movie_list,n_similar)
        similar_df.index += 1 
        print()
        print("************************** "+str(n_similar - 1) +" MOVIES SIMILAR TO :" +str(movie_list) +"  *****************")
        print()
        print(tabulate(similar_df, tablefmt="pipe", headers="keys"))
        print()
        print("*************************************************************************************") 
    elif len(sys.argv) == 4:
        movie_list=[sys.argv[1]]
        user_id=int(sys.argv[2])
        n_similar=int(sys.argv[3]) + 1
        
        predictions = [csr_matrix(user_vecs), csr_matrix(item_vecs.T)]       
        predictRatings = predict_ratings(predictions,item_vecs,user_id)
        actualRatings = test_data[user_id,:].toarray().reshape(-1)
        ratings_df = pd.DataFrame({"Predicted Ratings":predictRatings,"Actual Ratings":actualRatings})
        ratings_df.index += 1

        similar_df = similar_items(movies,als_model,movie_list,n_similar)
        similar_df.index += 1 

        recomm_df = recommendations(data,train_data,movies,als_model,sparse_user_item,user_id)
        recomm_df.index += 1 

        print()
        print("************************** TOP 20 RATINGS FOR USER :" +str(user_id) +" ****************")
        print()
        print(tabulate(ratings_df[:20], tablefmt="pipe", headers="keys"))
        print()
        print("************************** "+str(n_similar - 1) +" MOVIES SIMILAR TO :" +str(movie_list) +"  *****************")
        print()
        print(tabulate(similar_df, tablefmt="pipe", headers="keys"))
        print()
        print("************************** RECOMMEDATIONS FOR USER :" +str(user_id) +" ******************")
        print()
        print(tabulate(recomm_df, tablefmt="pipe", headers="keys"))
        print()
        print("*************************************************************************************")         
    else:
        movie_list=["Sliding Doors"]
        user_id=100
        n_similar=21
        predictions = [csr_matrix(user_vecs), csr_matrix(item_vecs.T)]       
        predictRatings = predict_ratings(predictions,item_vecs,user_id)
        actualRatings = test_data[user_id,:].toarray().reshape(-1)
        ratings_df = pd.DataFrame({"Predicted Ratings":predictRatings,"Actual Ratings":actualRatings})
        ratings_df.index += 1

        similar_df = similar_items(movies,als_model,movie_list,n_similar)
        similar_df.index += 1 

        recomm_df = recommendations(data,train_data,movies,als_model,sparse_user_item,user_id)
        recomm_df.index += 1 

        print()
        print("************************** TOP 20 RATINGS FOR USER :" +str(user_id) +" ****************")
        print()
        print(tabulate(ratings_df[:20], tablefmt="pipe", headers="keys"))
        print()
        print("************************** "+str(n_similar - 1) +" MOVIES SIMILAR TO :" +str(movie_list) +"  *****************")
        print()
        print(tabulate(similar_df, tablefmt="pipe", headers="keys"))
        print()
        print("************************** RECOMMEDATIONS FOR USER :" +str(user_id) +" ******************")
        print()
        print(tabulate(recomm_df, tablefmt="pipe", headers="keys"))
        print()
        print("*************************************************************************************")     

if __name__ =="__main__":
  main()