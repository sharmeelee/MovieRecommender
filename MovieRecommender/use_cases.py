import train_test_model
import pandas as pd
import numpy as np
#import math
import os, sys, time 
from scipy.sparse import csr_matrix, save_npz, load_npz
import pickle

##########################################################################################   
      
def usecases(predictions,item_vecs,model,movie_list=["Sliding Doors"],user_id=100,n_similar=20):
    def predict_ratings(predictions,item_vecs,user_id):
        item_vecs = predictions[1]
        user_vec = predictions[0][user_id,:]
        pred = user_vec.dot(item_vecs).toarray()[0].reshape(-1)
        return pred
    
    def similar_items(model,item_id,n_similar=10):
        # Use implicit to get similar items.
        movie_names = []
        similar = model.similar_items(item_id, n_similar)
        # Print the names of similar movies
        for item in similar:
            idx, rating = item
            movie_names.append(movies.name.loc[movies.item_id == idx].iloc[0])
        return movie_names
            
    def recommendations(model,sparse_user_item,user_id=user_id):
        # Use the implicit recommender.
        recommended = model.recommend(user_id, sparse_user_item)
        movies_recom = []
        ratings_recom = []
        # Get artist names from ids
        for item in recommended:
            idx, rating = item
            movies_recom.append((movies.name.loc[movies.item_id == idx].iloc[0]))
            ratings_recom.append(rating)
        # Create a dataframe of artist names and scores
        recommendations = pd.DataFrame({'movies': movies_recom, 'rating': ratings_recom})
        return recommendations
    
    #print("movie_list : ",movie_list, "User_id : ", user_id, "similar items : ", n_similar - 1)
    predict_ratings = predict_ratings(predictions,item_vecs,user_id)
    movies.name = movies.name.str.strip()
    item_id = movies.item_id.loc[movies.name.isin(movie_list)].iloc[0]
    similar_items = similar_items(als_model,item_id,n_similar)
    recommendations = recommendations(als_model,sparse_user_item,user_id)
    return predict_ratings, similar_items,recommendations

########################################################################################### 
def main():
  train_test_model.main()
  movies = pd.read_pickle("./output/movies.pkl")
  sparse_user_item = load_npz("./output/sparse_user_item.npz")
  item_vecs = np.load('./output/item_vecs.npy')
  user_vecs = np.load('./output/user_vecs.npy')
  with open('./output/als_model', 'rb') as file:  
    als_model = pickle.load(file)
    
  if len(sys.argv) >= 2:
    predict_ratings, similar_items,recommendations = usecases([csr_matrix(user_vecs), csr_matrix(item_vecs.T)],\
    item_vecs,als_model,movie_list=[sys.argv[1]],user=sys.argv[2],item=sys.argv[3],n_similar=21)  
  else:
    predict_ratings, similar_items,recommendations = usecases([csr_matrix(user_vecs), csr_matrix(item_vecs.T)],\
    item_vecs,als_model)
  print(predict_ratings, (similar_items-1), recommendations)

if __name__ =="__main__":
  train_test_model.main()
  movies = pd.read_pickle("./output/movies.pkl")
  sparse_user_item = load_npz("./output/sparse_user_item.npz")
  item_vecs = np.load('./output/item_vecs.npy')
  user_vecs = np.load('./output/user_vecs.npy')
  with open('./output/als_model', 'rb') as file:  
    als_model = pickle.load(file)
  
  print('Number of arguments:', len(sys.argv) - 1, 'arguments.')
  print('Argument List:', str(sys.argv))
  if len(sys.argv) == 2:
    movie_list=[sys.argv[1]]
    user_id=100
    n_similar=21
    predict_ratings, similar_items,recommendations = usecases([csr_matrix(user_vecs), csr_matrix(item_vecs.T)],\
    item_vecs,als_model,movie_list=movie_list,user_id=user_id,n_similar=n_similar)
  elif len(sys.argv) == 3:
    movie_list=[sys.argv[1]]
    user_id=100
    n_similar=int(sys.argv[2]) + 1
    predict_ratings, similar_items,recommendations = usecases([csr_matrix(user_vecs), csr_matrix(item_vecs.T)],\
    item_vecs,als_model,movie_list=movie_list,user_id=user_id,n_similar=n_similar)  
  elif len(sys.argv) == 4:
    movie_list=[sys.argv[1]]
    user_id=int(sys.argv[2])
    n_similar=int(sys.argv[3]) + 1
    predict_ratings, similar_items,recommendations = usecases([csr_matrix(user_vecs), csr_matrix(item_vecs.T)],\
    item_vecs,als_model,movie_list=movie_list,user_id=user_id,n_similar=n_similar)  
  else:
    movie_list=["Sliding Doors"]
    user_id=100
    n_similar=21
    predict_ratings, similar_items,recommendations = usecases([csr_matrix(user_vecs), csr_matrix(item_vecs.T)],\
    item_vecs,als_model)
  print()
  print("************************** PREDICTED RATINGS FOR USER :" +str(user_id) +" ****************")
  print()
  print(predict_ratings)
  print()
  print("************************** "+str(n_similar - 1) +" MOVIES SIMILAR TO :" +str(movie_list) +"  *****************")
  print()
  print(similar_items[1:])
  print()
  print("************************** RECOMMEDATIONS FOR USER :" +str(user_id) +" ******************")
  print()
  print(recommendations)
  print()
  print("*************************************************************************************")
  

