from MovieRecommender import extract_data
import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
from scipy.sparse import csr_matrix, save_npz, load_npz
import os, sys, time 
import csv
import matplotlib.pyplot as plt

##########################################################################################   
    
def process_data(data):
    data.drop('timestamp',inplace=True,axis=1,errors='ignore')
    #Seeing the distribution of ratings given by the users
    p = data.groupby('rating')['rating'].agg(['count'])
    # get movie count
    movie_count = data.item_id.unique().shape[0]
    # get customer count
    cust_count = data.user_id.unique().shape[0]
    # get rating count
    rating_count = data['user_id'].count()
    ax = p.plot(kind = 'barh', legend = False, figsize = (15,10))
    plt.title('Total pool: {:,} Movies, {:,} customers, {:,} ratings given'.format(movie_count, cust_count, rating_count), fontsize=20)
    plt.axis('off')
    for i in range(1,6):
        ax.text(p.iloc[i-1][0]/4, i-1, 'Rated {}: {:.0f}%'.format(i, p.iloc[i-1][0]*100 / p.sum()[0]), color = 'white', weight = 'bold',fontsize=15)

    directory = 'images'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig('images/DistributionOfRatings.png')
    return data, cust_count, movie_count

########################################################################################### 

def create_sparse_matrix(data):
    data_matrix = data.loc[data.rating != 0]

    users = list(np.sort(data_matrix.user_id.unique())) # Get our unique users
    items = list(np.sort(data_matrix.item_id.unique())) # Get our unique movies 
    rating = list(data_matrix.rating) # All of our ratings

    rows = data_matrix.user_id.astype(CategoricalDtype(categories=users)).cat.codes 
    cols = data_matrix.item_id.astype(CategoricalDtype(categories=items)).cat.codes
    
    # The implicit library expects data as a item-user matrix so we
    # create two matricies, one for fitting the model (item-user) 
    # and one for recommendations (user-item)
    sparse_item_user = csr_matrix((rating, (cols, rows)), shape=(len(items), len(users)))
    sparse_user_item = csr_matrix((rating, (rows, cols)), shape=(len(users), len(items)))
    print("Sparse matrices created : sparse_item_user ", sparse_item_user.shape, "sparse_user_item" ,sparse_user_item.shape)
    return sparse_item_user,sparse_user_item

########################################################################################### 

def main():
  extract_data.main()
  data = pd.read_pickle("./output/ratings.pkl")
  data, num_users, num_items = process_data(data)
  #num_users = data.user_id.unique().shape[0]
  #num_items = data.item_id.unique().shape[0]
  print(data.shape, num_users, num_items)
  print(f'We have {data.shape[0]} ratings from {num_users} users and {num_items} items')
  #users,items,sparse_item_user,sparse_user_item = create_sparse_matrix(data)
  sparse_item_user,sparse_user_item = create_sparse_matrix(data)
  
  directory = './output'
  if not os.path.exists(directory):
    os.makedirs(directory)
      
  save_npz("./output/sparse_item_user.npz", sparse_item_user)
  save_npz("./output/sparse_user_item.npz", sparse_user_item)

if __name__ =="__main__":
  main()
