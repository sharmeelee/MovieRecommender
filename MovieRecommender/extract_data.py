# import required packages
import os
import pandas as pd
import zipfile
import requests


# download the data from remote url
def read_data_ml100k():
    '''
    This function will download data from the given data_url, splits the path
    using '/' delimiter and reads the filename. It creates a directory
    'data' if not already exising and saves the downloaded zipfile in that
    location ./data. It then reads the zipped file, extracts its contents.
    For this particular project, we would be accessing u.data and u.item
    files from the extracted contents.
    
    returns:
    
    data - This is a dataframe obtained from u.data file that has user_id, 
    item_id, rating and timestamp.
    
    movies - This is a dataframe obtained from u.item that has movies 
    information delimited by '|' as line item.
    
    num_users  - Unique users from data dataframe
    
    num_items - Unique movies from data dataframe
    
    '''
    data_url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'

    def download_and_extract_data(url):
        """Download and extract a zip/tar file."""
        directory = './data'
        if not os.path.exists(directory):
            os.makedirs(directory)
        fname = os.path.join('./data', url.split('/')[-1])

        if os.path.exists(fname):
            print(f'File {fname} already exists. Reading it')
        else:
            print(f'Downloading {fname} from {url}...')
            r = requests.get(url, stream=True, verify=True)
            with open(fname, 'wb') as f:
                f.write(r.content)

        base_dir = os.path.dirname(fname)
        data_dir, ext = os.path.splitext(fname)

        fp = zipfile.ZipFile(fname, 'r')
        fp.extractall(base_dir)
        print('Done!')
        return data_dir

    data_dir = download_and_extract_data(data_url)
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv(os.path.join(data_dir, 'u.data'), '\t',
                       names=names, engine='python')
    movies = pd.read_csv(os.path.join(data_dir, 'u.item'), '\t',
                         names=['movies'], engine='python')
    num_users = data.user_id.unique().shape[0]
    num_items = data.item_id.unique().shape[0]
    return data, movies, num_users, num_items


# prepare movie rating data for inputing in the sparse matrix
def get_movies_ratings(movies):
    '''
    This function takes movies dataframe as argument, processes each row by
    splitting on the delimiter '|' and gets the movie id and the movie name
    
    returns:
    
    a dataframe consiting of movie id and movie name
    
    '''
    # data, movies, num_users, num_items = read_data_ml100k()
    res = []
    id = []
    for row in movies['movies']:
        movie_id = row.split('|')[0]
        id.append(int(movie_id))
        movie = row.split('|')[1]
        movie = movie.split('(')[0]
        res.append(movie)
    return pd.DataFrame({"item_id": id, "name": res},
                        columns=["item_id", "name"])


# run the program
def main():
    '''
    This function executes the steps sequentially and writes
    data and movies into pickle files to be used later    
    '''
    data, movies, num_users, num_items = read_data_ml100k()
    movies = get_movies_ratings(movies)
    directory = './output'
    if not os.path.exists(directory):
        os.makedirs(directory)
    data.to_pickle("./output/ratings.pkl")
    movies.to_pickle("./output/movies.pkl")


if __name__ == "__main__":
    main()
