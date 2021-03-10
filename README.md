![alt text](https://travis-ci.org/sharmeelee/MovieRecommender.svg?branch=main)

# IMDB Movie Recommender System
Group: Priyanka Bijlani, Sharmeelee Bijlani, Laura Thriftwood, Lakshmi Venkatasubramanian
## Introduction
When considering which movie to watch, users have access to an overwhelming number of options. Users want custom recommendations to ensure optimal use of their time watching content. Business models benefit from strong recommender systems by increasing user engagement and addiction to streaming platforms. 
With this project, we can create our own movie recommendation system that takes user input of one movie and utilizes a rich dataset of movie titles, ratings and user information to output a recommended movie. 
### Data
Streaming Data - [MovieLens | GroupLens ](https://grouplens.org/datasets/movielens/100k/)
- Over 100k ratings
- 1700+ movie titles
- 1000+ users
### Directory Structure
### Use Cases
1. User will get a movie recommendation from the system based on their previous ratings
- Training input: users, movies, ratings
- User input: user name/id
- Outputs: movie(s)
- ML algorithm: Collaborative filtering (analyzes historical data)

2. User will be able to provide a movie name and get similar movies
- Training input: users, movies, ratings
- User input: movie
- Outputs: movie(s)
- ML algorithm: Collaborative filtering 

3. User will be able to predict the rating of a given movie based on past ratings by the same user
- Training input: users, movies, ratings
- User input: User, movie
- Outputs: rating
- ML algorithm: Collaborative filtering 
## Running the Program
### Installation Instructions
Run the following commands.
Note: Python 3.8 required.
```
git clone https://github.com/sharmeelee/MovieRecommender.git
```
```
cd MovieRecommender
```
```
python3 setup.py install
```
