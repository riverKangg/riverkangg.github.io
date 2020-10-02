# -*- coding: utf-8 -*-
"""item-based-CF.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rQXdGFtPqRCdkWLNaEQJsmFOmltXNTMn
"""

import zipfile
from google.colab import drive

drive.mount('/content/drive/')

zip_ref = zipfile.ZipFile("/content/drive/My Drive/Colab Notebooks/data/movieLens.zip", 'r')
zip_ref.extractall("/tmp")
zip_ref.close()

import os
arr = os.listdir('/tmp')
print(arr)

import pandas as pd
rating_data = pd.read_csv('/tmp/rating.csv')
rating_data.drop('timestamp', axis=1, inplace=True)
rating_data.head()

movie_data = pd.read_csv('/tmp/movie.csv')
movie_data.head()

user_movie_rating = pd.merge(rating_data, movie_data, on='movieId')
user_movie_rating.head()

user_movie_rating = user_movie_rating[:len(user_movie_rating)//2]

movie_user_rating = user_movie_rating.pivot_table(values='rating', index='title', columns='userId').fillna(0)
movie_user_rating.head()

user_movie_rating = user_movie_rating.pivot_table(values='rating', index='userId', columns='title').fillna(0)
user_movie_rating.head()

from sklearn.metrics.pairwise import cosine_similarity
item_based_collabor = cosine_similarity(movie_user_rating)
item_based_collabor

item_based_collabor = pd.DataFrame(data=item_based_collabor, index=movie_user_rating.index, columns=movie_user_rating.index)
item_based_collabor.head()

def get_item_based_collabor(title):
    return item_based_collabor[title].sort_values(ascending=False)[:6]

get_item_based_collabor('Godfather, The (1972)')