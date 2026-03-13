import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

# print(movies.head())
# print(movies.info())

# print(credits.head())
# print(credits.info())

movies = movies.merge(credits, on='title')

movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
# print(movies.head())
# print(movies.isnull().sum())
movies.dropna(inplace=True)
# print(movies.duplicated().sum())

# print(movies.iloc[0].genres)

# [{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]
# we have to convert structure of genres as shown above to:
# ['Action', 'Adventure', 'Fantasy', 'SciFi]

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
# print(movies.head())
movies['keywords'] = movies['keywords'].apply(convert)

def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break        
    return L

movies['cast'] = movies['cast'].apply(convert)

# print(movies['crew'][0])

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['crew'] = movies['crew'].apply(fetch_director)
# print(movies.head())

# print(movies['overview'][0])
movies['overview'] = movies['overview'].apply(lambda x:x.split())
# print(movies['overview'])

movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])
# print(movies.head())

movies['tags'] = movies['overview'] + movies['cast'] + movies['crew'] + movies['genres'] + movies['keywords']
# print(movies.head())

new_df = movies[['movie_id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))
# print(new_df['tags'][0])

new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())


        # // stemming  ['danced', 'dancing', 'dance'] -> ['danc']
ps = PorterStemmer()
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    
    return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)

        # // Count Vectorization
cv = CountVectorizer(stop_words='english', max_features=5000)
vectors = cv.fit_transform(new_df['tags']).toarray()
# print(cv.get_feature_names_out()) 

similarity = cosine_similarity(vectors)     # using cosine similarity
sorted(list(enumerate(similarity[0])), reverse=True, key=lambda x:x[1])[1:6]

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]

    for i in movies_list:
        print(new_df.iloc[i[0]].title)

# print(recommend('Batman Begins'))

# pickle.dump(new_df, open('movies.pkl', 'wb'))
# pickle.dump(new_df.to_dict(), open('movie_dict.pkl', 'wb'))
# pickle.dump(similarity, open('similarity.pkl', 'wb'))