import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer

desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',10)

# Removal of stop-words
tfidf = TfidfVectorizer(stop_words="english")
count = CountVectorizer(stop_words='english')

# Read the TMDB datasets
df1 = pd.read_csv("../../data/tmdb-movie-metadata/tmdb_5000_credits.csv")
df2 = pd.read_csv("../../data/tmdb-movie-metadata/tmdb_5000_movies.csv")

# Let us merge both dataframe on 'id'
df2 = df2.merge(df1, left_on='id', right_on='movie_id', how='left')

"""Part1: Plot Description Based Recommender"""

# Replace NaN with empty string in 'overview' column of df2
df2['overview'] = df2['overview'].fillna('')

tfidf_matrix = tfidf.fit_transform(df2['overview'])
print("movies x words: {}".format(tfidf_matrix.shape))

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Construct a reverse map of indices and movie titles
indices = pd.Series(df2.index, index=df2['title_y']).drop_duplicates()

# Function to take in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # get index of the title
    index_no = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    similarity_scores = list(enumerate(cosine_sim[index_no]))

    # Sort the movies based on similarity score
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Return the movie names
    return df2['title_y'].iloc[[i[0] for i in similarity_scores[1:11]]]

print("Other movies similar to Jaws plot:\n{}".format(get_recommendations('Jaws')))

"""Part 2: Credits, Genres and Keywords Based Recommender"""

# Parse the unstructured features into their corresponding python objects
features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(literal_eval)

# Get the director's name from the crew feature. If director is not listed, return NaN
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

# Returns the list top 3 elements or entire list; whichever is more.
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        # Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names
    # Return empty list in case of missing/malformed data
    return []


# Define new director, cast, genres and keywords features that are in a suitable form.
df2['director'] = df2['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(get_list)

# Print the new features of the first 3 films
print(df2[['title_y', 'cast', 'director', 'keywords', 'genres']].head(3))

# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

# Apply clean_data function to your features.
features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    df2[feature] = df2[feature].apply(clean_data)

# Join all features
def join_all(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

df2['metadata'] = df2.apply(join_all, axis=1)

# Apply countvectorizer
count_matrix = count.fit_transform(df2['metadata'])

# Compute the cosine similarity matrix
cosine_sim2 = linear_kernel(count_matrix, count_matrix)

# Now get recommendations
print("Other movies similar to Spectre plot:\n{}".format(get_recommendations('Spectre', cosine_sim2)))
