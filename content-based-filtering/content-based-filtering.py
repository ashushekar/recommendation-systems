import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Removal of stop-words
tfidf = TfidfVectorizer(stop_words="english")

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