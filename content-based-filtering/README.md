# Content Based Filtering

In this recommender system the content of the movie (overview, cast, crew, keyword, tagline etc) is used to find its similarity with 
other movies. Then the movies that are most likely to be similar are recommended.

## Part 1: Plot Description Based Recommender
Here we will compute pairwise similarity scores for all movies based on their plot descriptions and recommend movies based on 
that similarity score. The plot is given in the **overview** feature from TMDB movie dataset.

### TF-IDF vectorization

Now we'll compute Term Frequency-Inverse Document Frequency (TF-IDF) vectors for each overview.
Now if you are wondering what is term frequency , it is the relative frequency of a word in a document and is given as (term 
instances/total instances). Inverse Document Frequency is the relative count of documents containing the term is given as 
log(number of documents/documents with term) The overall importance of each word to the documents in which they appear is 
equal to TF * IDF.

This will give you a matrix where each column represents a word in the overview vocabulary (all the words that appear in at least 
one document) and each column represents a movie, as before.This is done to reduce the importance of words that occur frequently 
in plot overviews and therefore, their significance in computing the final similarity score.

Fortunately, scikit-learn gives you a built-in TfIdfVectorizer class that produces the TF-IDF matrix in a couple of lines.

```sh
# Replace NaN with an empty string
df2['overview'] = df2['overview'].fillna('')

# Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df2['overview'])

print(tfidf_matrix,shape)
# (4803, 20978)
```

We see that over 20,000 different words were used to describe the 4800 movies in our dataset.
With this matrix in hand, we can now compute a similarity score. There are several candidates for this; such as the euclidean, 
the Pearson and the cosine similarity scores. There is no right answer to which score is the best. Different scores work well 
in different scenarios and it is often a good idea to experiment with different metrics.

We will be using the cosine similarity to calculate a numeric quantity that denotes the similarity between two movies. We use 
the cosine similarity score since it is independent of magnitude and is relatively easy and fast to calculate. 

```sh
# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
```

We are going to define a function that takes in a movie title as an input and outputs a list of the 10 most similar movies. 
Firstly, for this, we need a reverse mapping of movie titles and DataFrame indices. In other words, we need a mechanism to identify 
the index of a movie in our metadata DataFrame, given its title.

#### Recommendation function

We are now in a good position to define our recommendation function. These are the following steps we'll follow :-

1. Get the index of the movie given its title.
2. Get the list of cosine similarity scores for that particular movie with all movies. Convert it into a list of tuples where the 
first element is its position and the second is the similarity score.
3. Sort the aforementioned list of tuples based on the similarity scores; that is, the second element.
4. Get the top 10 elements of this list. Ignore the first element as it refers to self (the movie most similar to a particular movie 
is the movie itself).
5. Return the titles corresponding to the indices of the top elements.

```sh
# Construct a reverse map of indices and movie titles
indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()

# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df2['title'].iloc[movie_indices]
    
print("Other movies similar to Jaws plot:\n{}".format(get_recommendations('Jaws')))
```

Output: 
```sh
Other movies similar to Jaws plot:
2003             Jaws: The Revenge
2094                        Jaws 2
3124                      The Reef
3948                      Creature
2742                  The Shallows
1437                          Bait
525                     Shark Tale
1911                   Shark Night
2816    Ace Ventura: Pet Detective
3150                   Heavy Metal
```
While our system has done a decent job of finding movies with similar plot descriptions, the quality of recommendations is not that 
great. "Jaws" returns all shark movies while it is more likely that the people who liked that movie are more inclined to enjoy other 
James Cameroon movies. This is something that cannot be captured by the Plot Based Recommendation System.

## Part 2: Credits, Genres and Keywords Based Recommender

