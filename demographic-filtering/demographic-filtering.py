import pandas as pd
import matplotlib.pyplot as plt

# Read the TMDB datasets
df1 = pd.read_csv("../../data/tmdb-movie-metadata/tmdb_5000_credits.csv")
df2 = pd.read_csv("../../data/tmdb-movie-metadata/tmdb_5000_movies.csv")

# Let us merge both dataframe on 'id'
df2 = df2.merge(df1, left_on='id', right_on='movie_id', how='left')

# Calculate mean vote across the whole report
C = df2['vote_average'].mean()

# Calculate minimum votes required to be listed in chart
m = df2['vote_count'].quantile(0.9)

# Now let us filter out movies which qualifies the chart
qualified_movies = df2.copy().loc[df2['vote_count'] > m]

# Calculate Weighted-rating
def weightedrating(x, m=m, C=C):
    R = x['vote_average']
    v = x['vote_count']
    return (R * (v/(v + m))) + (C * (m/(v + m)))

qualified_movies['score'] = qualified_movies.apply(weightedrating, axis=1)

#Sort movies based on score calculated above
qualified_movies = qualified_movies.sort_values('score', ascending=False)

#Print the top 15 movies
print(qualified_movies[['title_y', 'vote_count', 'vote_average', 'score']].head(10))

pop= df2.sort_values('popularity', ascending=False)
plt.figure(figsize=(12,4))
plt.barh(pop['title_y'].head(20),pop['popularity'].head(20), align='center', color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel("Popularity")
plt.title("Popular Movies")
plt.show(block=True)