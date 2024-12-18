import pandas as pd
import numpy as np
import requests

# Define the URL for movie data
myurl = "https://liangfgithub.github.io/MovieData/movies.dat?raw=true"

# Fetch the data from the URL
response = requests.get(myurl)

# Split the data into lines and then split each line using "::"
movie_lines = response.text.split('\n')
movie_data = [line.split("::") for line in movie_lines if line]

# Create a DataFrame from the movie data
movies = pd.DataFrame(movie_data, columns=['MovieID', 'title', 'genres'])
movies['MovieID'] = movies['MovieID'].astype(int)

genres = list(
    sorted(set([genre for genres in movies.genres.unique() for genre in genres.split("|")]))
)

top10_movie = pd.read_csv('https://github.com/cqjw/MovieRecommendApp/raw/main/top10_rated_movies.csv')
new_similarity_df = pd.read_csv('https://github.com/cqjw/MovieRecommendApp/raw/main/new_similarity_matrix.csv', index_col=0)

def get_displayed_movies():
    """Return the first 100 movies for display."""
    return movies.head(100)

def get_top10_popular_movies(movies = movies): # return top10 movies by ave rating
    #res = top10_movie['MovieID'].apply(lambda x: f"m{x}")
    res = top10_movie['MovieID']
    #print('res', res, movies.head())
    ordered_movies = movies[movies['MovieID'].isin(res.values)]
    
    # Reorder based on the index of res values
    ordered_movies = ordered_movies.set_index('MovieID')
    ordered_movies = ordered_movies.loc[res.values]  # Reorder movies according to res
    ordered_movies = ordered_movies.reset_index()

    return ordered_movies

def myIBCF(newuser, s_matrix = new_similarity_df, movie_popularity = top10_movie, movies = movies, top_n=10 ):
    if len(newuser)==0:
        return get_top10_popular_movies()
    # Create a copy of the new user's ratings and convert them to a 1D array
    w = pd.Series(newuser.copy())
    w = w.reindex(s_matrix.columns)
    # Movies already rated by the new user (non-NA entries in w)
    rated_movies = w.dropna().index
    
    # Prepare the prediction array
    predictions = np.full(w.shape, np.nan)
    #print(w.index)
    # Iterate through each movie to make predictions
    for movie_id in w.index: # Iterate using movie IDs
        if pd.isna(w[movie_id]):  # If the movie is not rated by the user
            # Find the set of similar movies that the user has rated
            similar_movies = s_matrix.loc[movie_id,:].dropna()  # similarity scores for movie i
            common_ratings = w[similar_movies.index].dropna()  # user ratings for similar movies
            
            if len(common_ratings) >= 1:  # At least 1 common ratings required
                numerator = np.sum(similar_movies[common_ratings.index] * common_ratings)
                denominator = np.sum(np.abs(similar_movies[common_ratings.index]))
                if denominator != 0:
                    predictions[w.index.get_loc(movie_id)] = numerator / denominator # Get index location using .get_loc
    
    # Now, get the top-n recommendations
    top_recommendations = pd.Series(predictions, index=w.index).nlargest(top_n)
    #print('top',top_recommendations)
    # If there are fewer than top_n predictions, add popular movies
    if len(top_recommendations) < top_n:
        # Get the remaining number of recommendations
        remaining = top_n - len(top_recommendations)
        
        # Exclude movies that have already been rated by the user
        unranked_movies = movie_popularity[~movie_popularity['MovieID'].isin(rated_movies)].head(remaining)
        #unranked_movies['MovieID'] = unranked_movies['MovieID'].apply(lambda x: f"m{x}")

        #print(unranked_movies.head())
        # Add the unranked movies to the top recommendations
        additional_recommendations = unranked_movies['MovieID']
        #print('add', additional_recommendations)
        top_recommendations = pd.concat([top_recommendations, pd.Series(additional_recommendations, index=additional_recommendations)])

    #recom = top_recommendations.index.to_frame(index=False)
    #recom.columns = ['MovieID']
    #print(top_recommendations)
    recom = top_recommendations.index.str.lstrip('m').astype(int)
    #recom['MovieID'] = recom['MovieID'].str.lstrip('m').astype(int)
    ordered_movies = movies[movies['MovieID'].isin(recom.values)]
    ordered_movies = ordered_movies.set_index('MovieID')
    ordered_movies = ordered_movies.loc[recom.values]  # Reorder movies according to res
    ordered_movies = ordered_movies.reset_index()

    return ordered_movies
    #return movies[movies['MovieID'].isin(recom['MovieID'].values)]

def get_recommended_movies(new_user_ratings, s_matrix = new_similarity_df, movie_popularity = top10_movie):
    return myIBCF(pd.Series(new_user_ratings, dtype = 'float64'), s_matrix, movie_popularity, movies, top_n=10 )
