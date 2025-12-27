import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. SETUP DATASET
# Creating a sample dataset of movies
movies_data = {
    'movie_id': [1, 2, 3, 4, 5],
    'title': ['The Dark Knight', 'Inception', 'The Matrix', 'Toy Story', 'Finding Nemo'],
    'genres': ['Action Crime Drama', 'Action Sci-Fi Thriller', 'Action Sci-Fi', 'Animation Adventure Comedy', 'Animation Adventure']
}

# Creating sample user ratings for Collaborative Filtering
ratings_data = {
    'user_id': [1, 1, 2, 2, 3, 3, 4, 4],
    'movie_id': [1, 2, 1, 3, 4, 5, 2, 3],
    'rating': [5, 4, 4, 5, 5, 4, 2, 1]
}

df_movies = pd.DataFrame(movies_data)
df_ratings = pd.DataFrame(ratings_data)

# --- APPROACH 1: CONTENT-BASED FILTERING ---
def content_based_recommendations(movie_title, df):
    # Initialize TF-IDF Vectorizer to convert genres into numbers
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['genres'])
    
    # Compute Cosine Similarity between all movies
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Get index of the movie that matches the title
    idx = df[df['title'] == movie_title].index[0]
    
    # Get pairwise similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the top 2 most similar movies (excluding itself)
    sim_scores = sim_scores[1:3]
    movie_indices = [i[0] for i in sim_scores]
    
    return df['title'].iloc[movie_indices]

# --- APPROACH 2: COLLABORATIVE FILTERING (User-Item) ---
def collaborative_recommendations(user_id, df_ratings, df_movies):
    # Create a User-Item Matrix
    user_item_matrix = df_ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
    
    # Compute similarity between users
    user_sim = cosine_similarity(user_item_matrix)
    user_sim_df = pd.DataFrame(user_sim, index=user_item_matrix.index, columns=user_item_matrix.index)
    
    # Find users most similar to the target user
    similar_users = user_sim_df[user_id].sort_values(ascending=False).index[1]
    
    # Suggest movies that the similar user liked but the target user hasn't seen
    user_seen_movies = df_ratings[df_ratings['user_id'] == user_id]['movie_id'].values
    similar_user_movies = df_ratings[df_ratings['user_id'] == similar_users]
    
    recommendations = similar_user_movies[~similar_user_movies['movie_id'].isin(user_seen_movies)]
    return df_movies[df_movies['movie_id'].isin(recommendations['movie_id'])]['title']

# --- TESTING THE SYSTEM ---
print("--- Content-Based Recommendations for 'The Dark Knight' ---")
print(content_based_recommendations('The Dark Knight', df_movies))

print("\n--- Collaborative Recommendations for User 1 ---")
print(collaborative_recommendations(1, df_ratings, df_movies))