from fastapi import FastAPI, Query, HTTPException
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load your pre-trained models and encoders from pickle files
with open('kmeans_model.pkl', 'rb') as f:
    kmeans_model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('mlb.pkl', 'rb') as f:
    mlb = pickle.load(f)

# Load the movies dataset.
movies = pd.read_csv('movie.csv')
movies['tmdbId'] = pd.to_numeric(movies['tmdbId'], errors='coerce')
movies['imdbId'] = pd.to_numeric(movies['imdbId'], errors='coerce')
movies = movies.dropna(subset=['tmdbId', 'imdbId'])
movies['tmdbId'] = movies['tmdbId'].astype(int)
movies['imdbId'] = movies['imdbId'].astype(int)

# def cosine_similarity_custom(A, B):
#     """
#     Compute the cosine similarity between each row of A and each row of B.
    
#     Parameters:
#         A (np.ndarray): shape (m, n)
#         B (np.ndarray): shape (p, n)
    
#     Returns:
#         np.ndarray: similarity matrix of shape (m, p)
#     """
#     # Compute dot products between rows of A and rows of B
#     dot_product = A.dot(B.T)
#     # Compute the L2 norms for rows of A and B
#     norm_A = np.linalg.norm(A, axis=1, keepdims=True)
#     norm_B = np.linalg.norm(B, axis=1, keepdims=True)
#     # Compute outer product of norms to get denominator
#     denominator = norm_A.dot(norm_B.T)
#     # Return elementwise division (handle potential division by zero if necessary)
#     return dot_product / denominator

def recommend_movies(movie_tmdb_id, movies_df, kmeans_model, scaler_model, mlb_model, top_n=10):
    from sklearn.metrics.pairwise import cosine_similarity
    relevant_columns = ['rating','(no genres listed)', 'Action', 'Adventure', 'Animation', 'Children', 
                        'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
                        'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    
    # Ensure missing feature columns exist
    for col in relevant_columns:
        if col not in movies_df.columns:
            movies_df[col] = 0

    movie_row = movies_df[movies_df['tmdbId'] == movie_tmdb_id]
    if movie_row.empty:
        return f"Movie with TMDB ID '{movie_tmdb_id}' not found in dataset."

    movie_features = movie_row[relevant_columns].copy()
    movie_features_np = movie_features.to_numpy()
    movie_scaled = scaler_model.transform(movie_features_np)
    movie_cluster = kmeans_model.predict(movie_scaled)[0]
    
    cluster_movies = movies_df[movies_df['Cluster'] == movie_cluster].copy()
    for col in relevant_columns:
        if col not in cluster_movies.columns:
            cluster_movies[col] = 0

    cluster_movie_features = cluster_movies[relevant_columns].copy()
    cluster_movie_features_np = cluster_movie_features.to_numpy()
    # similarities = cosine_similarity_custom(movie_scaled, scaler_model.transform(cluster_movie_features_np)).flatten()
    similarities = cosine_similarity(movie_scaled, scaler_model.transform(cluster_movie_features_np)).flatten()
    sorted_indices = np.argsort(similarities)[::-1]
    
    unique_indices = []
    for idx in sorted_indices:
        if idx < len(cluster_movies) and cluster_movies.iloc[idx]['tmdbId'] != movie_tmdb_id:
            unique_indices.append(idx)
        if len(unique_indices) >= top_n:
            break

    if not unique_indices:
        return f"No similar movies found for TMDB ID '{movie_tmdb_id}'. Try a different movie."
    
    # Obtain recommendations and drop duplicates by tmdbId
    similar_movies = cluster_movies.iloc[unique_indices][['tmdbId', 'rating']]
    similar_movies = similar_movies.drop_duplicates(subset='tmdbId')
    return similar_movies

@app.get("/recommend")
async def get_recommendations(
    tmdb_id: int = Query(..., description="TMDB ID of the movie"),
    top_n: int = Query(10, description="Number of recommendations to return")
):
    result = recommend_movies(tmdb_id, movies, kmeans_model, scaler, mlb, top_n=top_n)
    if isinstance(result, str):
        raise HTTPException(status_code=404, detail=result)
    return result.to_dict(orient='records')

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)