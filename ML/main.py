from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn

app = FastAPI()

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Configuration ---
KMEANS_MODEL_PATH = 'kmeans_model.pkl'
SCALER_PATH = 'scaler.pkl'
MLB_PATH = 'mlb.pkl'
MOVIES_DATA_PATH = 'movie.csv'
RELEVANT_COLUMNS = [
    'rating', '(no genres listed)', 'Action', 'Adventure', 'Animation', 'Children',
    'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
    'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]

# --- Model and Data Loading ---
def load_model(path):
    """Loads a pickle model from the specified path."""
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise RuntimeError(f"Model file not found at {path}")
    except Exception as e:
        raise RuntimeError(f"Error loading model from {path}: {e}")

def load_movie_data(path):
    """Loads and preprocesses the movie dataset from a CSV file."""
    try:
        movies = pd.read_csv(path)
        movies['tmdbId'] = pd.to_numeric(movies['tmdbId'], errors='coerce')
        movies['imdbId'] = pd.to_numeric(movies['imdbId'], errors='coerce')
        movies = movies.dropna(subset=['tmdbId', 'imdbId'])
        movies['tmdbId'] = movies['tmdbId'].astype(int)
        movies['imdbId'] = movies['imdbId'].astype(int)
        
        # Ensure all relevant columns exist
        for col in RELEVANT_COLUMNS:
            if col not in movies.columns:
                movies[col] = 0

        # Optimization: The CSV contains multiple user ratings per movie.
        # This causes the DataFrame to be 100k+ rows instead of ~9k unique movies.
        # We group by tmdbId so each movie has exactly 1 row, averaging the rating.
        agg_funcs = {col: 'max' for col in RELEVANT_COLUMNS if col != 'rating'}
        agg_funcs['rating'] = 'mean'
        # Keep necessary metadata columns
        for col in ['imdbId', 'title']:
            if col in movies.columns:
                agg_funcs[col] = 'first'
        
        movies = movies.groupby('tmdbId', as_index=False).agg(agg_funcs)

        return movies
    except FileNotFoundError:
        raise RuntimeError(f"Movie data file not found at {path}")
    except Exception as e:
        raise RuntimeError(f"Error loading movie data from {path}: {e}")

# Load all models and data at startup
kmeans_model = load_model(KMEANS_MODEL_PATH)
scaler = load_model(SCALER_PATH)
mlb = load_model(MLB_PATH)
movies = load_movie_data(MOVIES_DATA_PATH)

# Pre-compute scaled features and clusters to save time during requests
print("Pre-computing features for faster recommendations...")
_all_features = movies[RELEVANT_COLUMNS].to_numpy()
_scaled_features = scaler.transform(_all_features)
movies['Cluster'] = kmeans_model.predict(_scaled_features)
# Store the scaled features directly in the dataframe for instant lookup
movies['scaled_features'] = list(_scaled_features)

# --- Recommendation Logic ---

def get_movie_features(movie_id, movies_df):
    """Extracts features for a given movie ID."""
    movie_row = movies_df[movies_df['tmdbId'] == movie_id]
    if movie_row.empty:
        return None
    return movie_row.iloc[0]['scaled_features'].reshape(1, -1)

def get_cluster_movies(cluster_id, movies_df):
    """Gets all movies from a specific cluster."""
    return movies_df[movies_df['Cluster'] == cluster_id]

def recommend_movies(movie_tmdb_id, movies_df, kmeans_model, scaler_model, top_n=10):
    """
    Recommends movies similar to a given movie.

    Args:
        movie_tmdb_id (int): The TMDB ID of the movie to get recommendations for.
        movies_df (pd.DataFrame): The DataFrame of all movies.
        kmeans_model: The trained KMeans model.
        scaler_model: The trained scaler model.
        top_n (int): The number of recommendations to return.

    Returns:
        pd.DataFrame: A DataFrame of recommended movies.
    """
    movie_scaled = get_movie_features(movie_tmdb_id, movies_df)
    if movie_scaled is None:
        return f"Movie with TMDB ID '{movie_tmdb_id}' not found in dataset."

    # Using the pre-calculated cluster
    movie_cluster = movies_df[movies_df['tmdbId'] == movie_tmdb_id].iloc[0]['Cluster']

    cluster_movies = get_cluster_movies(movie_cluster, movies_df).copy()
    cluster_movies.reset_index(drop=True, inplace=True)

    # Use the pre-calculated scaled features
    cluster_scaled_features = np.stack(cluster_movies['scaled_features'].values)
    
    similarities = cosine_similarity(movie_scaled, cluster_scaled_features).flatten()
    sorted_indices = np.argsort(similarities)[::-1]

    recommended_movies = []
    for idx in sorted_indices:
        if len(recommended_movies) >= top_n:
            break
        
        # Check if idx is within the bounds of the cluster_movies DataFrame
        if idx < len(cluster_movies):
            movie_id = cluster_movies.iloc[idx]['tmdbId']
            if movie_id != movie_tmdb_id:
                recommended_movies.append(cluster_movies.iloc[idx])

    if not recommended_movies:
        return f"No similar movies found for TMDB ID '{movie_tmdb_id}'. Try a different movie."

    # Return without the massive scaled_features array
    result_df = pd.DataFrame(recommended_movies)[['tmdbId', 'rating']]
    if 'title' in cluster_movies.columns:
        result_df['title'] = pd.DataFrame(recommended_movies)['title']
        
    return result_df.drop_duplicates(subset='tmdbId')


# --- API Endpoints ---

@app.get("/recommend")
async def get_recommendations(
    tmdb_id: int = Query(..., description="TMDB ID of the movie"),
    top_n: int = Query(10, description="Number of recommendations to return")
):
    """
    Get movie recommendations based on a given TMDB movie ID.
    """
    result = recommend_movies(tmdb_id, movies, kmeans_model, scaler, top_n=top_n)
    if isinstance(result, str):
        raise HTTPException(status_code=404, detail=result)
    return result.to_dict(orient='records')

# To run the app: uvicorn main:app --reload
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)