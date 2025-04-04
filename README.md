# Movie Recommendation System with FastAPI

This project implements a movie recommendation system that suggests similar movies based on clustering and feature similarity. The application leverages pre-trained machine learning models to cluster movies and returns recommendations through a FastAPI-based RESTful API.

> **Data Source:**  
> The movie data is derived from [MovieLens](https://movielens.org/). Make sure that your dataset complies with MovieLens usage terms.

## Overview

The system processes a CSV file containing movie data with various features (e.g., genres, ratings). It uses the following pre-trained components:
- **KMeans Model:** Clusters movies into groups using scikit‑learn.
- **StandardScaler:** Normalizes movie feature data.
- **MultiLabelBinarizer:** Encodes movie genre information.

When a movie is requested via its TMDB ID, the application:
1. Loads the corresponding movie's features.
2. Scales the features and determines its cluster.
3. Computes cosine similarity between the movie and others in the same cluster.
4. Returns similar movies based on their similarity scores.

## Project Structure

```
Movie-Recommendation-System/
└── ML/
    ├── main.py               # FastAPI application and recommendation logic
    ├── movie.csv             # Dataset with movie information (MovieLens data)
    ├── kmeans_model.pkl      # Pre-trained KMeans model
    ├── scaler.pkl            # Pre-trained StandardScaler
    ├── mlb.pkl               # Pre-trained MultiLabelBinarizer for genre encoding
    ├── requirements.txt      # Project Python dependencies
    └── .gitignore            # Git ignore configuration
```

## Requirements

This project uses the following primary libraries:
- **FastAPI** – For building the RESTful API
- **uvicorn** – ASGI server to run the FastAPI app
- **pandas** – For data manipulation and CSV handling
- **numpy** – For numerical operations
- **scikit-learn** – For pre-trained machine learning models (KMeans, StandardScaler, and cosine similarity)

## Local Setup & Usage

1. **Clone the Repository:**

   ```bash
   git clone <repository-url>
   cd Movie-Recommendation-System/ML
   ```

2. **Create and Activate the Virtual Environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate        # For Linux/macOS
   venv\Scripts\activate           # For Windows
   ```

3. **Install Project Dependencies:**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Run the Application Locally:**

   Start the FastAPI application using uvicorn:

   ```bash
   uvicorn main:app --reload
   ```

   The API will be available at [http://localhost:8000](http://localhost:8000).

5. **API Endpoint:**

   To retrieve movie recommendations, send a GET request to the following endpoint:
   
   ```
   GET /recommend?tmdb_id=<TMDB_ID>&top_n=<NUMBER_OF_RECOMMENDATIONS>
   ```

   For example:
   
   ```
   http://localhost:8000/recommend?tmdb_id=862&top_n=10
   ```

## Additional Information

- **Warnings:**  
  You might see warnings related to mismatched scikit‑learn versions when loading pickled models. These are informational and do not affect functionality, but aligning versions between training and serving environments is recommended.

- **Customization:**  
  You can adjust the feature columns and similarity logic in `main.py` to fit different datasets or recommendation criteria.

Happy recommending!
