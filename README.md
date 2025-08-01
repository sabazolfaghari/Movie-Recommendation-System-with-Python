# Basic Movie Recommendation System with Python

A **content-based movie recommender system** that suggests movies similar to a user’s favorite film.  
It uses **TF-IDF (Term Frequency–Inverse Document Frequency)** and **cosine similarity** to compare metadata features such as **genres, keywords, tagline, cast, and director**.

## Dataset

This project uses the movies.csv dataset which is included to the repository. 
Some of the key columns include:

- `index` → Unique index for each movie  
- `title` → Movie title  
- `genres` → Genres of the movie (e.g., Action, Adventure, Fantasy)  
- `keywords` → Keywords describing the theme/story  
- `tagline` → Promotional tagline for the movie  
- `cast` → Main actors  
- `director` → Director’s name  
- *(other columns like `budget`, `revenue`, `release_date` are present but not used in this project)*

## How It Works
1. **Preprocessing**
   - Selects relevant features: `genres`, `keywords`, `tagline`, `cast`, `director`
   - Fills null values with empty strings
   - Combines them into a single text field

2. **Feature Extraction**
   - Applies **TF-IDF Vectorizer** to transform text into numerical vectors

3. **Similarity Calculation**
   - Computes pairwise **cosine similarity** between all movies

4. **Recommendation**
   - User inputs a favorite movie title
   - Finds closest match in the dataset
   - Returns the **top 10 most similar movies**

## Example Run

```bash
$ python movie_recommender.py
 Enter your favourite movie name : Despicable Me
