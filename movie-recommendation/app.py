from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

# Load datasets
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Simple recommendation based on user ratings
def recommend_movies(user_id):
    user_ratings = ratings[ratings['userId'] == user_id]
    if user_ratings.empty:
        return movies.sample(5)  # Return random movies if no ratings

    rated_movies = user_ratings['movieId'].tolist()
    recommended_movies = movies[~movies['movieId'].isin(rated_movies)]
    return recommended_movies.sample(5)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.form['user_id'])
    recommended = recommend_movies(user_id)
    return render_template('results.html', movies=recommended.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
