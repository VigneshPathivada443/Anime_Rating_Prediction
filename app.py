from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model & encoder
model = joblib.load("model.pkl")
encoder = joblib.load("encoder.pkl")

feature_names = encoder.get_feature_names_out(['genre', 'type'])

def predict_rating(genre, anime_type, episodes, members):
    input_df = pd.DataFrame({
        'genre': [genre],
        'type': [anime_type],
        'episodes': [int(episodes)],
        'members': [int(members)]
    })

    encoded = encoder.transform(input_df[['genre', 'type']])
    encoded_df = pd.DataFrame(encoded, columns=feature_names)

    final_input = pd.concat(
        [encoded_df, input_df[['episodes', 'members']].reset_index(drop=True)],
        axis=1
    )

    prediction = model.predict(final_input)[0]
    return round(prediction, 2)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        genre = request.form['genre']
        anime_type = request.form['anime_type']
        episodes = request.form['episodes']
        members = request.form['members']

        rating = predict_rating(genre, anime_type, episodes, members)

        return render_template("result.html", predicted_rating=rating)

    except Exception as e:
        return render_template("error.html", error_message=str(e))

if __name__ == "__main__":
    app.run(debug=True)
