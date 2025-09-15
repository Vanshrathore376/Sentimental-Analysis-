from flask import Flask, render_template, request
import os
import pickle

app = Flask(__name__)

# ------------------------------
# Load model and vectorizer
# ------------------------------
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_folder = os.path.join(root_path, "model")

# 👉 Change filename here if you want to use Naive Bayes
model_path = os.path.join(model_folder, "sentiment_model_log.pkl")
vectorizer_path = os.path.join(model_folder, "vectorizer.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

# ------------------------------
# Routes
# ------------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review'].strip()   # remove extra \n or spaces
    review_vectorized = vectorizer.transform([review])  # transform input
    prediction = model.predict(review_vectorized)[0]   # get label
    return render_template('index.html', review=review, prediction=prediction)

# ------------------------------
# Run app
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True)
