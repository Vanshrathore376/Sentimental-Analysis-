import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle
import os

# ------------------------------
# Load dataset
# ------------------------------
data_path = "data/reviews.csv"  # Change path if needed
df = pd.read_csv(data_path)

# ------------------------------
# Text preprocessing
# ------------------------------
df['review'] = df['review'].str.lower()  # lowercase
df['review'] = df['review'].str.replace(r'[^\w\s]', '', regex=True)  # remove punctuation

X = df['review'].values
y = df['label'].values  # 0 = negative, 1 = positive

# ------------------------------
# Split dataset
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------
# Vectorizer
# ------------------------------
vectorizer = TfidfVectorizer(max_features=5000)  # more features for better learning
X_train_transformed = vectorizer.fit_transform(X_train)
X_test_transformed = vectorizer.transform(X_test)

# ------------------------------
# Train Multinomial Naive Bayes
# ------------------------------
nb_model = MultinomialNB()
nb_model.fit(X_train_transformed, y_train)

# ------------------------------
# Evaluate
# ------------------------------
y_pred = nb_model.predict(X_test_transformed)
accuracy = accuracy_score(y_test, y_pred)
print(f"Naive Bayes Accuracy: {accuracy * 100:.2f}%")

# ------------------------------
# Save model and vectorizer
# ------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_folder = os.path.join(project_root, "model")
os.makedirs(model_folder, exist_ok=True)

with open(os.path.join(model_folder, "sentiment_model_nb.pkl"), "wb") as f:
    pickle.dump(nb_model, f)

with open(os.path.join(model_folder, "vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

print(f"Naive Bayes model and vectorizer saved in '{model_folder}'")
