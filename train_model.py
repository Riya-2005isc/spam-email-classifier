import tarfile
import os
import pandas as pd
import joblib

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# Create folders
os.makedirs("data/easy_ham", exist_ok=True)
os.makedirs("data/spam", exist_ok=True)
os.makedirs("model", exist_ok=True)


# Extract HAM dataset
with tarfile.open(
    "data/20021010_easy_ham.tar.bz2",
    "r:bz2"
) as tar:
    tar.extractall("data/easy_ham")


# Extract SPAM dataset
with tarfile.open(
    "data/20021010_spam.tar.bz2",
    "r:bz2"
) as tar:
    tar.extractall("data/spam")


emails = []
labels = []


# Read HAM emails
ham_path = "data/easy_ham/easy_ham"

for file in os.listdir(ham_path):

    file_path = os.path.join(ham_path, file)

    if os.path.isfile(file_path):

        with open(
            file_path,
            "r",
            encoding="latin-1"
        ) as f:

            emails.append(f.read())
            labels.append("ham")


# Read SPAM emails
spam_path = "data/spam/spam"

for file in os.listdir(spam_path):

    file_path = os.path.join(spam_path, file)

    if os.path.isfile(file_path):

        with open(
            file_path,
            "r",
            encoding="latin-1"
        ) as f:

            emails.append(f.read())
            labels.append("spam")


# Create DataFrame
df = pd.DataFrame({
    "Message": emails,
    "Label": labels
})


print("Dataset Shape:", df.shape)
print("\nMessage Count:")
print(df["Label"].value_counts())


# Bag of Words
bow = CountVectorizer()

X = bow.fit_transform(df["Message"])


# Train Naive Bayes model
model = MultinomialNB()

model.fit(X, df["Label"])


# Save trained model
joblib.dump(
    model,
    "model/spam_model.pkl"
)


# Save vectorizer
joblib.dump(
    bow,
    "model/bow_vectorizer.pkl"
)


print("\nModel trained successfully!")

print("Created:")
print("model/spam_model.pkl")
print("model/bow_vectorizer.pkl")
