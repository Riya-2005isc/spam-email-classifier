import tarfile, os, pandas as pd, joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Put these two archives in a data/ folder before running this script:
# 20021010_easy_ham.tar.bz2
# 20021010_spam.tar.bz2

os.makedirs("data/easy_ham", exist_ok=True)
os.makedirs("data/spam", exist_ok=True)

with tarfile.open("data/20021010_easy_ham.tar.bz2", "r:bz2") as tar:
    tar.extractall("data/easy_ham")

with tarfile.open("data/20021010_spam.tar.bz2", "r:bz2") as tar:
    tar.extractall("data/spam")

emails, labels = [], []

for folder, label in [("data/easy_ham/easy_ham", "ham"),
                      ("data/spam/spam", "spam")]:
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if os.path.isfile(path):
            with open(path, "r", encoding="latin-1") as f:
                emails.append(f.read())
                labels.append(label)

df = pd.DataFrame({"Message": emails, "Label": labels})

bow = CountVectorizer()
X = bow.fit_transform(df["Message"])

model = MultinomialNB()
model.fit(X, df["Label"])

joblib.dump(model, "model/spam_model.pkl")
joblib.dump(bow, "model/bow_vectorizer.pkl")

print("Model saved successfully.")
print(df["Label"].value_counts())
