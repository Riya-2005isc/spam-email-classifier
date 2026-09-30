# Spam Email Classifier

A Natural Language Processing (NLP) based spam email classification system using **Bag of Words (CountVectorizer)** and **Multinomial Naive Bayes**.

## Project Overview

This project classifies email messages into two categories:

* **Spam** — unwanted or potentially harmful email
* **Ham** — legitimate email

The model uses email text as input and predicts whether the message is spam or ham.

## NLP Approach

The project uses:

1. **CountVectorizer (Bag of Words)** for converting email text into numerical features.
2. **Multinomial Naive Bayes** for classification.

The practical achieved an accuracy of **97.22%** using the Bag of Words approach.

## Dataset

The dataset contains:

* 2,551 ham emails
* 501 spam emails
* Total: 3,052 emails

The data is divided into training and testing sets using an 80:20 split.

## Model Performance

| Model                                  | Accuracy |
| -------------------------------------- | -------: |
| Bag of Words + Multinomial Naive Bayes |   97.22% |
| TF-IDF + Multinomial Naive Bayes       |   90.18% |

The Bag of Words model was selected for deployment because it achieved the higher accuracy in the practical.

## Project Structure

```text
spam-email-classifier/
│
├── api/
│   └── index.py
│
├── public/
│   └── index.html
│
├── model/
│   ├── spam_model.pkl
│   └── bow_vectorizer.pkl
│
├── train_model.py
├── requirements.txt
├── vercel.json
└── README.md
```

## How It Works

```text
Email Message
      ↓
CountVectorizer
      ↓
Numerical Features
      ↓
Multinomial Naive Bayes
      ↓
SPAM / HAM
```

## Deployment

The application is designed to be deployed using **Vercel**.

The trained model and vectorizer are saved using Joblib and loaded by the API during prediction.

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Flask
* Joblib
* NLP
* Bag of Words
* Multinomial Naive Bayes
* Vercel
# 📧 Spam Email Classifier

An NLP-based machine learning application that classifies emails as **Spam** or **Ham**.

## 🚀 Live Demo

👉 **[Try the Spam Email Classifier] (https://spam-email-classifier-nvy4.vercel.app/)**

## 🛠️ Technologies Used

- Python
- NLP
- Machine Learning
- TF-IDF
- Streamlit
- Vercel
## Author

Riya Rathod
