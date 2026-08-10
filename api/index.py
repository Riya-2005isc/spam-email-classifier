from flask import Flask, request, jsonify, send_file
import os
import joblib

app = Flask(__name__)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(
    os.path.join(BASE, "model", "spam_model.pkl")
)

vectorizer = joblib.load(
    os.path.join(BASE, "model", "bow_vectorizer.pkl")
)


@app.route("/", methods=["GET"])
def home():
    return send_file(
        os.path.join(BASE, "index.html")
    )


@app.route("/api/predict", methods=["POST"])
def predict():

    data = request.get_json(silent=True) or {}

    message = data.get("message", "").strip()

    if not message:
        return jsonify({
            "error": "Please provide an email message."
        }), 400

    features = vectorizer.transform([message])

    prediction = model.predict(features)[0]

    result = (
        "SPAM"
        if str(prediction).lower() == "spam"
        else "HAM"
    )

    response = {
        "prediction": result
    }

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(features)[0]

        response["confidence"] = round(
            float(max(probabilities)) * 100,
            2
        )

    return jsonify(response)
