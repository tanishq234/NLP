from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import re
from io import BytesIO
import os
import traceback

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import pandas as pd
import base64
import nltk
import joblib
from xgboost import XGBClassifier

nltk.download('stopwords')

STOPWORDS = set(stopwords.words("english"))

app = Flask(__name__)
CORS(app, supports_credentials=True)

try:
    predictor = XGBClassifier()
    predictor.load_model("Models/model_xgb.json")  
    scaler = joblib.load("Models/scaler.pkl")
    cv = joblib.load("Models/countVectorizer.pkl")
except Exception as e:
    print("\n❌ Error loading model files:", e)
    predictor = scaler = cv = None

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("landing.html")

@app.route("/test", methods=["GET"])
def test():
    return "✅ Test request received. Service is running."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not all([predictor, scaler, cv]):
            raise RuntimeError("Model files not loaded.")

        if "file" in request.files:
            file = request.files["file"]
            data = pd.read_csv(file)

            if "Sentence" not in data.columns:
                return jsonify({"error": "CSV must contain a 'Sentence' column."}), 400

            predictions, graph = bulk_prediction(data)

            response = send_file(
                predictions,
                mimetype="text/csv",
                as_attachment=True,
                download_name="Predictions.csv"
            )

            response.headers["X-Graph-Exists"] = "true"
            response.headers["X-Graph-Data"] = base64.b64encode(graph.getbuffer()).decode("utf-8")
            return response

        elif request.is_json and "text" in request.json:
            text_input = request.json["text"]
            sentiment = single_prediction(text_input)
            return jsonify({"prediction": sentiment})

        else:
            return jsonify({"error": "No valid input received."}), 400

    except Exception as e:
        print("\n❌ Exception in /predict:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

def preprocess(text):
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", str(text)).lower().split()
    return " ".join([stemmer.stem(word) for word in review if word not in STOPWORDS])

def single_prediction(text_input):
    processed = preprocess(text_input)
    X = cv.transform([processed]).toarray()
    X_scaled = scaler.transform(X)
    pred = predictor.predict_proba(X_scaled).argmax(axis=1)[0]
    return "Positive" if pred == 1 else "Negative"

def bulk_prediction(data):
    data["processed"] = data["Sentence"].apply(preprocess)
    X = cv.transform(data["processed"]).toarray()
    X_scaled = scaler.transform(X)
    preds = predictor.predict_proba(X_scaled).argmax(axis=1)
    data["Predicted sentiment"] = ["Positive" if p == 1 else "Negative" for p in preds]

    csv_output = BytesIO()
    data.to_csv(csv_output, index=False)
    csv_output.seek(0)

    return csv_output, generate_pie_chart(data)

def generate_pie_chart(data):
    fig = plt.figure(figsize=(5, 5))
    colors = ("green", "red")
    wp = {"linewidth": 1, "edgecolor": "black"}
    tags = data["Predicted sentiment"].value_counts()
    explode = (0.01,) * len(tags)

    tags.plot(
        kind="pie",
        autopct="%1.1f%%",
        shadow=True,
        colors=colors,
        startangle=90,
        wedgeprops=wp,
        explode=explode,
        title="Sentiment Distribution",
        xlabel="",
        ylabel="",
    )

    graph = BytesIO()
    plt.savefig(graph, format="png", bbox_inches="tight")
    plt.close()
    graph.seek(0)
    return graph

if __name__ == "__main__":
    app.run(port=8000, debug=True)
