from flask import Flask, render_template, request, jsonify
from joblib import load

app = Flask(__name__)

model = load("spam_model.pkl")
vectorizer = load("tfidf_vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    confidence = ""
    message = ""

    if request.method == "POST":
        message = request.form["message"]

        vector = vectorizer.transform([message])
        result = model.predict(vector)
        probs = model.predict_proba(vector)[0]

        if result[0] == 1:
            prediction = "SPAM"
            confidence = f"{probs[1] * 100:.1f}%"
        else:
            prediction = "NOT SPAM"
            confidence = f"{probs[0] * 100:.1f}%"

    return render_template("index.html", prediction=prediction, confidence=confidence, message=message)

@app.route("/api/classify", methods=["POST", "OPTIONS"])
def classify_api():
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "POST"
        return response

    data = request.get_json() or {}
    message = data.get("message", "")
    if not message:
        return jsonify({"error": "Message is required"}), 400

    vector = vectorizer.transform([message])
    result = model.predict(vector)
    probs = model.predict_proba(vector)[0]

    if result[0] == 1:
        prediction = "SPAM"
        confidence = f"{probs[1] * 100:.1f}%"
    else:
        prediction = "NOT SPAM"
        confidence = f"{probs[0] * 100:.1f}%"

    resp = jsonify({
        "prediction": prediction,
        "confidence": confidence,
        "message": message
    })
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp

if __name__ == "__main__":
    app.run(debug=True)