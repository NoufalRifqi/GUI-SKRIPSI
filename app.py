from flask import Flask, render_template, request
from helper import Helper

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/classification", methods=["POST", "GET"])
def predict():
    prediction = "-"
    if request.method == "POST":
        inputUser = request.form['inputText']
        preprocess = Helper().preprocessing(inputUser)
        seqpad = Helper().text2seqpad(preprocess)
        prediction = Helper().model_classification(seqpad)
        return render_template("classification.html", prediction = prediction)
    else:
        return render_template("classification.html", prediction = prediction)

if(__name__) == '__main__':
    app.run(debug=True, host="localhost", port=8000)