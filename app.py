import pickle
from flask import Flask, render_template, request, jsonify
from models.logistic import stemming

app = Flask(__name__)


@app.route("/")
def hello_world():
    return render_template('index.html'), 201


@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    score = 'not determined'
    try:
        print(request.get_json())
        mdl = request.get_json()['model']
        text = request.get_json()['text']
        print(f"model : {mdl} , Text : {text}")

        if mdl == "logistic":
            # loading the pre-trained models
            vectorized = pickle.load(
                open(r"./models/logistic_vectorizer.pkl", "rb"))
            model = pickle.load(
                open(r"./models/Logistic_Regression.pkl", "rb"))
            # pre-processing
            text = stemming(text)
            # prediction
            x = vectorized.transform([text])
            pred = model.predict(x)
            print(f"Prediction : {pred}")

            # sending results.
            if pred == 1:
                score = 'positive'
            elif pred == 0:
                score = 'neutral'
            else:
                score = 'negative'
            return jsonify({'score': score})
        else:
            score = 'model cannot be determined'

        return jsonify({'score': score})

    except Exception as e:
        print(e)
        return jsonify({'score': 'server error.'})


if __name__ == '__main__':
    app.run(debug=True)
