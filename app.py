import pickle
import re
from flask import Flask, render_template, request, jsonify
from models.logistic import stemming
from nltk.tokenize import word_tokenize


app = Flask(__name__)


@app.route("/")
def hello_world():
    return render_template('index.html'), 201


@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    score = 'not determined'
    try:
        mdl = request.get_json()['model']
        text = request.get_json()['text']
        print(f"model : {mdl} , Text : {text}")

        if mdl == "logistic" and text != '':
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

            # mapping results.
            if pred == 1:
                score = 'positive'
            elif pred == 0:
                score = 'neutral'
            else:
                score = 'negative'

        elif mdl == 'bayes' and text != '':
            # loading the pre-trained models
            vectorized = pickle.load(
                open(r"./models/vectorizer_navies.pkl", "rb")
            )
            model = pickle.load(
                open(r"./models/Naive_Bayes.pkl", "rb")
            )
            text = word_tokenize(
                re.sub(r'[^a-zA-Z\s]', '', text.lower())
            )

            # pre-processing
            x = vectorized.transform([' '.join(text)])

            # prediction
            pred = model.predict(x)
            print("Predicted sentiment:", pred[0])
            score = pred[0]

        elif mdl == 'xgboost' and text != '':
            # loading the pre-trained models
            vectorized = pickle.load(
                open(r"./models/mdl3_count_vectorizer.pkl", "rb")
            )
            model = pickle.load(
                open(r"./models/xgb.pkl", "rb")
            )
            scaler = pickle.load(
                open(r"./models/mdl3_scaler.pkl", "rb")
            )

            # pre-processing
            x = stemming(text)
            x = vectorized.transform([x])
            x = scaler.transform(x.toarray())

            # prediction
            y = model.predict_proba(x).argmax(axis=1)[0]

            labels = {0: 'sadness', 1: 'joy', 2: 'love',
                      3: 'anger', 4: 'fear', 5: 'surprise'}

            if y in labels:
                score = labels[y]
                print(f'Prediction : {score}')

        else:
            score = 'cannot be determined'
            raise Exception("Invalid input!")

        return jsonify({
            'model': mdl,
            'score': score
        }), 201

    except Exception as e:
        print(f"Error : {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
