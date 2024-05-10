import pickle
import re
from flask import Flask, render_template, request, jsonify
from models.logistic import stemming
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import matplotlib

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


def map_intensity_to_emotion(score):
    if score >= 0.5:
        return 'joy'
    elif score >= 0.05 and score < 0.5:
        return 'pleasant'
    elif score >= -0.05 and score < 0.05:
        return 'neutral'
    elif score > -0.5 and score < -0.05:
        return 'disappointed'
    else:
        return 'sad'


def generate_reply(emotion):
    if emotion == 'joy':
        return "That's great to hear!"
    elif emotion == 'pleasant':
        return "I'm glad things are going well for you."
    elif emotion == 'neutral':
        return "Hmm, interesting."
    elif emotion == 'disappointed':
        return "I'm sorry to hear that."
    else:
        return "Oh, that's not good."


def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    vader_scores = analyzer.polarity_scores(text)
    emotion = map_intensity_to_emotion(vader_scores['compound'])
    reply = generate_reply(emotion)
    return vader_scores, emotion, reply


@app.route('/chatbot', methods=['GET', 'POST'])
def index():
    matplotlib.use('Agg')
    if request.method == 'POST':
        text = request.form['text']
        vader_scores, emotion, reply = analyze_sentiment(text)

        color = 'grey'
        alpha = 1.0
        if vader_scores['compound'] >= 0.1:
            color = 'green'
            alpha = 0.7
        elif vader_scores['compound'] <= -0.1:
            color = 'red'
            alpha = 0.9

        plt.figure(figsize=(2, 2))
        plt.bar(emotion, vader_scores['compound'], color=color, alpha=alpha)
        plt.xlabel('Emotion')
        plt.ylabel('Intensity Score')
        plt.title(f'Intensity of {emotion.capitalize()} in the Text')
        plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)

        # Save the plot to a file
        plot_path = 'static/plot.png'
        plt.savefig(plot_path)
        plt.close()

        return jsonify({'reply': reply, 'plot_path': plot_path})

    return render_template('chatbot.html'), 201


if __name__ == '__main__':
    app.run(debug=True)
