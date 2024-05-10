import os
import pickle
import re
from flask import Flask, render_template, request, jsonify
from models.logistic import stemming
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import googleapiclient.discovery
from wordcloud import WordCloud
import pandas as pd

matplotlib.use('Agg')


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


@app.route('/YtAnalysis', methods=['POST', 'GET'])
def analyze():
    def preprocess_comment(comment):
        comment = re.sub(r'[^a-zA-Z\s]', '', comment).lower()
        return comment

    def tokenize_comment(comment):
        return word_tokenize(comment)

    def get_comments(video_id):
        api_service_name = "youtube"
        api_version = "v3"
        DEVELOPER_KEY = "AIzaSyD1lXAVHpBQbDXaao5C-kTrBBkDbn1tvEI"

        youtube = googleapiclient.discovery.build(
            api_service_name, api_version, developerKey=DEVELOPER_KEY)

        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=10
        )

        response = request.execute()

        comments = []

        for item in response['items']:
            comment_text = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment_text)

        return pd.DataFrame(comments, columns=['Comment'])

    if request.method == 'POST':
        data = request.get_json()
        video_id = data.get('video_id', '')

        df_comments = get_comments(video_id)

        df_comments['Preprocessed_Comment'] = df_comments['Comment'].apply(
            preprocess_comment)

        df_comments['Tokens'] = df_comments['Preprocessed_Comment'].apply(
            tokenize_comment)

        vectorizer = pickle.load(
            open(r"./models/vectorizer_navies.pkl", "rb")
        )
        clf = pickle.load(
            open(r"./models/Naive_Bayes.pkl", "rb"))

        X_comments_bow = vectorizer.transform(
            [' '.join(tokens) for tokens in df_comments['Tokens']])

        df_comments['Sentiment'] = clf.predict(X_comments_bow)

        sentiment_counts = df_comments['Sentiment'].value_counts()

        sentiment_counts = {key: int(value)
                            for key, value in sentiment_counts.items()}

        positive_count = sentiment_counts.get('positive', 0)
        negative_count = sentiment_counts.get('negative', 0)
        uncertain_count = sentiment_counts.get('uncertain', 0)
        litigious_count = sentiment_counts.get('litigious', 0)

        if positive_count == negative_count:
            neutral_count = positive_count
        else:
            neutral_count = 0

        if positive_count > negative_count and positive_count > uncertain_count and positive_count > litigious_count:
            overall_sentiment = "Positive"
        elif negative_count > positive_count and negative_count > uncertain_count and negative_count > litigious_count:
            overall_sentiment = "Negative"
        elif uncertain_count > positive_count and uncertain_count > negative_count and uncertain_count > litigious_count:
            overall_sentiment = "Uncertain"
        elif litigious_count > positive_count and litigious_count > negative_count and litigious_count > uncertain_count:
            overall_sentiment = "Litigious"
        elif positive_count == negative_count:
            overall_sentiment = "Neutral"
        else:
            overall_sentiment = "Uncertain"

        comments_with_sentiment = [{'comment': comment, 'sentiment': sentiment}
                                   for comment, sentiment in zip(df_comments['Comment'], df_comments['Sentiment'])]

        return jsonify({
            'comments_with_sentiment': comments_with_sentiment,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'uncertain_count': uncertain_count,
            'litigious_count': litigious_count,
            'overall_sentiment': overall_sentiment
        })

    return render_template('YtAnalysis.html')


@app.route('/review_analysis', methods=['GET', 'POST'])
def review_analysis():
    global vectorizer, clf

    vectorizer = pickle.load(
        open(r"./models/vectorizer_navies.pkl", "rb")
    )
    clf = pickle.load(
        open(r"./models/Naive_Bayes.pkl", "rb"))

    def preprocess_data(data):
        return re.sub(r'[^a-zA-Z\s]', '', data)

    def tokenize_data(preprocessed_data):
        return word_tokenize(preprocessed_data)

    def generate_wordcloud(sentiment, df):
        text = ' '.join(df[df['Sentiment'] == sentiment]['Preprocessed_data'])
        wordcloud = WordCloud(width=800, height=400,
                              background_color='white').generate(text)
        plt.figure(figsize=(5, 3))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'{sentiment} Sentiment Word Cloud')
        plt.axis('off')
        wordcloud_filename = f'{sentiment}_wordcloud.png'
        wordcloud_path = os.path.join('static', wordcloud_filename)

        try:
            plt.savefig(wordcloud_path)
            plt.close()
            return wordcloud_path
        except Exception as e:
            print(f"Error saving word cloud image for sentiment '{
                  sentiment}': {e}")
            return None

    def analyze_sentiment(csv_file_path):
        df = pd.read_csv(csv_file_path, header=None)

        df['Preprocessed_data'] = df[0].apply(preprocess_data)

        df['Tokens'] = df['Preprocessed_data'].apply(tokenize_data)

        X_bow = vectorizer.transform(
            [' '.join(tokens) for tokens in df['Tokens']])

        df['Sentiment'] = clf.predict(X_bow)

        wordcloud_paths = [generate_wordcloud(
            sentiment, df) for sentiment in df['Sentiment'].unique()]

        sentiment_counts = df['Sentiment'].value_counts()
        total_reviews = len(df)
        sentiment_percentages = {sentiment: count / total_reviews *
                                 100 for sentiment, count in sentiment_counts.items()}
        top_reviews = {sentiment: df[df['Sentiment'] == sentiment][0].head(
            5).tolist() for sentiment in df['Sentiment'].unique()}

        plt.figure(figsize=(8, 6))
        df['Sentiment'].value_counts().plot(kind='bar', color='skyblue')
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        sentiment_bar_plot_path = os.path.join(
            "static", "sentiment_distribution.png")
        plt.savefig(sentiment_bar_plot_path)
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.pie(df['Sentiment'].value_counts(), labels=['Positive', 'Uncertain', 'Negative', 'Litigious'],
                colors=['red', 'green', 'blue', 'yellow'], autopct='%1.1f%%', startangle=90)
        sentiment_pie_chart_path = os.path.join(
            "static", "sentiment_pie_chart.png")
        plt.savefig(sentiment_pie_chart_path)
        plt.close()

        return sentiment_counts, sentiment_percentages, top_reviews, wordcloud_paths, sentiment_bar_plot_path, sentiment_pie_chart_path

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('Review.html', message='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('Review.html', message='No selected file')

        if file:
            sentiment_counts, sentiment_percentages, top_reviews, wordcloud_paths, sentiment_bar_plot_path, sentiment_pie_chart_path = analyze_sentiment(
                file)
            return render_template('Results.html', sentiment_counts=sentiment_counts, sentiment_percentages=sentiment_percentages, top_reviews=top_reviews, wordcloud_paths=wordcloud_paths, sentiment_bar_plot_path=sentiment_bar_plot_path, sentiment_pie_chart_path=sentiment_pie_chart_path)

    return render_template('Review.html')


if __name__ == '__main__':
    app.run(debug=True)
