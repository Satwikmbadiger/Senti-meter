import pandas as pd
import re
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import pickle
from wordcloud import WordCloud

def generate_wordcloud(sentiment, df):
    text = ' '.join(df[df['Sentiment'] == sentiment]['Preprocessed_data'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(5, 3))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'{sentiment} Sentiment Word Cloud')
    plt.axis('off')
    plt.show()

def preprocess_data(data):
    return re.sub(r'[^a-zA-Z\s]', '', data)

def tokenize_data(preprocessed_data):
    return word_tokenize(preprocessed_data)

with open('Sentimeter\\vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('naive_bayes_model.pkl', 'rb') as f:
    clf = pickle.load(f)

def analyze_sentiment(csv_file_path):
    df = pd.read_csv(csv_file_path, header=None)

    df['Preprocessed_data'] = df[0].apply(preprocess_data)

    df['Tokens'] = df['Preprocessed_data'].apply(tokenize_data)

    X_bow = vectorizer.transform([' '.join(tokens) for tokens in df['Tokens']])

    df['Sentiment'] = clf.predict(X_bow)

    for sentiment in df['Sentiment'].unique():
        generate_wordcloud(sentiment, df)

    df['Sentiment'].value_counts().plot(kind='bar', color='skyblue')
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()

    plt.pie(df['Sentiment'].value_counts(), labels=['Positive','Uncertain','Negative','Litigious'], colors=['red', 'green','blue','yellow'], autopct='%1.1f%%', startangle=90)

    sentiment_counts = df['Sentiment'].value_counts()
    total_reviews = len(df)
    sentiment_percentages = {sentiment: count / total_reviews * 100 for sentiment, count in sentiment_counts.items()}

    print("Sentiment Analysis Results:\n")
    for sentiment, count in sentiment_counts.items():
        print(f"{sentiment}: {count} reviews ({sentiment_percentages[sentiment]:.2f}%)")

    print('\nTop Reviews in Each Sentiment Category:')
    for sentiment in df['Sentiment'].unique():
        top_reviews = df[df['Sentiment'] == sentiment][0].head(5)
        print(f"\nTop {sentiment} Reviews:\n")
        for review in top_reviews:
            print(review)

    print("\nSentiment Distribution Pie Chart:")
    plt.show()

analyze_sentiment('Reviews.csv')