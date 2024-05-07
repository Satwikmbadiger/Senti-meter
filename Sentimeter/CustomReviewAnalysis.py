import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob

dataset = pd.read_csv('Reviews.csv', header=None)
data = pd.DataFrame(dataset)
data = data.dropna()

def sentiment_calc(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity

    if polarity > 0:
        return 'Positive', polarity
    elif polarity < 0:
        return 'Negative', polarity
    else:
        return 'Neutral', polarity

data['Sentiment'], data['Polarity'] = zip(*data[0].apply(sentiment_calc))

sentiment_counts = data['Sentiment'].value_counts()

sentiment_percentages = sentiment_counts / len(data) * 100

plt.bar(sentiment_counts.index, sentiment_counts.values, color=['green', 'red', 'white'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

positive_text = ' '.join(data[data['Sentiment'] == 'Positive'][0])
negative_text = ' '.join(data[data['Sentiment'] == 'Negative'][0])

positive_wordcloud = WordCloud(width=800, height=400).generate(positive_text)
negative_wordcloud = WordCloud(width=800, height=400).generate(negative_text)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.title('Positive Sentiment Word Cloud')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.title('Negative Sentiment Word Cloud')
plt.axis('off')

plt.show()

print('Sentiment Analysis:')
for sentiment, count in sentiment_counts.items():
    print(f"{sentiment}: {count} reviews ({sentiment_percentages[sentiment]:.2f}%)")

print('\nTop Positive Reviews:')
for review in data[data['Sentiment'] == 'Positive'].nlargest(5, 'Polarity')[0]:
    print(review)

print('\nTop Negative Reviews:')
for review in data[data['Sentiment'] == 'Negative'].nsmallest(5, 'Polarity')[0]:
    print(review)

positive_avg_polarity = data[data['Sentiment'] == 'Positive']['Polarity'].mean()
negative_avg_polarity = data[data['Sentiment'] == 'Negative']['Polarity'].mean()

print(f"\nAverage Polarity Score for Positive Reviews: {positive_avg_polarity:.2f}")
print(f"Average Polarity Score for Negative Reviews: {negative_avg_polarity:.2f}")

print('\nAnalysis Completed!')