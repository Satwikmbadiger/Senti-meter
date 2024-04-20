# importing packages.
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import re
import pickle


# loading the dataset.
df = pd.read_csv('./datasets/sentiment_analysis.csv')
print(df.head())


texts = df['text']
print(texts.head())


# importing stopwords.
nltk.download('stopwords')
print(stopwords.words('english'))

print(df.isnull().sum())

print(df['sentiment'].value_counts())

# mapping each sentiment with a number.
mapping = {'neutral': 0, 'positive': 1, 'negative': -1}
df['result'] = df['sentiment'].map(mapping)


# ploting the count of sentiments.
plt.hist(df['result'], color='lightgreen', ec='black', bins=15)


port_stem = PorterStemmer()
# Removing stopwords and stemming the words in each data.


def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(
        word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


# calling the stemming function.
df['stemmed_content'] = df['text'].apply(stemming)


x = df['stemmed_content']
y = df['result']

# INtializing vectorizer and ferforming feature extraction and saving it.
vector = TfidfVectorizer()
pickle.dump(vector, open('./models/logistic_vectorizer.pkl', 'wb'))
x = vector.fit_transform(x)

# slpliting datasets.
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=2)
print(x.shape, x_train.shape, x_test.shape)
print(y.shape, y_train.shape, y_test.shape)


print(x_train.get_shape())
print(x_test.get_shape())

# training and saving the model.
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)
pickle.dump(model, open('./models/Logistic_Regression.pkl', 'wb'))

# prediction on test data.
pred = model.predict(x_test)
accuracy = accuracy_score(y_test, pred)
print(accuracy)
