import pandas as pd
import re
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#import dataset
csv_file_path = "C:\\Users\\SATWIK M BADIGER\\Desktop\\projects\\ML\\Sentimeter\\dataset.csv"
df = pd.read_csv(csv_file_path)

# Data Preprocessing
df = df[df['Language'] == 'en']
df.drop_duplicates(inplace=True)
df['Text'] = df['Text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
df['Text'] = df['Text'].apply(word_tokenize)

# Vectorization
vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform([' '.join(tokens) for tokens in df['Text']])

#test and train data splitting
X_train, X_test, y_train, y_test = train_test_split(X_bow, df['Label'], test_size=0.2, random_state=42)

#training Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

#making predictions
y_pred = clf.predict(X_test)

#model accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)