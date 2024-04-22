import pandas as pd
import re
import pickle
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

try:
    # ? Relative path starts the place where file is executed not where the file is present.
    # ! do not run this python file from this directory
    # ! run the file from the root directory of the project.
    df = pd.concat(
        map(
            pd.read_csv,
            [
                './datasets/1/xaa.csv',
                './datasets/1/xab.csv',
                './datasets/1/xac.csv',
                './datasets/1/xad.csv',
                './datasets/1/xae.csv',
                './datasets/1/xad.csv',
                './datasets/1/xag.csv',
                './datasets/1/xah.csv'
            ],
        ),
        ignore_index=True
    )
except Exception as error:
    print(error)
    exit()

print(df.head())

# checking for null values
null_items = df[df.isnull().any(axis=1)]
null_indices = null_items.index.tolist()
print(f" null indices : {null_indices}")

# removing rows having null values
df = df.drop(null_indices)

# total rows
print("Number of rows:", len(df))

# only en language
df = df[df['Language'] == 'en']

# remove duplicates
df.drop_duplicates(inplace=True)

# number of rows after filtering
print("Number of rows after filtering:", len(df))

# print the preprocessed df
print("\nPreprocessed data:")
print(df.head())

# keep only alphabets
df['Text'] = df['Text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
df['Text'] = df['Text'].apply(word_tokenize)

print("\nTokenized data:")
print(df.head())

# Vectorization
vectorizer = CountVectorizer()
pickle.dump(vectorizer, open('./models/vectorizer_navies.pkl', 'wb'))
X_bow = vectorizer.fit_transform([' '.join(tokens) for tokens in df['Text']])

# test and train data splitting
X_train, X_test, y_train, y_test = train_test_split(
    X_bow, df['Label'], test_size=0.2, random_state=42)

# training Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# saving model
pickle.dump(clf, open('./models/Naive_Bayes.pkl', 'wb'))

# making predictions
y_pred = clf.predict(X_test)

# model accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

user_input = "I am happy"

# preprocess the user input
user_input_tokens = word_tokenize(
    re.sub(r'[^a-zA-Z\s]', '', user_input.lower()))

# Vectorization
user_input_bow = vectorizer.transform([' '.join(user_input_tokens)])

# making prediction
user_input_pred = clf.predict(user_input_bow)

# print result
print("Predicted sentiment:", user_input_pred[0])
