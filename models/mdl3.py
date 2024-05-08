# importing required packages.
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import re
import pickle


# Removing stop words and stemming the words in each data.
def stemming(content):
    stemmer = PorterStemmer()
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower().split()
    stemmed_content = [stemmer.stem(
        word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


def main():
    # loading the dataset.
    train_df = pd.read_csv('./datasets/2/training.csv')
    test_df = pd.read_csv('./datasets/2/test.csv')
    print(f" Training dataset : \n{train_df.head()}")
    print(f" Test dataset : \n{test_df.head()}")

    # Creating training and test corpus.
    x_train = train_df['text'].apply(stemming).tolist()
    x_test = test_df['text'].apply(stemming).tolist()

    # feature extraction.
    cv = CountVectorizer(max_features=2500)
    x_train = cv.fit_transform(x_train).toarray()
    x_test = cv.transform(x_test).toarray()

    y_train = train_df['label'].values
    y_test = test_df['label'].values

    # Saving the Count Vectorizer model.
    pickle.dump(cv, open('./models/mdl3_count_vectorizer.pkl', 'wb'))

    # Scaling the values to 0-1
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Saving the scaler model
    pickle.dump(scaler, open('./models/mdl3_scaler.pkl', 'wb'))

    print(f"X train: {x_train.shape}")
    print(f"y train: {y_train.shape}")
    print(f"X test: {x_test.shape}")
    print(f"y test: {y_test.shape}")

    # Random Classifier
    model_rf = RandomForestClassifier()
    model_rf.fit(x_train, y_train)
    print('\nRandom Classifier')

    # Saving the Random Forest classifier
    pickle.dump(model_rf, open('./models/random_forest.pkl', 'wb'))
    print("Training Accuracy :", model_rf.score(x_train, y_train))
    print("Testing Accuracy :", model_rf.score(x_test, y_test))

    # XGB Classifier
    model_xgb = XGBClassifier()
    model_xgb.fit(x_train, y_train)
    print('\nXGB Boost')

    # Saving the XGBoost classifier
    pickle.dump(model_xgb, open('./models/xgb.pkl', 'wb'))

    # Accuracy of the model on training and testing data
    print("Training Accuracy :", model_xgb.score(x_train, y_train))
    print("Testing Accuracy :", model_xgb.score(x_test, y_test))

    # Decision Tree.
    print('Decision Tree')
    model_dt = DecisionTreeClassifier()
    model_dt.fit(x_train, y_train)

    # Saving the Decision Tree classifier
    pickle.dump(model_xgb, open('./models/decision-tree.pkl', 'wb'))

    # Accuracy of the model on training and testing data
    print("Training Accuracy :", model_dt.score(x_train, y_train))
    print("Testing Accuracy :", model_dt.score(x_test, y_test))
