import re
from nltk.tokenize import word_tokenize
from model import vectorizer, clf
from webscrapping import get_comments
df = get_comments()

def preprocess_comment(comment):
    # Lowercase each comment and remove non-alphabetic characters
    comment = re.sub(r'[^a-zA-Z\s]', '', comment).lower()
    return comment

# Apply preprocessing to all comments in the DataFrame
df['Preprocessed_Comment'] = df['Comment'].apply(preprocess_comment)

# Tokenization function
def tokenize_comment(comment):
    return word_tokenize(comment)

# Tokenize the preprocessed comments
df['Tokens'] = df['Preprocessed_Comment'].apply(tokenize_comment)

# Vectorization function (using the same vectorizer as used during training)
# Assuming you have 'vectorizer' available, if not, you need to load it from a file as well
X_comments_bow = vectorizer.transform([' '.join(tokens) for tokens in df['Tokens']])

# Predict sentiment for each comment
df['Sentiment'] = clf.predict(X_comments_bow)

# Count the occurrences of each sentiment label
sentiment_counts = df['Sentiment'].value_counts()

# Get counts for each sentiment label, defaulting to 0 if not found
positive_count = sentiment_counts.get('positive', 0)
negative_count = sentiment_counts.get('negative', 0)
uncertain_count = sentiment_counts.get('uncertainy', 0)
litigious_count = sentiment_counts.get('litigious', 0)
if(positive_count == negative_count):
    neutral_count = positive_count
else:
    neutral_count = 0

# Check the counts to determine the overall sentiment
if positive_count > negative_count and positive_count > uncertain_count and positive_count > litigious_count:
    print("Positive")
elif negative_count > positive_count and negative_count > uncertain_count and negative_count > litigious_count:
    print("Negative")
elif uncertain_count > positive_count and uncertain_count > negative_count and uncertain_count > litigious_count:
    print("Uncertain")
elif litigious_count > positive_count and litigious_count > negative_count and litigious_count > uncertain_count:
    print("Litigious")
elif positive_count == negative_count:
    print("Neutral")
else:
    print("Uncertain")

print(positive_count, negative_count, uncertain_count, litigious_count, neutral_count)