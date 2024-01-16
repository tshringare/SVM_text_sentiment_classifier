import joblib
import re
from nltk import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer as tVector
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = pd.read_csv('cyberbullying_tweets.csv')                      # loading the data and the required pre-processing
x = data['tweet_text']                                              # functions
ps = PorterStemmer()
lem = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
vector = tVector()


def process(u):                                     # data preprocessing, and fitting the data into a tfidf vector
    processed_u = u.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$', 'emailaddress')
    processed_u = processed_u.str.replace(r'^RT @\w+:', 'tag')
    processed_u = processed_u.str.replace(r'#(\w+)', r'\1')
    processed_u = processed_u.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$', 'webaddress')
    processed_u = processed_u.str.replace(r'£|\$', 'moneysymb')
    processed_u = processed_u.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$', 'phonenumbr')
    processed_u = processed_u.str.replace(r'\d+(\.\d+)?', 'numbr')
    processed_u = processed_u.str.replace(r'[^\w\d\s]', ' ')
    processed_u = processed_u.str.replace(r'\s+', ' ')
    processed_u = processed_u.str.replace(r'^\s+|\s+?$', '')
    processed_u = processed_u.str.lower()
    processed_u = processed_u.apply(lambda l: ' '.join(term for term in l.split() if term not in stop_words))
    processed_u = processed_u.apply(lambda l: ' '.join(ps.stem(term) for term in l.split()))
    processed_u = processed_u.apply(lambda l: ' '.join(lem.lemmatize(term) for term in l.split()))
    global vector
    vector = tVector(max_features=220)
    processed_u = vector.fit_transform(processed_u)
    return processed_u


def pre_classification(u, v):                                        # fitting the tweet to be classified into the
    u = re.sub(r"^.+@[^.].*\.[a-z]{2,}$", 'emailaddress', u)         # tfdif vector for the dataset
    u = re.sub(r'^RT @\w+:', 'tag', u)
    u = re.sub(r'#(\w+)', r'\1', u)
    u = re.sub(r"^http://[a-zA-Z0-9\-.]+\.[a-zA-Z]{2,3}(/\S*)?$", 'webaddress', u)
    u = re.sub(r'[£$]', 'moneysymb', u)
    u = re.sub(r'^\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}$', 'phonenumbr', u)
    u = re.sub(r'\d+(\.\d+)?', 'numbr', u)
    u = re.sub(r'[^\w\s]', ' ', u)
    u = re.sub(r'\s+', ' ', u)
    u = re.sub(r'^\s+|\s+?$', '', u)
    u = u.lower()
    u = ' '.join(term for term in u.split() if term not in stop_words)
    u = ' '.join(ps.stem(term) for term in u.split())
    u = ' '.join(lem.lemmatize(term) for term in u.split())
    j = v.transform([u])
    return j


if __name__ == "__main__":                                  # this part contains the model building and it's metrics
    X = process(x)                                          # along with an example
    y = data['cyberbullying_type'].tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy is:", accuracy_score(y_test, y_pred)*100)
    tweet = r"I hate you"                        # change this to try out different tweets
    print(tweet, " Is classified as-", model.predict(pre_classification(tweet, vector))[0])

    with open('model.joblib', 'wb') as f:                   # saving the model so that it can be executed without
        m_v = [model, vector]                               # rebuilding in the cyberbullying_classifier file
        joblib.dump(m_v, f)
