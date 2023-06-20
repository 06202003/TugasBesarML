from flask import Flask,render_template,url_for,request
from scipy.sparse import csr_matrix
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.model_selection import train_test_split
import re
from html import unescape
import os
import tarfile
import urllib.request
import email
import nltk
import urlextract
import email.policy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier


app = Flask(__name__,template_folder='templates')

DOWNLOAD_ROOT = "http://spamassassin.apache.org/old/publiccorpus/"
HAM_URL = DOWNLOAD_ROOT + "20030228_easy_ham.tar.bz2"
SPAM_URL = DOWNLOAD_ROOT + "20030228_spam.tar.bz2"
SPAM_PATH = os.path.join("datasets", "spam")

class EmailToWordCounterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, strip_headers=True, lower_case=True,
                 remove_punctuation=True, replace_urls=True,
                 replace_numbers=True, stemming=True):
        self.strip_headers = strip_headers
        self.lower_case = lower_case
        self.remove_punctuation = remove_punctuation
        self.replace_urls = replace_urls
        self.replace_numbers = replace_numbers
        self.stemming = stemming
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_transformed = []
        for email in X:
            text = email_to_text(email) or ""
            if self.lower_case:
                text = text.lower()
            if self.replace_urls and url_extractor is not None:
                urls = list(set(url_extractor.find_urls(text)))
                urls.sort(key=lambda url: len(url), reverse=True)
                for url in urls:
                    text = text.replace(url, " URL ")
            if self.replace_numbers:
                text = re.sub(r'\d+(?:\.\d*)?(?:[eE][+-]?\d+)?', 'NUMBER', text)
            if self.remove_punctuation:
                text = re.sub(r'\W+', ' ', text, flags=re.M)
            word_counts = Counter(text.split())
            if self.stemming and stemmer is not None:
                stemmed_word_counts = Counter()
                for word, count in word_counts.items():
                    stemmed_word = stemmer.stem(word)
                    stemmed_word_counts[stemmed_word] += count
                word_counts = stemmed_word_counts
            X_transformed.append(word_counts)
        return np.array(X_transformed)
    
class WordCounterToVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size
    def fit(self, X, y=None):
        total_count = Counter()
        for word_count in X:
            for word, count in word_count.items():
                total_count[word] += min(count, 10)
        most_common = total_count.most_common()[:self.vocabulary_size]
        self.vocabulary_ = {word: index + 1
                            for index, (word, count) in enumerate(most_common)}
        return self
    def transform(self, X, y=None):
        rows = []
        cols = []
        data = []
        for row, word_count in enumerate(X):
            for word, count in word_count.items():
                rows.append(row)
                cols.append(self.vocabulary_.get(word, 0))
                data.append(count)
        return csr_matrix((data, (rows, cols)),
                          shape=(len(X), self.vocabulary_size + 1))

def fetch_spam_data(ham_url=HAM_URL, spam_url=SPAM_URL, spam_path=SPAM_PATH):
    if not os.path.isdir(spam_path):
        os.makedirs(spam_path)
    for filename, url in (("ham.tar.bz2", ham_url), ("spam.tar.bz2", spam_url)):
        path = os.path.join(spam_path, filename)
        if not os.path.isfile(path):
            urllib.request.urlretrieve(url, path)
        tar_bz2_file = tarfile.open(path)
        tar_bz2_file.extractall(path=spam_path)
        tar_bz2_file.close()

def load_email(filepath):
    with open(filepath, "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)
    
def get_email_structure(email):
    if isinstance(email, str):
        return email
    payload = email.get_payload()
    if isinstance(payload, list):
        multipart = ", ".join([get_email_structure(sub_email)
                               for sub_email in payload])
        return f"multipart({multipart})"
    else:
        return email.get_content_type()

def structures_counter(emails):
    structures = Counter()
    for email in emails:
        structure = get_email_structure(email)
        structures[structure] += 1
    return structures

def html_to_plain_text(html):
    text = re.sub('<head.*?>.*?</head>', '', html, flags=re.M | re.S | re.I)
    text = re.sub('<a\s.*?>', ' HYPERLINK ', text, flags=re.M | re.S | re.I)
    text = re.sub('<.*?>', '', text, flags=re.M | re.S)
    text = re.sub(r'(\s*\n)+', '\n', text, flags=re.M | re.S)
    return unescape(text)

def email_to_text(email):
    html = None
    for part in email.walk():
        ctype = part.get_content_type()
        if not ctype in ("text/plain", "text/html"):
            continue
        try:
            content = part.get_content()
        except: # in case of encoding issues
            content = str(part.get_payload())
        if ctype == "text/plain":
            return content
        else:
            html = content
    if html:
        return html_to_plain_text(html)


ham_dir, spam_dir = fetch_spam_data()
ham_filenames = [f for f in sorted(ham_dir.iterdir()) if len(f.name) > 20]
spam_filenames = [f for f in sorted(spam_dir.iterdir()) if len(f.name) > 20]

print(len(ham_filenames))
print(len(spam_filenames))

ham_emails = [load_email(filepath) for filepath in ham_filenames]
spam_emails = [load_email(filepath) for filepath in spam_filenames]

print(ham_emails[1].get_content().strip())
print(spam_emails[6].get_content().strip())

print(structures_counter(ham_emails).most_common())
print(structures_counter(spam_emails).most_common())
for header, value in spam_emails[0].items():
    print(header, ":", value)
spam_emails[0]["Subject"]

X = np.array(ham_emails + spam_emails, dtype=object)
y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

html_spam_emails = [email for email in X_train[y_train==1] if get_email_structure(email) == "text/html"]
sample_html_spam = html_spam_emails[7]
print(sample_html_spam.get_content().strip()[:1000], "...")

print(html_to_plain_text(sample_html_spam.get_content())[:1000], "...")

print(email_to_text(sample_html_spam)[:100], "...")

stemmer = nltk.PorterStemmer()
for word in ("Computations", "Computation", "Computing", "Computed", "Compute",
             "Compulsive"):
    print(word, "=>", stemmer.stem(word))

url_extractor = urlextract.URLExtract()
some_text = "Will it detect github.com and https://youtu.be/7Pq-S557XQU?t=3m32s"
url_extractor.find_urls(some_text)

X_few = X_train[:3]
X_few_wordcounts = EmailToWordCounterTransformer().fit_transform(X_few)
print(X_few_wordcounts)

vocab_transformer = WordCounterToVectorTransformer(vocabulary_size=10)
X_few_vectors = vocab_transformer.fit_transform(X_few_wordcounts)
print(X_few_vectors)

X_few_vectors.toarray()
vocab_transformer.vocabulary_

preprocess_pipeline = Pipeline([
    ("email_to_wordcount", EmailToWordCounterTransformer()),
    ("wordcount_to_vector", WordCounterToVectorTransformer()),
])

X_train_transformed = preprocess_pipeline.fit_transform(X_train)


random_forest_clf = RandomForestClassifier(random_state=42)
lg_reg            = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
svm_clf           = SVC(gamma='auto')
ada_clf           = AdaBoostClassifier(n_estimators=100, random_state=42)

estimators = [
    random_forest_clf, 
    lg_reg, 
    svm_clf, 
    ada_clf
]

for estimator in estimators:
    print("Training the", estimator)
    estimator.fit(X_train_transformed, y_train)

named_estimators = [
    ("random_forest_clf", random_forest_clf),
    ("logistic_regression", lg_reg),
    ("svm_clf", svm_clf),
    ("adaboost_clf", ada_clf),
]

voting_clf = VotingClassifier(named_estimators)
voting_clf.fit(X_train_transformed, y_train)


emaila = "This is a sample email. Please ignore."
parsed_email = email.parser.BytesParser(policy=email.policy.default).parsebytes(emaila.encode('utf-8'))
text = email_to_text(parsed_email)
print(text)
preprocessed_email = preprocess_pipeline.transform([text])

# Make the prediction
prediction = voting_clf.predict(preprocessed_email)

# Print the prediction
if prediction[0] == 0:
    print("The email is classified as ham (non-spam).")
else:
    print("The email is classified as spam.")

@app.route('/')
def index():
	 return render_template('index.html')

@app.route('/Predict',methods=['POST'])
# def Predict():
#     if request.method == 'POST':
#         message = request.form['message']
#         data = [message]
#         my_prediction = voting_clf.predict(data)
#     return render_template('result.html',prediction = my_prediction)

def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        email_message = email.message.EmailMessage()
        email_message.set_content(message)
        preprocessed_data = preprocess_pipeline.transform(data)  # Preprocess the input data
        my_prediction = voting_clf.predict(preprocessed_data)
        return render_template('result.html', prediction=my_prediction[0])

app.run(debug=True)