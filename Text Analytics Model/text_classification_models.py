from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import RegexpTokenizer
from six.moves import cPickle as Pickle
from sklearn import preprocessing
from sklearn import metrics
import pandas as pd
import joblib

df = pd.read_csv('Movie_Metadata_Sentiments.csv')
# Subset only emotions required to get overall emotion detected from the text content
sub_df = df[['anger', 'joy', 'fear', 'sadness']]
# Label the movie with the highest count of emotions
df['Max'] = sub_df.idxmax(axis=1)
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 1), tokenizer=token.tokenize)
cv = cv.fit(df['Text_Content'])
text_counts = cv.transform(df['Text_Content'])
# Save the vectorizer
joblib.dump(cv, "vectorizer.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    text_counts, df['Max'], test_size=0.2, random_state=1)

print(X_train.shape)
le = preprocessing.LabelEncoder()
le.fit(y_train)
print(le.classes_)
y_train = le.transform(y_train)
y_test = le.transform(y_test)

# Model Generation Using Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train)
predicted = clf.predict(X_test)
print(predicted[:10])
print(y_test[:10])
print("MultinomialNB Accuracy:", metrics.accuracy_score(y_test, predicted))
print(classification_report(y_test, predicted, target_names=le.classes_))

# Logistics Regression
clf = LogisticRegression(C=1e3).fit(X_train, y_train)
predicted = clf.predict(X_test)
print(predicted[:10])
print(y_test[:10])
print("Logistics Regression Accuracy:", metrics.accuracy_score(y_test, predicted))
print(classification_report(y_test, predicted, target_names=le.classes_))

# Model Generation using Linear Support Vector Machine
clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=1000, tol=None).fit(X_train,
                                                                                                          y_train)
predicted = clf.predict(X_test)
print(predicted[:10])
print(y_test[:10])
print("Linear Support Vector Machine Accuracy:", metrics.accuracy_score(y_test, predicted))
print(classification_report(y_test, predicted, target_names=le.classes_))

# Save the model to be imported into Telegram bot
with open('Movie_Metadata_Sentiments_LSV_model.model', 'wb') as file_handle:
    Pickle.dump(clf, file_handle)
