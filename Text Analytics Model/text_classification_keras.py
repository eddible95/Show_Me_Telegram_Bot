from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import utils
import pandas as pd
import numpy as np
import joblib

# tokenizer to remove unwanted elements from out data like symbols and numbers
from sklearn.utils import compute_class_weight

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

# Neural Network
encoder = preprocessing.LabelEncoder()
encoder.fit(y_train)
print(encoder.classes_)
y_train = encoder.transform(y_train)
y_test = encoder.transform(y_test)

# Resolves the imbalance in dataset
class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)

class_weights_dict = dict(zip(encoder.transform(list(encoder.classes_)), class_weights))
print(class_weights_dict)

num_classes = np.max(y_train) + 1
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)
print(y_train)
print(y_train.shape)

batch_size = 64
epochs = 3

# Build the model
model = Sequential()
model.add(Dense(256, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1,
                    class_weight=class_weights_dict)

score = model.evaluate(X_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test accuracy:', score[1])

# Prints the Classification Report
y_pred = model.predict(X_test)
print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1), target_names=encoder.classes_))

# Export the Model
model.save('Movie_Metadata_Sentiments_Weighted_Keras.h5', history)
