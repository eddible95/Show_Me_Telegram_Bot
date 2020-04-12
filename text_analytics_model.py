from nltk.stem import WordNetLemmatizer, SnowballStemmer
from sklearn.model_selection import train_test_split
from six.moves import cPickle as Pickle
from gensim import matutils
from gensim import corpora
from tqdm import tqdm
import pandas as pd
import numpy as np
import guidedlda
import gensim


# Lemmatize each token before stemming to get root word
def lemmatize_stemming(text):
    stemmer = SnowballStemmer("english")
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


# Tokenize and lemmatize
def preprocess(text):
    # simple_process reads a file line-by-line process one line of the file at a time
    result = [lemmatize_stemming(token) for token in gensim.utils.simple_preprocess(text) if
              (token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3)]
    return result


# Convert document into the bag-of-words format
def bow_iterator(docs, dictionary):
    for doc in docs:
        yield dictionary.doc2bow(doc)


# Converts bag-of-words into a sparse matrix
def get_term_matrix(msgs, dictionary):
    bow = bow_iterator(msgs, dictionary)
    X = np.transpose(matutils.corpus2csc(bow).astype(np.int64))
    return X


# Extract only textual content from the Text_Content column
def extract_data(df):
    processed_docs = []
    for x in tqdm(range(len(df))):
        processed_docs.append(preprocess(df['Text_Content'].iloc[x]))
    return processed_docs


def test_model():
    # Read ind data set
    df = pd.read_csv('Movie_Metadata_Sentiments_Test.csv')
    # Subset only emotions required to get overall emotion detected from the text content
    emotion_list = ['anger', 'joy', 'fear', 'sadness']
    # Pre-process data to be fed into Guide LDA Model
    test_docs = extract_data(df)

    # Read in trained GuidedLDA model
    with open('Movie_Metadata_Sentiments.model', 'rb') as file_handle:
        model = Pickle.load(file_handle)

    # Read in dictionary saved
    dict_name = 'Movie_Metadata_Sentiments'
    dictionary = gensim.corpora.Dictionary.load("./dictionaries/" + dict_name + ".dict")
    # Get term matrix
    term_matrix = get_term_matrix(test_docs, dictionary)
    # Get predictions
    probabilities = model.transform(term_matrix)

    count = 0
    correct = 0
    for p in probabilities:
        pred_emotion = emotion_list[np.argmax(p)]
        print('Predicted emotion: ', pred_emotion, ' Actual emotion: ', df['Max'].iloc[count])
        actual_emotion = df['Max'].iloc[count]
        count += 1

        if pred_emotion == actual_emotion:
            print('correct')
            correct += 1

    print("Accuracy: ", correct / count)


def run():
    # Read in data set
    df = pd.read_csv('Movie_Metadata_Sentiments.csv')
    # Subset only emotions required to get overall emotion detected from the text content
    sub_df = df[['anger', 'joy', 'fear', 'sadness']]
    df['Max'] = sub_df.idxmax(axis=1)
    # Split into train and test data set
    train, test = train_test_split(df, test_size=0.2, random_state=1)
    # Save to csv file
    df.to_csv('Movie_Metadata_Sentiments_Modified.csv', encoding='utf-8', header=True)
    test.to_csv('Movie_Metadata_Sentiments_Test.csv', encoding='utf-8', header=True)
    # Pre-process data to be fed into Guide LDA Model
    processed_docs = extract_data(train)

    dictionary = gensim.corpora.Dictionary(processed_docs)
    dictionary.filter_extremes(no_below=5, no_above=0.4, keep_n=100000)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    dict1 = dictionary.token2id
    X = get_term_matrix(processed_docs, dictionary)

    print("Guided LDA")
    emolex_df = pd.read_csv('NRC-Emotion-Lexicon-Wordlevel-v0.92.txt', names=["word", "emotion", "association"],
                            sep='\t')

    # Create seed list for each category
    anger_df = emolex_df[(emolex_df.association == 1) & (emolex_df.emotion == 'anger')].word
    anger_seed = [item for item in anger_df]

    joy_df = emolex_df[(emolex_df.association == 1) & (emolex_df.emotion == 'joy')].word
    joy_seed = [item for item in joy_df]

    fear_df = emolex_df[(emolex_df.association == 1) & (emolex_df.emotion == 'fear')].word
    fear_seed = [item for item in fear_df]

    sadness_df = emolex_df[(emolex_df.association == 1) & (emolex_df.emotion == 'sadness')].word
    sadness_seed = [item for item in sadness_df]

    # Append all topic list to be fed into model
    seed_topic_list = [anger_seed, joy_seed, fear_seed, sadness_seed]

    seed_topics = {}
    for t_id, st in enumerate(seed_topic_list):
        for word in st:
            try:
                seed_topics[dict1[word]] = t_id
            except:
                pass

    # Train the GuidedLDA model
    model = guidedlda.GuidedLDA(alpha=0.1, n_topics=4, n_iter=1000, random_state=7, refresh=20)
    model.fit(X, seed_topics=seed_topics, seed_confidence=0.20)

    # Check top n words in each topic (emotions)
    n_top_words = 15
    topic_word = model.topic_word_
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(list(dict1.keys()))[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))

    # Save the model to be imported into Telegram bot
    with open('Movie_Metadata_Sentiments.model', 'wb') as file_handle:
        Pickle.dump(model, file_handle)

    # Save the Dictionary and Corpus
    name = "Movie_Metadata_Sentiments"
    dictionary.save("./dictionaries/" + name + ".dict")
    corpora.MmCorpus.serialize("./corpus/" + name + ".mm", bow_corpus)
    # Test the model on test set
    test_model()


if __name__ == '__main__':
    run()
