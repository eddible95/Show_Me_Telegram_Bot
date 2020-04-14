import pandas as pd
from tqdm import tqdm
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import nltk

nltk.download('punkt')

df = pd.read_csv('Modified_Movies_Metadata.csv')
list_of_plots = df['Text_Content'].to_list()
file_path = "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
emolex_df = pd.read_csv(file_path,
                        names=["word", "emotion", "association"],
                        sep='\t')
emolex_words = emolex_df.pivot(index='word',
                               columns='emotion',
                               values='association').reset_index()
emotions = emolex_words.columns.drop('word')
emo_df = pd.DataFrame(0, index=df.index, columns=emotions)

stemmer = SnowballStemmer("english")

emotions_dict = {}
with tqdm(total=len(list_of_plots)) as pbar:
    for index, plot in enumerate(list_of_plots):
        pbar.update(1)
        document = word_tokenize(plot)
        emotions = {'anger': 0, 'anticipation': 0, 'disgust': 0, 'fear': 0, 'joy': 0,
                    'negative': 0, 'positive': 0, 'sadness': 0, 'surprise': 0, 'trust': 0}
        for word in document:
            word = stemmer.stem(word.lower())
            emo_score = emolex_words[emolex_words.word == word]
            if not emo_score.empty:
                emotions['anger'] += emo_score['anger'].values[0]
                emotions['anticipation'] += emo_score['anticipation'].values[0]
                emotions['disgust'] += emo_score['disgust'].values[0]
                emotions['fear'] += emo_score['fear'].values[0]
                emotions['joy'] += emo_score['joy'].values[0]
                emotions['negative'] += emo_score['negative'].values[0]
                emotions['positive'] += emo_score['positive'].values[0]
                emotions['sadness'] += emo_score['sadness'].values[0]
                emotions['surprise'] += emo_score['surprise'].values[0]
                emotions['trust'] += emo_score['trust'].values[0]
        emotions_dict[index] = emotions

data_frame = pd.DataFrame(emotions_dict)
data_frame = data_frame.T
new_df = pd.concat([df, data_frame], axis=1)
new_df.to_csv('Movie_Metadata_Sentiments.csv')
