import pandas as pd
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import nltk

nltk.download('punkt')
from tqdm import tqdm


def text_emotion(df, column):
    '''
    Takes a DataFrame and a specified column of text and adds 10 columns to the
    DataFrame for each of the 10 emotions in the NRC Emotion Lexicon, with each
    column containing the value of the text in that emotions
    INPUT: DataFrame, string
    OUTPUT: the original DataFrame with ten new columns
    '''

    new_df = df.copy()

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

    with tqdm(total=len(list(new_df.iterrows()))) as pbar:
        for i, row in enumerate(new_df.itertuples()):
            pbar.update(1)
            document = word_tokenize(row[8])
            for word in document:
                word = stemmer.stem(word.lower())
                emo_score = emolex_words[emolex_words.word == word]
                if not emo_score.empty:
                    print(emo_score['anger'].values[0])
                    emo_df.at[i, 'anger'] += emo_score['anger']
                    emo_df.at[i, 'anticipation'] += emo_score['anticipation']
                    emo_df.at[i, 'disgust'] += emo_score['disgust']
                    emo_df.at[i, 'fear'] += emo_score['fear']
                    emo_df.at[i, 'joy'] += emo_score['joy']
                    emo_df.at[i, 'negative'] += emo_score['negative']
                    emo_df.at[i, 'positive'] += emo_score['positive']
                    emo_df.at[i, 'sadness'] += emo_score['sadness']
                    emo_df.at[i, 'surprise'] += emo_score['surprise']
                    emo_df.at[i, 'trust'] += emo_score['trust']
            if i % 100 == 0:
                dataframe = pd.concat([new_df, emo_df], axis=1)
                dataframe.to_csv('Sentiments.csv')

    new_df = pd.concat([new_df, emo_df], axis=1)
    return new_df


# df = pd.read_csv('wiki_movie_plots_deduped.csv')
# list_of_plots = df['Plot'].to_list()
# df_emotion = text_emotion(df, 'Plot')
# print(df_emotion.head())
# # df_emotion.to_csv('Sentiments.csv')


df = pd.read_csv('wiki_movie_plots_deduped.csv')
list_of_plots = df['Plot'].to_list()
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
new_df.to_csv('Sentiments.csv')