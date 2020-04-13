import pandas as pd
import ast
from tqdm import tqdm

# Read in data
meta_df = pd.read_csv('movies_metadata.csv')
keywords_df = pd.read_csv('keywords.csv')
# Merge both data sets and drop duplicate
df = pd.merge(meta_df, keywords_df, on='id', how='inner')
df.drop_duplicates(subset="id",
                   keep=False, inplace=True)

df['new_genres'] = ""
df['Text_Content'] = ""

# Convert from JSON format to strings
for index in tqdm(range(len(df))):
    # Extract each genre type from the JSON
    genres = ast.literal_eval(df['genres'].iloc[index])
    genre_list = [genre['name'] for genre in genres]

    # Extract keywords associated to each movie from JSON
    keywords = ast.literal_eval(df['keywords'].iloc[index])
    keyword_list = [word['name'] for word in keywords]

    # Combine Movie's overview, tagline and keywords into a textual field
    overview = df['overview'].iloc[index]
    tagline = df['tagline'].iloc[index]
    text = str(overview) + str(tagline)
    for item in keyword_list:
        text = text + " " + item
    # Saving into dataframe
    df['new_genres'].iloc[index] = genre_list
    df['Text_Content'].iloc[index] = text

df.to_csv('Modified_Movies_Metadata.csv')
