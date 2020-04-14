# Show_Me_Telegram_Bot
An AI-Enabled Telegram Bot implemented in Python to recommend movies based on user's emotions

# Data Set
movies_metadata.csv & keywords.csv contains the raw data<br>
Modified_Movies_Metadata.csv contains the preprocessed raw data<br>
Movie_Metadata_Sentiments.csv contains the final data used for model training after emotions are analysed<br>

# Running of Python Scripts
data_extraction.py is executed to combine both movies_metadata.csv and keywords.csv to form a new column 'Text_Content' in Modified_Movies_Metadata.csv<br>
sentiment_analysis.py is executed to analyse and determine the overall sentiments for text in the 'Text_Content' column to produce Movie_Metadata_Sentiments.csv<br>
