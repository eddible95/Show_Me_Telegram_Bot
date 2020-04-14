from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, ConversationHandler
from chatterbot.trainers import ChatterBotCorpusTrainer
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove
from keras.backend import set_session
from keras.models import load_model
from chatterbot import ChatBot
import tensorflow as tf
import numpy as np
import pandas as pd
import random
import logging
import joblib
import ast


# Load trained Keras Model
sess = tf.Session()
graph = tf.get_default_graph()
set_session(sess)
text_model = load_model('./Text Analytics Model/Movie_Metadata_Sentiments_Weighted_Keras.h5')

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

GENRE, YEAR, RATINGS, CONVERSATION, PREDICT = range(5)


class MyBot(object):
    """Stores all variables and functions for the Telegram bot."""

    def __init__(self, chatter_bot, text_model, data_frame, vectorizer):
        self.chatterBot = chatter_bot
        self.textModel = text_model
        self.vectorizer = vectorizer
        self.database = data_frame
        self.genre = ''
        self.year = ''
        self.rating = ''
        self.count = 0
        self.content = ''

    def restart(self):
        self.genre = ''
        self.year = ''
        self.rating = ''
        self.count = 0
        self.content = ''

    def start(self, update, context):
        """Command for user to initiate conversation with the bot."""
        reply_keyboard = [['Drama', 'Comedy'], ['Action', 'Romance'], ['Thriller', 'Family']]
        update.message.reply_text("Hey there friend! Please select a genre of movie that you would be interested in.\n"
                                  "Send /cancel to stop me anytime ❌",
                                  reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True))
        return GENRE

    def get_year(self, update, context):
        """Receives the genre of movies and prompt user for released year of movies."""
        user = update.message.from_user
        self.genre = update.message.text
        update.message.reply_text('I see you are interested in movies of {} genres'.format(update.message.text))
        logger.info("Genre Selected By %s: %s", user.first_name, update.message.text)
        reply_keyboard = [['>2015', '2010-2015'], ['2005-2010', '<2005']]
        update.message.reply_text("Alrighty! What about the year range for the movie released date? \n"
                                  "Send /cancel to stop me anytime ❌",
                                  reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True))
        return YEAR

    def get_rating(self, update, context):
        """Receives the released year of movies and prompt user for ratings."""
        user = update.message.from_user
        self.year = update.message.text
        update.message.reply_text('No issue! I\'ll only show movies from this period: {}'.format(update.message.text))
        logger.info("Year Range Selected By %s: %s", user.first_name, update.message.text)
        reply_keyboard = [['Average', 'Excellent']]
        update.message.reply_text("Alrighty! What about the ratings of the movies to be shown?\n"
                                  "Send /cancel to stop me anytime ❌"
                                  , reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True))
        return RATINGS

    def get_mood(self, update, context):
        """Receives the ratings of movies and prompt user for their current mood."""
        user = update.message.from_user
        self.rating = update.message.text
        update.message.reply_text(
            'You sure have a good taste. Only movies of {} ratings will be recommended!'.format(update.message.text))
        logger.info("Ratings Selected By %s: %s", user.first_name, update.message.text)
        update.message.reply_text(
            "Let's have a chat to know you better! Please start by describing your current mood and "
            "try your best to continue the conversation with me 😀\nSend /cancel to stop me anytime ❌\n"
            "Send /describe if you prefer to just describe your current mood to be recommended "
            "a movie\n",
            reply_markup=ReplyKeyboardRemove())
        return CONVERSATION

    def get_prediction(self, user_input):
        """Classify the user's conversation into one of the 4 emotions using trained text classifier."""
        global sess
        global graph
        with graph.as_default():
            set_session(sess)
            emotion_list = ['Anger', 'Fear', 'Joy', 'Sadness']
            # Convert user input into format accepted by the text analytics model
            d = {'Text_Content': [str(user_input)]}
            df = pd.DataFrame(data=d)
            text_counts = self.vectorizer.transform(df['Text_Content'])
            # Reshape into format required by MLP
            text_np = text_counts.toarray()
            prediction = self.textModel.predict(text_np[0:1])
            print(prediction)
            pred_emotion = emotion_list[np.argmax(prediction)]
            return pred_emotion

    def filter_movie(self, pred_emotion):
        """Filter movie titles based on previous user's preference and predicted emotion."""
        data_base = self.database
        print(self.genre)
        print(self.rating)
        print(self.year)
        # Filter by Genre
        data_base = data_base[[str(self.genre) in row for row in data_base['new_genres']]]
        # Filter by Emotions Predicted
        data_base = data_base[data_base.Max == str(pred_emotion.lower())]
        # Filter by Movie Ratings
        if self.rating == 'Average':
            data_base = data_base[(data_base.vote_average < 8) & (data_base.vote_average > 5)]
        if self.rating == 'Excellent':
            data_base = data_base[data_base.vote_average >= 8]
        # Filter by Movie Released Date
        if self.year == '>2015':
            data_base = data_base[data_base.release_date >= 2015]
        if self.year == '2010-2015':
            data_base = data_base[(data_base.release_date >= 2010) & (data_base.release_date < 2015)]
        if self.year == '2005-2010':
            data_base = data_base[(data_base.release_date >= 2005) & (data_base.release_date < 2010)]
        if self.year == '<2005':
            data_base = data_base[data_base.release_date < 2005]
        title_list = data_base.title.to_list()
        self.restart()
        # If no movies are found, randomly recommend 5 movies
        if len(title_list) == 0:
            title_list = random.sample(self.database.title.to_list(), k=5)
        # If there are more than 5 movies, randomly recommend 5
        if len(title_list) > 5:
            title_list = random.sample(title_list, k=5)
        print(title_list)
        return title_list

    def coversation(self, update, context):
        """Conversation with the user for 10 rounds."""
        # If user choose to describe the mood instead of conversing
        if update.message.text == '/describe':
            self.describe(update, context)
            return ConversationHandler.END
        response = self.chatterBot.get_response(update.message.text)
        if str(response) == '':
            response = 'Sorry, can you repeat again?'
        update.message.reply_text(str(response))
        # All user's reply will be stored and used for sentiment analysis
        self.content = self.content + " " + update.message.text
        self.count += 1
        if self.count < 5:
            return CONVERSATION
        else:
            return PREDICT

    def anaylse_sentiments(self, update, context):
        """Handle last conversation and analyse sentiments of all user's conversation."""
        # Prompt user to type /start for a new conversation
        if (self.genre == '') | (self.rating == '') | (self.year == ''):
            update.message.reply_text("Hmmm...you look new here. Please type /start to begin then! 😀😀")
            return
        update.message.reply_text("Thank you for talking to me...let me analyse your current mood.😨😭😡😀")
        self.content = self.content + " " + update.message.text
        print(self.content)
        pred_emotion = self.get_prediction(self.content)
        self.content = ''
        if pred_emotion == 'Anger':
            update.message.reply_text("Oh dear..you seemed 🔥😡😡🔥")
        if pred_emotion == 'Fear':
            update.message.reply_text("Have you seen a 👻? Why do you looked 😨")
        if pred_emotion == 'Joy':
            update.message.reply_text("Whats the occasion today? You are feeling 🎉🥳😀🎉")
        if pred_emotion == 'Sadness':
            update.message.reply_text("Hey there cheer up! Let's not keep a 😥 ya! ")

        movies_title = self.filter_movie(pred_emotion)
        update.message.reply_text("I shall recommend you the following movies!🎬")
        reply = ''
        for index, item in enumerate(movies_title):
            reply = reply + "{}. {}\n".format(index + 1, item)
        update.message.reply_text(reply)
        update.message.reply_text(
            "Please Enjoy the recommended movies!🎞🎞 \nPlease type /start to be recommended more movies!!")
        return ConversationHandler.END

    def textMessage(self, update, context):
        """Handle user's description about mood."""
        # Prompt user to type /start for a new conversation
        if (self.genre == '') | (self.rating == '') | (self.year == ''):
            update.message.reply_text("Hmmm...you look new here. Please type /start to begin then! 😀😀")
            return ConversationHandler.END
        update.message.reply_text("Hmmm...let me analyse your current mood.😨😭😡😀")
        print(update.message.text)
        pred_emotion = self.get_prediction(str(update.message.text))
        if pred_emotion == 'Anger':
            update.message.reply_text("Oh dear..you seemed 🔥😡😡🔥")
        if pred_emotion == 'Fear':
            update.message.reply_text("Have you seen a 👻? Why do you looked 😨")
        if pred_emotion == 'Joy':
            update.message.reply_text("Whats the occasion today? You are feeling 🎉🥳😀🎉")
        if pred_emotion == 'Sadness':
            update.message.reply_text("Hey there cheer up! Let's not keep a 😥 ya! ")

        movies_title = self.filter_movie(pred_emotion)
        update.message.reply_text("I shall recommend you the following movies!🎬")
        reply = ''
        for index, item in enumerate(movies_title):
            reply = reply + "{}. {}\n".format(index + 1, item)
        update.message.reply_text(reply)
        update.message.reply_text(
            "Please Enjoy the recommended movies!🎞🎞 \nPlease type /start to be recommended more movies!!")

    def describe(self, update, context):
        """Command for user to describe own mood."""
        update.message.reply_text("Please describe your current mood. 😀😀")
        return ConversationHandler.END

    def cancel(self, update, context):
        """Command for user to cancel any conversation with bot."""
        self.restart()
        user = update.message.from_user
        logger.info("User %s canceled the conversation.", user.first_name)
        update.message.reply_text('Bye! Do find me again for recommendations via /start!',
                                  reply_markup=ReplyKeyboardRemove())
        return ConversationHandler.END

    def help(self, update, context):
        """Command for user to ask what the bot is about."""
        update.message.reply_text('❓ type /start to begin chatting with Show Me!')

    def error(self, update, context):
        """Log Errors caused by Updates."""
        logger.warning('Update "%s" caused error "%s"', update, context.error)


def initalise_bot():
    """Initalise the chatter bot and text classification mode."""
    chatter_bot = ChatBot("My ChatterBot")
    corpus_trainer = ChatterBotCorpusTrainer(chatter_bot)
    # Trained with Corpus provided with response to some questions
    corpus_trainer.train('chatterbot.corpus.english.conversations')
    corpus_trainer.train('chatterbot.corpus.english.greetings')
    corpus_trainer.train('chatterbot.corpus.english.emotion')
    print("Training Complete!")

    # Reading in all data
    df = pd.read_csv('./Text Analytics Model/Movie_Metadata_Sentiments_Modified.csv')
    # Convert data format for ease of filtering
    for i, row in enumerate(df.itertuples()):
        df.at[i, 'new_genres'] = ' '.join([str(elem) for elem in ast.literal_eval(row[29])])
        if len(str(row[18]).split('/')) > 2:
            df.at[i, 'release_date'] = int(str(row[18]).split('/')[2])

    vectorizer = joblib.load("./Text Analytics Model/vectorizer.pkl")
    print('Text Analytics Model Loaded!')

    # Initialise Telegram bot instance
    my_bot = MyBot(chatter_bot, text_model, df, vectorizer)
    return my_bot


def main():
    # Initalise the Telegram bot instance
    my_bot = initalise_bot()
    updater = Updater('1078316849:AAHiV_ErRcERtAhaEx_fBZ6rR9pk-ljOKAA', use_context=True)

    # Create dispatcher
    dp = updater.dispatcher

    # Conversation Handler to gather inputs from user
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', my_bot.start)],
        states={
            GENRE: [MessageHandler(Filters.regex('^(Drama|Comedy|Action|Romance|Thriller|Family)$'),
                                   my_bot.get_year)],

            YEAR: [MessageHandler(Filters.regex('^(>2015|2010-2015|2005-2010|<2005)$'), my_bot.get_rating)],

            RATINGS: [MessageHandler(Filters.regex('^(Average|Excellent)$'), my_bot.get_mood)],

            CONVERSATION: [MessageHandler(Filters.text, my_bot.coversation)],

            PREDICT: [MessageHandler(Filters.text, my_bot.anaylse_sentiments)]
        },
        fallbacks=[CommandHandler('cancel', my_bot.cancel)]
    )
    dp.add_handler(conv_handler)

    # Add handler for commands
    dp.add_handler(CommandHandler('help', my_bot.help))
    dp.add_handler(CommandHandler('describe', my_bot.describe))

    # Add handler for text message
    text_message_handler = MessageHandler(Filters.text, my_bot.textMessage)
    dp.add_handler(text_message_handler)

    # log all errors
    dp.add_error_handler(my_bot.error)
    print('Bot Up!')

    # Start Polling Response
    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
