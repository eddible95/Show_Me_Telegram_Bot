from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, ConversationHandler
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove
from chatterbot.trainers import ListTrainer, ChatterBotCorpusTrainer
from six.moves import cPickle as Pickle
from chatterbot import ChatBot
import pandas as pd
import random
import logging
import joblib
import ast

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

GENRE, YEAR, RATINGS, CONVERSATION, PREDICT = range(5)


# Stores all variables and functions for the Telegram bot
class MyBot(object):
    def __init__(self, chatter_bot, text_model, data_frame, vectorizer):
        self.chatterBot = chatter_bot
        self.textModel = text_model
        self.vectorizer = vectorizer
        self.genre = ''
        self.year = ''
        self.rating = ''
        self.count = 0
        self.content = ''
        self.database = data_frame

    # Command for user to initiate conversation with the bot
    def start(self, update, context):
        reply_keyboard = [['Drama', 'Comedy'], ['Action', 'Romance'], ['Thriller', 'Family']]
        update.message.reply_text("Hey there friend! Please select a genre of movie that you would be interested in.\n"
                                  "Send /cancel to stop me anytime ❌",
                                  reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True))
        return GENRE

    # Receives the genre of movies and prompt user for released year of movies
    def get_year(self, update, context):
        user = update.message.from_user
        self.genre = update.message.text
        update.message.reply_text('I see you are interested in movies of {} genres'.format(update.message.text))
        logger.info("Genre Selected By %s: %s", user.first_name, update.message.text)
        reply_keyboard = [['>2015', '2010-2015'], ['2005-2010', '<2005']]
        update.message.reply_text("Alrighty! What about the year range for the movie released date? \n"
                                  "Send /cancel to stop me anytime ❌",
                                  reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True))
        return YEAR

    # Receives the released year of movies and prompt user for ratings
    def get_rating(self, update, context):
        user = update.message.from_user
        self.year = update.message.text
        update.message.reply_text('No issue! I\'ll only show movies from this period: {}'.format(update.message.text))
        logger.info("Year Range Selected By %s: %s", user.first_name, update.message.text)
        reply_keyboard = [['Average', 'Excellent']]
        update.message.reply_text("Alrighty! What about the ratings of the movies to be shown?\n"
                                  "Send /cancel to stop me anytime ❌"
                                  , reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True))
        return RATINGS

    # Receives the ratings of movies and prompt user for their current mood
    def get_mood(self, update, context):
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

    # Classify the user's conversation into one of the 4 emotions using trained text classifier
    def get_prediction(self, user_input):
        emotion_list = ['Anger', 'Fear', 'Joy', 'Sadness']
        # Convert user input into format accepted by the text analytics model
        d = {'Text_Content': [str(user_input)]}
        df = pd.DataFrame(data=d)
        text_counts = self.vectorizer.transform(df['Text_Content'])
        # Predicts the emotions
        pred_emotion = self.textModel.predict(text_counts)
        return emotion_list[pred_emotion[0]]

    # Filter movie titles based on previous user's preference and predicted emotion
    def filter_movie(self, pred_emotion):
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
        # If there are more than 5 movies, randomly recommend 5
        if len(title_list) > 5:
            title_list = random.sample(title_list, k=5)
        print(title_list)
        self.genre = ''
        self.rating = ''
        self.year = ''
        return title_list

    # Conversation with the user for 10 rounds
    def coversation(self, update, context):
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
        if self.count < 10:
            return CONVERSATION
        else:
            return PREDICT

    # Handle last conversation and analyse sentiments of all user's conversation
    def anaylse_sentiments(self, update, context):
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
        # response = self.chatterBot.get_response(update.message.text)
        # print(response)
        update.message.reply_text("I shall recommend you the following movies!🎬")
        reply = ''
        for index, item in enumerate(movies_title):
            reply = reply + "{}. {}\n".format(index + 1, item)
        update.message.reply_text(reply)
        update.message.reply_text(
            "Please Enjoy the recommended movies!🎞🎞 \nPlease type /start to be recommended more movies!!")
        return ConversationHandler.END

    # Handle user's description about mood
    def textMessage(self, update, context):
        # Prompt user to type /start for a new conversation
        if (self.genre == '') | (self.rating == '') | (self.year == ''):
            update.message.reply_text("Hmmm...you look new here. Please type /start to begin then! 😀😀")
            return
        update.message.reply_text("Hmmm...let me analyse your current mood.😨😭😡😀")
        print(update.message.text)
        pred_emotion = self.get_prediction(str(update.message.text))
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
        # response = self.chatterBot.get_response(update.message.text)
        # print(response)
        update.message.reply_text("I shall recommend you the following movies!🎬")
        reply = ''
        for index, item in enumerate(movies_title):
            reply = reply + "{}. {}\n".format(index + 1, item)
        update.message.reply_text(reply)
        update.message.reply_text(
            "Please Enjoy the recommended movies!🎞🎞 \nPlease type /start to be recommended more movies!!")

    # Command for user to describe own mood
    def describe(self, update, context):
        update.message.reply_text("Please describe your current mood. 😀😀")
        return ConversationHandler.END

    # Command for user to cancel any conversation with bot
    def cancel(self, update, context):
        user = update.message.from_user
        logger.info("User %s canceled the conversation.", user.first_name)
        update.message.reply_text('Bye! Do find me again for recommendations via /start!',
                                  reply_markup=ReplyKeyboardRemove())
        return ConversationHandler.END

    # Command for user to ask what the bot is about
    def help(self, update, context):
        update.message.reply_text('❓ type /start to begin chatting with Show Me!')

    # For logging of error
    def error(self, update, context):
        """Log Errors caused by Updates."""
        logger.warning('Update "%s" caused error "%s"', update, context.error)


# Initalise the chatter bot and text classification model
def initalise_bot():
    # Chatter bot training for having conversation
    chatter_bot = ChatBot(
        "My ChatterBot",
        logic_adapters=[
            {
                'import_path': 'chatterbot.logic.BestMatch',
                'default_response': 'I am sorry, but I do not understand. I am still learning.'
            }
        ]
    )
    corpus_trainer = ChatterBotCorpusTrainer(chatter_bot)
    # Trained with Corpus provided with response to some questions
    corpus_trainer.train('chatterbot.corpus.english.conversations')
    corpus_trainer.train('chatterbot.corpus.english.greetings')
    corpus_trainer.train('chatterbot.corpus.english.emotion')
    print("Training Complete!")

    # Reading in all data
    df = pd.read_csv('Movie_Metadata_Sentiments_Modified.csv')
    # Convert data format for ease of filtering
    for i, row in enumerate(df.itertuples()):
        df.at[i, 'new_genres'] = ' '.join([str(elem) for elem in ast.literal_eval(row[29])])
        if len(str(row[18]).split('/')) > 2:
            df.at[i, 'release_date'] = int(str(row[18]).split('/')[2])

    # Read in trained Text Classification Model & Vectorzer
    with open('Movie_Metadata_Sentiments_LSV_model.model', 'rb') as file_handle:
        text_model = Pickle.load(file_handle)
    vectorizer = joblib.load("vectorizer.pkl")
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

            YEAR: [MessageHandler(Filters.regex('^(>2015|2010-2015|2005-2010|<2000)$'), my_bot.get_rating)],

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
