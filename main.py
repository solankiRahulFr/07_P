import string
# text preprocessing modules
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re  # regular expression
import uvicorn
from fastapi import FastAPI
import langdetect
from num2words import num2words 
import re
import spacy
from spacy.lang.en import stop_words as spacy_stopwords
import contractions
import emoji
from autocorrect import Speller
import pandas as pd
import requests as r
import pickle
nltk.download('averaged_perceptron_tagger')


app = FastAPI(
    title="Sentiment Model API",
    description="tweet reviews",
)




spell = Speller(lang='en')
stop_words = spacy_stopwords.STOP_WORDS
nlp = spacy.load('en_core_web_lg')

# Defining regex patterns.
urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|(www\.)[^ ]*)"
userPattern       = r'@[^\s]+'
sequencePattern   = r"(.)\1\1+"
seqReplacePattern = r"\1"
chatSlangDf = pd.read_csv("./chat_slang.csv")
chatSlangDf = chatSlangDf[['acronym', 'expansion']]
chatSlangDf.loc[len(chatSlangDf.index)] = ['2an', "to an"]
chatSlangDf

def replace_chat_words(word):
    normal_word = chatSlangDf[chatSlangDf['acronym'].isin([word])]['expansion'].values
    if len(normal_word):
        if word == "lol":
            return normal_word[1]
        else:
            return normal_word[0]
    elif word.isnumeric():
        return num2words(word)
    else:
        return word     
    

def preprocessingText(text):
    text = text.lower()
    
    # Removing all URls
    text = re.sub(urlPattern,'',text)
    
    # Removing all usernames
    text = re.sub(userPattern,'', text)
    
    # Replace all emojis from the emoji shortcodes
    text = " ".join([" ".join(emoji.demojize(x)[1:-1].split("_")) if emoji.is_emoji(x) else x for x in text.split()])

    # Replacing the chat words and numbers
    text = " ".join([replace_chat_words(word) if nltk.pos_tag([word])[0][1] in ['NN', "CD"] else word for word in text.split()])


    # Replacing contractions
    text = contractions.fix(text)

    # Remove non-alphanumeric and symbols (puntuations)
    text = "".join([i for i in text if i not in string.punctuation])

    # Replace 3 or more consecutive letters by 1 letter and lemmatizing the words
    text = " ".join([re.sub(sequencePattern, seqReplacePattern, str(token)) if token.pos_ in ["PROPN", 'NOUN'] else token.text for token in nlp(text)])

    # Replacing mistake of spellings
    text = spell(text)

    # Remove whitespces
    text = text.strip()

    return text



# load the sentiment model
from keras.models import load_model
# model = load_model('model_03.h5')
model = load_model('./model_03.h5', compile=False)
model.compile() 
from tensorflow.keras.preprocessing.sequence import pad_sequences

@app.get("/predict-review")
def predict_sentiment(review: str):
    # A simple function that receive a review content and predict the sentiment of the content.

    # clean the review
    cleaned_review = preprocessingText(review)

    with open("./tokenizer.pkl", 'rb') as pickle_file:
      tokenizer = pickle.load(pickle_file)

    processed_review  = pad_sequences(tokenizer.texts_to_sequences([cleaned_review]), maxlen=300)
    prediction = model.predict(processed_review)
    return ("Negative" if prediction[0][0].item() < 0.5 else "Positive", prediction[0][0].item())