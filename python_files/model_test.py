import numpy as np
import pandas as pd
from sklearn import linear_model
#from sklearn.externals import joblib
import joblib
import pickle
import re
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('wordnet')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer


review_model = pickle.load(open("review_model.pkl", 'rb'))
review_cv = pickle.load(open("review_vectorizer.pkl", 'rb'))

#review_model = joblib.load('review_model.pkl')
#review_cv = joblib.load('review_vectorizer.pkl')

def pre_processing(text):
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = re.sub('[0-9]+','num',text)
    word_list = nltk.word_tokenize(text)
    word_list =  [lemmatizer.lemmatize(item) for item in word_list]
    return ' '.join(word_list)

review_text = pre_processing("The sneakers are great")

rr = review_model.predict_proba(review_cv.transform([review_text]))
#print(rr)
#if rr[:, 0] > rr[:,1]:
#  print('Kindly rate this as a negative review')
#else:
#  print('Kindly rate this as a positive review ')

if rr[1][1] >= 0.5:
    prediction = "Negative"
    print(prediction)
    # pr = prob[0][0]
else:
    prediction = "Positive"
    print(prediction)
    # pr = prob[0][0]
