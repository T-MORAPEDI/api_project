from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import re
import numpy as np

nltk.download('wordnet')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer

review_model = pickle.load(open("review_model.pkl", 'rb'))
review_cv = pickle.load(open("review_vectorizer.pkl", 'rb'))

def pre_processing(text):
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = re.sub('[0-9]+','num',text)
    word_list = nltk.word_tokenize(text)
    word_list =  [lemmatizer.lemmatize(item) for item in word_list]
    return ' '.join(word_list)

# Create your views here.
@api_view(['GET', 'POST'])
def predict(request):
    if request.method == "GET":
        return Response({'message': 'Review Classifictaion API is working!'}, status=status.HTTP_200_OK)
    
    elif request.method == "POST":
        to_predict_list = request.form.to_dict()
        review_text = pre_processing(to_predict_list['review_text'])
        
        pred = review_model.predict(review_cv.transform([review_text]))
        prob = review_model.predict_proba(review_cv.transform([review_text]))
        
        if prob[0][0]>0.5:
            prediction = "Negative"
            prob_bar = np.round(prob[0][0],3) * 100

        else:
            prediction = "Positive"
            prob_bar = np.round(prob[0][1], 3) * 100
            
        return render('form.html', prediction = prediction, prob = prob_bar)