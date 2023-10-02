################################# LOADING MODULES ################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import time
from sklearn.model_selection import train_test_split
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import re

# NLTK Stop words
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['make', 'want', 'seem', 'run', 'need', 'even', 'not', 'would', 'say', 'could', '_', 
                'be', 'know', 'go', 'get', 'do','get','far','also','way','app','usd','eur','jai','hind','jai_hind',
'done', 'try', 'many','from', 'subject', 're', 'edu','some', 'nice', 'thank','singh','mast','untuk','apne','nise','vgood',
'think', 'see', 'rather', 'lot', 'line', 'even', 'also', 'may', 'use','goog','nce','aap','thik','hai','setu','aarogya_setu','aarogya',
'take', 'come','recommend','recommended','paypal','paytm', 'let','ad','add'])

from nltk.stem import PorterStemmer

############################ DATA PREPROCESSING ##############################

data = pd.read_csv('sneakers_Reviews_Dataset.csv', sep = ';')

data['review_text'] = data['review_text'].apply(str)

def remove_tags(raw_text):
    cleaned_text = re.sub(re.compile('<.*?>'), '', raw_text)
    return cleaned_text

data['review_text'] = data['review_text'].apply(remove_tags)

data['review_text'] = data['review_text'].apply(lambda x:x.lower())

sw_list = stopwords.words('english')
data['review_text'] = data['review_text'].apply(lambda x: [item for item in x.split() if item not in sw_list]).apply(lambda x:" ".join(x))

######################### FEATURE EXTRACTION ##########################

data_reviews = data[['product_id', 'rating', 'review_text']]

######################### LABEL CREATION ##############################

sentiment_mapping = {
    5: 'Positive',
    4: 'Positive',
    3: 'Neutral',
    2: 'Negative',
    1: 'Negative'
}

data_reviews['sentiment'] = data_reviews['rating'].map(sentiment_mapping)

# RATINGS VS NUMBER OF REVIEWS
label_count = data_reviews['rating'].value_counts()
label_count = label_count.sort_index()

#fig = plt.figure(figsize=(6, 6))
#ax = sns.barplot(label_count.index, label_count.values)
#plt.title("Class Distribution",fontsize = 20)
#plt.ylabel('Number of Reviews', fontsize = 12)
#plt.xlabel('Sentiment', fontsize = 12)

# SENTIMENT VS NUMBER OF REVIEWS
label_count = data_reviews['sentiment'].value_counts()
label_count = label_count.sort_index()

#fig = plt.figure(figsize=(6, 6))
#ax = sns.barplot(label_count.index, label_count.values)
#plt.title("Class Distribution",fontsize = 20)
#plt.ylabel('Number of Reviews', fontsize = 12)
#plt.xlabel('Sentiment', fontsize = 12)


X = data_reviews[['product_id', 'rating', 'review_text']] #variables
y = data_reviews['sentiment'] #target

#
encoder = LabelEncoder()
y = encoder.fit_transform(y)

############################ MODEL SELECTION ##############################

# LOGISTIC REGRESSION
lr = LogisticRegression(solver = 'liblinear', random_state = 42, max_iter=1000)

########################### TRAIN-TEST SPLIT ##############################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

########################### SAVING REVIEW VECTORIZER ######################

# Saving model transformer
cv = CountVectorizer()

X_train_ = cv.fit_transform(X_train['review_text']).toarray()
X_test_ = cv.fit_transform(X_test['review_text']).toarray()

pickle.dump(cv, open("review_vectorizer.pkl", "wb"))

########################## MODEL TRAINING ################################

model_fit = lr.fit(X_train_,y_train)
y_pred = lr.predict(X_test_)

# save the model to disk
filename = 'review_model.pkl'
pickle.dump(lr, open(filename, 'wb'))

########################## MODEL EVALUATING ##############################

# ACCURACY SCORE and CONFUSION MATRIX
accuracy_score(y_test, y_pred)

# CLASSIFICATION REPORT
classification_report(y_test, y_pred)

# FEATURE IMPORTANCE
feature_importance = lr.coef_[0][:10]
for i,v in enumerate(feature_importance):
    print('Feature: ', list(cv.vocabulary_.keys())[list(cv.vocabulary_.values()).index(i)], 'Score: ', v)

feature_importance = lr.coef_[0]
sorted_idx = np.argsort(feature_importance)

# TOP 10 WORD/PHRASES FOR THE NEGATIVE RATINGS
top_10_neg_w = [list(cv.vocabulary_.keys())[list(cv.vocabulary_.values()).index(w)] for w in sorted_idx[range(-1,-11, -1)]]

#fig = plt.figure(figsize=(10, 6))
#ax = sns.barplot(top_10_neg_w, feature_importance[sorted_idx[range(-1,-11, -1)]])
#plt.title("Most Important Words Used for Negative Sentiment",fontsize = 20)
#x_locs,x_labels = plt.xticks()
#plt.setp(x_labels, rotation = 40)
#plt.ylabel('Feature Importance', fontsize = 12)
#plt.xlabel('Word', fontsize = 12)

# Save the plot as 'Actual_VS_Predicted.png'
#plt.savefig("TOP_10_NEGATIVE_RATING_PRASES.png")

# TOP 10 WORD/PHRASES FOR THE POSITIVE RATINGS
top_10_pos_w = [list(cv.vocabulary_.keys())[list(cv.vocabulary_.values()).index(w)] for w in sorted_idx[:10]]

#fig = plt.figure(figsize=(10, 6))
#ax = sns.barplot(top_10_pos_w, feature_importance[sorted_idx[:10]])
#plt.title("Most Important Words Used for Positive Sentiment",fontsize = 20)
#x_locs,x_labels = plt.xticks()
#plt.setp(x_labels, rotation = 40)
#plt.ylabel('Feature Importance', fontsize = 12)
#plt.xlabel('Word', fontsize = 12)

# Save the plot as 'Actual_VS_Predicted.png'
#plt.savefig("TOP_10_POSITIVE_RATING_PRASES.png")

# TESTING REVIEWS USING THE SAVED MODEL
# POSITIVE REVIEW
test_review = cv.transform(["The sneakers are great"])
lr.predict_proba(test_review)

# NEGATIVE REVIEW
test_review = cv.transform(["The sneakers are poor"])
lr.predict_proba(test_review)

# TESTING IF THE MODEL CAN BE ABLE TO CLASSIFY CORRECTLY
# TEST REVIEW 1
test_review = cv.transform(["The sneakers are poor"])
prediction = lr.predict_proba(test_review)
if prediction[:, 0] > prediction[:,1]:
    print('Kindly rate this as a negative review')
elif prediction[:, 0] < prediction[:,1]:
    print('Kindly rate this as a positive review ')

# TEST REVIEW 2
test_review2 = cv.transform(["The sneakers are great"])
prediction = lr.predict_proba(test_review2)
if prediction[:, 0] > prediction[:,1]:
    print('Kindly rate this as a negative review')
elif prediction[:, 0] < prediction[:,1]:
    print('Kindly rate this as a positive review ')