from flask import Flask, request, jsonify
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
# Python libraries
import numpy as np
import pandas as pd
import joblib
import re
from bs4 import BeautifulSoup
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet 
from sklearn.preprocessing import MultiLabelBinarizer

app = Flask(__name__)

from pickle import load
final_model=load(open("final_model.pkl","rb"))
multilabel_binarizer=load(open("multilabel_binarizer.pkl","rb"))
#question cleaned:
def get_wordnet_pos(word):
   """Map POS tag to first character lemmatize() accepts"""
   tag = nltk.pos_tag([word])[0][1][0].upper()
   tag_dict = {"J": wordnet.ADJ,
               "N": wordnet.NOUN,
               "V": wordnet.VERB,
               "R": wordnet.ADV}
            
   return tag_dict.get(tag, wordnet.NOUN)


def sentence_cleaner(text):
   """Function clean text by removing extra spaces, unicode characters,
       English contractions, links, punctuation and numbers.
   """
   # Make text lowercase
   text = text.lower()
   # Remove English contractions
   text = re.sub("\'\w+", ' ', text)
   #
   text = text.encode("ascii", "ignore").decode()
   # Remove ponctuation (except # and ++ for c# and c++)
   text = re.sub('[^\\w\\s#\\s++]', ' ', text)
   # Remove numbers
   text = re.sub(r'\w*\d+\w*', ' ', text)
   # Remove extra spaces
   text = re.sub('\s+', ' ', text)
   text = BeautifulSoup(text, "lxml").text
   # Tokenization
   text = text.split()

   # List of stop words from NLTK
   stop_words = stopwords.words('english')
   # Remove stop words
   text = [word for word in text if word not in stop_words
           and len(word) > 2]
   # Lemmatizer
   lemm = WordNetLemmatizer()
   text = [lemm.lemmatize(w, get_wordnet_pos(w)) for w in text]

   # Return cleaned text
   return text
@app.route('/results', methods=['POST'])
def predict():

 data = request.get_json(force=True)
 print('data=', data)
 question=sentence_cleaner(data["question"])
 print(question)

 y_pred = final_model.predict_proba(np.array(['question']))
 print('y_pred=', y_pred)
 threshold=0.02
 y_pred = (y_pred > threshold) * 1
 y_pred_inversed = multilabel_binarizer.inverse_transform(y_pred)
 print('tags proposées=', y_pred_inversed)

 response = {"tags proposées": y_pred_inversed[0]}
 return jsonify(response)
 
if __name__ == '__main__':
 app.run(port=5000, debug=True)

