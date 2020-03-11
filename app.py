import pandas as pd
from flask import Flask, jsonify, request,json
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import requests
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re


app = Flask(__name__)

@app.route('/', methods=['POST'])
def predict():
    # get data
   
    body_dict = request.get_json()
    #data = body_dict['content']
    # predictions

    #tmp = re.sub("\\\\", "", body_dict)
    #txt=tmp.replace('\\','')
    #jsontxt=json.loads(txt)
    
    clean_content=[]
    #scrape_txt['content']=[c.lower() for c in data['content']]
    #for content in txtList:
        # Convert posts to words, then append to clean_train_content.   
    #    clean_content.append(review_to_words(content))

    #tfid_vectorizer = TfidfVectorizer(max_df=.8,ngram_range=(1,2))
# Fit and transform the processed titles
    #count_data = tfid_vectorizer.fit_transform(clean_content)    

    #r = requests.post(url = "https://news-model.herokuapp.com/", data = count_data) 
# S3 Connect
    #s3 = boto3.client('s3')

    # Uploaded File
    #s3.put_object(Bucket=BUCKET_NAME, Key=FILE_NAME, Body=txt)

    return body_dict['content']
    
def review_to_words(raw_content):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    
    # 1. Remove HTML.
    content_text = BeautifulSoup(raw_content).get_text()
    
    # 2. Remove non-letters.
    letters_only = re.sub("[^a-zA-Z]", " ", content_text)
    
    # 3. Convert to lower case, split into individual words.
    words = letters_only.lower().split()
    
    # 4. In Python, searching a set is much faster than searching
    # a list, so convert the stop words to a set.
    stops = stopwords.words('english')
    stops.extend(['http','https','www','com','abcnews','rte','cnn','huffingtonpost','news','bbc','tass','dw','aljeezra','chinadaily','ie','go','politics','said','say','one','would','year','pm', 'nbcsn', 'csn', 'et', 'pt', 'ct', 'ht', 'mt','like','first','two','get'])
  
    # 5. Remove stop words.
    meaningful_words = [w for w in words if not w in stops]
    # 6. Lemmatize our words
    lemmatizer = WordNetLemmatizer()
    tokens_lem = [lemmatizer.lemmatize(i) for i in meaningful_words]
    
    # 7. Join the words back into one string separated by space, 
    # and return the result.
    return(" ".join(tokens_lem))

if __name__ == '__main__':
    app.run(port = 5000, debug=True)