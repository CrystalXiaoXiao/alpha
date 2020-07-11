import numpy as np 
import pandas as pd 
import string 
import nltk 
import pickle
import json
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


#print(news_df.shape) Total 2225,2  
#print(news_df['category'].value_counts()) sport: 511, business: 510, politics: 417, tech: 401, entertainment: 386

def preprocessing(data):
    article = []
    article = word_tokenize(data)
    stop_words = set(stopwords.words('english'))

    article = [char for char in article if char not in string.punctuation] #remove punctuatin
    # article = ' '.join(article) 
    article = [word for word in article if word.lower() not in stop_words] #remove stopwords
    article = ' '.join(article)

    return article    

def train_topic_detection():
    news_df = pd.read_csv('dataset/bbc-text.csv')
    # news_df = pd.read_json('dataset/News_Category_Dataset_v2.json', lines=True)
    count_vectorizer = CountVectorizer()
    x_train_cv = count_vectorizer.fit_transform(news_df['text'])
    pickle.dump(count_vectorizer.vocabulary_, open('model/count_vector.pkl', 'wb'))

    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_cv)
    pickle.dump(tfidf_transformer, open('model/tfidf.pkl', 'wb'))

    x_train, x_test, y_train, y_test = train_test_split(x_train_tfidf, news_df['category'], test_size=0.2)
    clf = MultinomialNB().fit(x_train, y_train)
    pickle.dump(clf, open('model/multinomial_nb.pkl', 'wb'))

    load_mnb = pickle.load(open('model/multinomial_nb.pkl', 'rb'))
    prediction = load_mnb.predict(x_test)

    result = pd.DataFrame({'actual_label': y_test, 'prediction_label':prediction})
    # result.to_csv('result_dataset_2.csv', sep = ',')
    print(result)

    #https://www.youtube.com/watch?v=HeKchZ1dauM&t=15s => kalo mau nonton tutorial + repo githubnya 

def topic_detection():
    test_df = pd.read_json('article_collection.json')
    load_vec = CountVectorizer(vocabulary=pickle.load(open("model/count_vector.pkl", "rb")))
    news_cv = load_vec.transform(test_df['content'])

    load_tfidf = pickle.load(open("model/tfidf.pkl", "rb"))
    news_tfidf = load_tfidf.transform(news_cv)

    load_mnb = pickle.load(open('model/multinomial_nb.pkl', 'rb'))
    prediction = load_mnb.predict(news_tfidf)
    print(prediction)
    add_prediction_to_json_output(prediction)

def add_prediction_to_json_output(prediction):
    with open('article_collection.json', 'rb') as input_json:
        data = json.load(input_json)
        for i in range(len(data)):
            data[i]['predicted_topic'] = prediction[i]
    
    with open('article_collection.json', 'w', encoding='utf-8') as output_json:
        json.dump(data, output_json, ensure_ascii=False, indent=4)
