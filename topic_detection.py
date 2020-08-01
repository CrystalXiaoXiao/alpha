import numpy as np 
import pandas as pd 
import string 
import nltk 
import pickle
import json
import re
import matplotlib.pyplot as plt  
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, plot_confusion_matrix


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
    article = article.lower()
    article = re.sub('\[\d+\]', '', article) #[1]/[2]/[n] => ''
    article = re.sub('\d+.', '', article) #1. / 2. => ''

    return article    

def train_naive_bayes():
    news_df = pd.read_csv('dataset/bbc-text.csv')
    news_df['text'] = news_df['text'].apply(lambda x: preprocessing(x))
    
    count_vectorizer = CountVectorizer()
    x_train_cv = count_vectorizer.fit_transform(news_df['text'])

    pickle.dump(count_vectorizer.vocabulary_, open('model/count_vector.pkl', 'wb'))

    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_cv)
    pickle.dump(tfidf_transformer, open('model/tfidf.pkl', 'wb'))

    x_train, x_test, y_train, y_test = train_test_split(x_train_tfidf, news_df['category'], test_size=0.2, random_state=42)
    clf_mnb = MultinomialNB(alpha=0.12).fit(x_train, y_train) #hyperparameter harus di tune, gotta research more on this 
    pickle.dump(clf_mnb, open('model/multinomial_nb.pkl', 'wb'))

    load_mnb = pickle.load(open('model/multinomial_nb.pkl', 'rb'))
    prediction = load_mnb.predict(x_test)

    # result = pd.DataFrame({'actual_label': y_test, 'prediction_label':prediction})

    # print("Confusion Matrix: \n", confusion_matrix(y_test, prediction))
    print("MNB Accuracy: ", accuracy_score(y_test,prediction))
    plot_confusion_matrix(clf_mnb, x_test, y_test)
    plt.show()

    # print(result)

    #https://www.youtube.com/watch?v=HeKchZ1dauM&t=15s => kalo mau nonton tutorial + repo githubnya 

def train_svm():
    news_df = pd.read_csv('dataset/bbc-text.csv')
    news_df['text'] = news_df['text'].apply(lambda x: preprocessing(x))
    
    count_vectorizer = CountVectorizer()
    x_train_cv = count_vectorizer.fit_transform(news_df['text'])

    pickle.dump(count_vectorizer.vocabulary_, open('model/count_vector.pkl', 'wb'))

    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_cv)
    pickle.dump(tfidf_transformer, open('model/tfidf.pkl', 'wb'))
    x_train, x_test, y_train, y_test = train_test_split(x_train_tfidf, news_df['category'], test_size=0.25, random_state=42)
    
    clf_svm = svm.SVC(kernel='linear')
    clf_svm.fit(x_train, y_train)
    pickle.dump(clf_svm, open("model/svm.pkl", "wb"))

    svm_prediction = clf_svm.predict(x_test)
    # svm_result = pd.DataFrame({'actual_label': y_test, 'prediction_label':svm_prediction})

    # print("Confusion Matrix: \n", confusion_matrix(y_test, svm_prediction))
    print("SVM Accuracy: ", accuracy_score(y_test,svm_prediction))
    plot_confusion_matrix(clf_svm, x_test, y_test)
    plt.show()


def topic_detection():
    test_df = pd.read_json('article_collection.json')
    test_df['content'] = test_df['content'].apply(lambda x: preprocessing(x))

    load_vec = CountVectorizer(vocabulary=pickle.load(open("model/count_vector.pkl", "rb")))
    news_cv = load_vec.transform(test_df['content'])

    load_tfidf = pickle.load(open("model/tfidf.pkl", "rb"))
    news_tfidf = load_tfidf.transform(news_cv)

    # load_mnb = pickle.load(open('model/multinomial_nb.pkl', 'rb'))
    load_svm = pickle.load(open('model/svm.pkl', 'rb'))
    prediction = load_svm.predict(news_tfidf)
    print(prediction)
    add_prediction_to_json_output(prediction)

def add_prediction_to_json_output(prediction):
    with open('article_collection.json', 'rb') as input_json:
        data = json.load(input_json)
        for i in range(len(data)):
            data[i]['predicted_topic'] = prediction[i]
    
    with open('article_collection.json', 'w', encoding='utf-8') as output_json:
        json.dump(data, output_json, ensure_ascii=False, indent=4)


# train_svm()
# train_naive_bayes()
# topic_detection()
