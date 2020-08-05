import numpy as np 
import pandas as pd 
import pickle
import json
import matplotlib.pyplot as plt  
import xgboost as xgb
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, plot_confusion_matrix, classification_report
from preprocessing import preprocessing

#print(news_df.shape) Total 2225,2  
#print(news_df['category'].value_counts()) sport: 511, business: 510, politics: 417, tech: 401, entertainment: 386

def train_naive_bayes():
    news_df = pd.read_csv('dataset/bbc-text.csv')
    news_df['text'] = news_df['text'].apply(lambda x: preprocessing(x))
    
    count_vectorizer = CountVectorizer()
    x_train_cv = count_vectorizer.fit_transform(news_df['text'])

    pickle.dump(count_vectorizer.vocabulary_, open('model/count_vector.pkl', 'wb'))

    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_cv)
    pickle.dump(tfidf_transformer, open('model/tfidf.pkl', 'wb'))

    x_train, x_test, y_train, y_test = train_test_split(x_train_tfidf, news_df['category'], test_size=0.25, random_state=42)
    clf_mnb = MultinomialNB(alpha=0.12).fit(x_train, y_train) #hyperparameter harus di tune, gotta research more on this 
    pickle.dump(clf_mnb, open('model/multinomial_nb.pkl', 'wb'))

    load_mnb = pickle.load(open('model/multinomial_nb.pkl', 'rb'))
    prediction = load_mnb.predict(x_test)

    print("MNB Accuracy: ", accuracy_score(y_test,prediction))
    plot_confusion_matrix(clf_mnb, x_test, y_test)
    print(classification_report(y_test, prediction))
    # plt.show()

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
    print("SVM Accuracy: ", accuracy_score(y_test,svm_prediction))
    plot_confusion_matrix(clf_svm, x_test, y_test)
    print(classification_report(y_test, svm_prediction))
    # plt.show()

def train_xgb():
    news_df = pd.read_csv('dataset/bbc-text.csv')
    news_df['text'] = news_df['text'].apply(lambda x: preprocessing(x))
    
    count_vectorizer = CountVectorizer()
    x_train_cv = count_vectorizer.fit_transform(news_df['text'])

    pickle.dump(count_vectorizer.vocabulary_, open('model/count_vector.pkl', 'wb'))

    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_cv)
    pickle.dump(tfidf_transformer, open('model/tfidf.pkl', 'wb'))
    x_train, x_test, y_train, y_test = train_test_split(x_train_tfidf, news_df['category'], test_size=0.25, random_state=42)

    clf_xgb = xgb.XGBClassifier(objective='multimax:softmax', num_class=5)
    clf_xgb.fit(x_train, y_train)
    pickle.dump(clf_xgb, open("model/xgboost.pkl", "wb"))
    
    xgboost_prediction = clf_xgb.predict(x_test)
    print("XGBoost Accuracy: ", accuracy_score(y_test,xgboost_prediction))
    plot_confusion_matrix(clf_xgb, x_test, y_test)
    print(classification_report(y_test, xgboost_prediction))

    # plt.show()

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


train_naive_bayes()
train_svm()
train_xgb()
# topic_detection()
