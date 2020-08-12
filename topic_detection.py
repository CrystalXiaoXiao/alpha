import numpy as np 
import pandas as pd 
import pickle
import json
import matplotlib.pyplot as plt  
import xgboost as xgb
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, plot_confusion_matrix, classification_report
from preprocessing import preprocessing
from sklearn.neural_network import MLPClassifier

# news_df = pd.read_csv('dataset/bbc-combined.csv')
# print(news_df.shape) #Total 2225,2  
# print(news_df['category'].value_counts()) #sport: 1022, business: 1020, politics: 834, tech: 802, entertainment: 772

def train_naive_bayes():
    res = []
    news_df = pd.read_csv('dataset/bbc-combined.csv')
    news_df['text'] = news_df['text'].apply(lambda x: preprocessing(x))
    
    count_vectorizer = CountVectorizer()
    x_train_cv = count_vectorizer.fit_transform(news_df['text'])

    pickle.dump(count_vectorizer.vocabulary_, open('model/count_vector.pkl', 'wb'))

    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_cv)
    pickle.dump(tfidf_transformer, open('model/tfidf.pkl', 'wb'))

    kf = KFold(n_splits=5, shuffle=True)
    for train_index, test_index in kf.split(x_train_tfidf, news_df['category']):
        x_train, x_test = x_train_tfidf[train_index], x_train_tfidf[test_index]
        y_train, y_test = news_df['category'][train_index], news_df['category'][test_index] 
        clf_mnb = MultinomialNB(alpha=0.12).fit(x_train, y_train) #hyperparameter harus di tune, gotta research more on this 
        pickle.dump(clf_mnb, open('model/multinomial_nb.pkl', 'wb'))

        load_mnb = pickle.load(open('model/multinomial_nb.pkl', 'rb'))
        prediction = load_mnb.predict(x_test)
        print('MNB: ',accuracy_score(y_test,prediction))
        res.append(accuracy_score(y_test,prediction))
    
    print(f"MNB AVG Accuracy: {sum(res)/len(res)}")  

        # plot_confusion_matrix(clf_mnb, x_test, y_test)
        # print(classification_report(y_test, prediction))
        # plt.show()

        #https://www.youtube.com/watch?v=HeKchZ1dauM&t=15s => kalo mau nonton tutorial + repo githubnya 

def train_svm():
    res = []
    news_df = pd.read_csv('dataset/bbc-combined.csv')
    news_df['text'] = news_df['text'].apply(lambda x: preprocessing(x))
    
    count_vectorizer = CountVectorizer()
    x_train_cv = count_vectorizer.fit_transform(news_df['text'])

    pickle.dump(count_vectorizer.vocabulary_, open('model/count_vector.pkl', 'wb'))

    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_cv)
    pickle.dump(tfidf_transformer, open('model/tfidf.pkl', 'wb'))

    kf = KFold(n_splits=5, shuffle=True)
    for train_index, test_index in kf.split(x_train_tfidf, news_df['category']):
        x_train, x_test = x_train_tfidf[train_index], x_train_tfidf[test_index]
        y_train, y_test = news_df['category'][train_index], news_df['category'][test_index] 
    
        clf_svm = svm.SVC(kernel='linear')
        clf_svm.fit(x_train, y_train)
        pickle.dump(clf_svm, open("model/svm.pkl", "wb"))

        svm_prediction = clf_svm.predict(x_test)
        print('SVM: ',accuracy_score(y_test,svm_prediction))
        res.append(accuracy_score(y_test,svm_prediction))
    
    print(f"SVM AVG Accuracy: {sum(res)/len(res)}")  
        # plot_confusion_matrix(clf_svm, x_test, y_test)
        # print(classification_report(y_test, svm_prediction))
        # plt.show()

def train_xgb():
    res = []
    news_df = pd.read_csv('dataset/bbc-combined.csv')
    news_df['text'] = news_df['text'].apply(lambda x: preprocessing(x))
    
    count_vectorizer = CountVectorizer()
    x_train_cv = count_vectorizer.fit_transform(news_df['text'])

    pickle.dump(count_vectorizer.vocabulary_, open('model/count_vector.pkl', 'wb'))

    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_cv)
    pickle.dump(tfidf_transformer, open('model/tfidf.pkl', 'wb'))

    kf = KFold(n_splits=5, shuffle=True)
    for train_index, test_index in kf.split(x_train_tfidf, news_df['category']):
        x_train, x_test = x_train_tfidf[train_index], x_train_tfidf[test_index]
        y_train, y_test = news_df['category'][train_index], news_df['category'][test_index] 
        clf_xgb = xgb.XGBClassifier(objective='multimax:softmax', num_class=5)
        clf_xgb.fit(x_train, y_train)
        pickle.dump(clf_xgb, open("model/xgboost.pkl", "wb"))
        
        xgboost_prediction = clf_xgb.predict(x_test)
        print('XGB: ',accuracy_score(y_test,xgboost_prediction))
        res.append(accuracy_score(y_test,xgboost_prediction))
    
    print(f"XGB Average Accuracy: {sum(res)/len(res)}")  

        # plot_confusion_matrix(clf_xgb, x_test, y_test)
        # print(classification_report(y_test, xgboost_prediction))
        # plt.show()

def train_knn():
    res = []
    news_df = pd.read_csv('dataset/bbc-combined.csv')
    news_df['text'] = news_df['text'].apply(lambda x: preprocessing(x))
    
    count_vectorizer = CountVectorizer()
    x_train_cv = count_vectorizer.fit_transform(news_df['text'])

    pickle.dump(count_vectorizer.vocabulary_, open('model/count_vector.pkl', 'wb'))

    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_cv)
    pickle.dump(tfidf_transformer, open('model/tfidf.pkl', 'wb'))


    kf = KFold(n_splits=5, shuffle=True)
    for train_index, test_index in kf.split(x_train_tfidf, news_df['category']):
        x_train, x_test = x_train_tfidf[train_index], x_train_tfidf[test_index]
        y_train, y_test = news_df['category'][train_index], news_df['category'][test_index] 
        
        clf_knn = KNeighborsClassifier(n_neighbors=4)
        clf_knn.fit(x_train, y_train)
        
        pickle.dump(clf_knn, open("model/knn.pkl", "wb"))
        knn_prediction = clf_knn.predict(x_test)
        print('KNN: ',accuracy_score(y_test,knn_prediction))
        res.append(accuracy_score(y_test,knn_prediction))
    
    print(f"KNN Mean Accuracy: {sum(res)/len(res)}")  

def train_logistic_regression():
    res = []
    news_df = pd.read_csv('dataset/bbc-combined.csv')
    news_df['text'] = news_df['text'].apply(lambda x: preprocessing(x))
    
    count_vectorizer = CountVectorizer()
    x_train_cv = count_vectorizer.fit_transform(news_df['text'])

    pickle.dump(count_vectorizer.vocabulary_, open('model/count_vector.pkl', 'wb'))

    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_cv)
    pickle.dump(tfidf_transformer, open('model/tfidf.pkl', 'wb'))


    kf = KFold(n_splits=5, shuffle=True)
    for train_index, test_index in kf.split(x_train_tfidf, news_df['category']):
        x_train, x_test = x_train_tfidf[train_index], x_train_tfidf[test_index]
        y_train, y_test = news_df['category'][train_index], news_df['category'][test_index] 
        
        clf_lr = LogisticRegression()
        clf_lr.fit(x_train, y_train)
        pickle.dump(clf_lr, open("model/lr.pkl", "wb"))

        lr_prediction = clf_lr.predict(x_test)
        print('LR: ',accuracy_score(y_test,lr_prediction))
        res.append(accuracy_score(y_test,lr_prediction))
    
    print(f"Logistic Regression Mean Accuracy: {sum(res)/len(res)}")  

def train_mlp():
    res = []
    news_df = pd.read_csv('dataset/bbc-combined.csv')
    news_df['text'] = news_df['text'].apply(lambda x: preprocessing(x))
    
    count_vectorizer = CountVectorizer()
    x_train_cv = count_vectorizer.fit_transform(news_df['text'])

    pickle.dump(count_vectorizer.vocabulary_, open('model/count_vector.pkl', 'wb'))

    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_cv)
    pickle.dump(tfidf_transformer, open('model/tfidf.pkl', 'wb'))

    kf = KFold(n_splits=5, shuffle=True)
    for train_index, test_index in kf.split(x_train_tfidf, news_df['category']):
        x_train, x_test = x_train_tfidf[train_index], x_train_tfidf[test_index]
        y_train, y_test = news_df['category'][train_index], news_df['category'][test_index] 
        clf_mlp = MLPClassifier(solver='adam', activation='relu', alpha=3e-4, hidden_layer_sizes=(15,)).fit(x_train, y_train) 
        pickle.dump(clf_mlp, open('model/mlp.pkl', 'wb'))

        load_mlp = pickle.load(open('model/mlp.pkl', 'rb'))
        prediction = load_mlp.predict(x_test)
        print('MLP: ',accuracy_score(y_test,prediction))
        res.append(accuracy_score(y_test,prediction))
    
    print(f"MLP AVG Accuracy: {sum(res)/len(res)}")

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


# train_naive_bayes()
# train_svm()
# train_xgb()
# train_knn()
# train_logistic_regression()
train_mlp()
# topic_detection()
