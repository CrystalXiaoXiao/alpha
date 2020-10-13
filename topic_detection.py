import numpy as np 
import pandas as pd 
import pickle
import json
import matplotlib.pyplot as plt  
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, plot_confusion_matrix, classification_report
from preprocessing import preprocessing
from scipy.sparse import csr_matrix
import math
# news_df = pd.read_csv('dataset/bbc-combined.csv')
# print(news_df.shape) #Total 2225,2  
# print(news_df['category'].value_counts()) #sport: 1022, business: 1020, politics: 834, tech: 802, entertainment: 772

def train_svm():
    res = []
    news_df = pd.read_csv('dataset/bbc-text.csv')
    news_df['text'] = news_df['text'].apply(lambda x: preprocessing(x))

    count_vectorizer = CountVectorizer()
    x_train_cv = count_vectorizer.fit_transform(news_df['text'])

    pickle.dump(count_vectorizer.vocabulary_, open('model/count_vector.pkl', 'wb'))

    tfidf_transformer = TfidfTransformer()
    tfidf = tfidf_transformer.fit_transform(x_train_cv)
    pickle.dump(tfidf_transformer, open('model/tfidf.pkl', 'wb'))

    matrix = tfidf.toarray()

    for i in range(0, len(matrix)):
        N = len(news_df.index) 
        n1 = (news_df['category']==news_df['category'][i]).sum()
        matrix[i] = matrix[i] * math.log2( N/n1)

    x_train_tfidf = csr_matrix(matrix)
    print(pd.DataFrame(x_train_tfidf.toarray(), columns=count_vectorizer.get_feature_names()))

    kf = KFold(n_splits=5, shuffle=True)
    for train_index, test_index in kf.split(x_train_tfidf, news_df['category']):  #training_data (n_sample, n_feature) , target variable (n_sample)           
        x_train, x_test = x_train_tfidf[train_index], x_train_tfidf[test_index] 
        y_train, y_test = news_df['category'][train_index], news_df['category'][test_index] 

        clf_svm = svm.SVC(kernel='linear', C=1.4)
        clf_svm.fit(x_train, y_train)

        svm_prediction = clf_svm.predict(x_test)
        # svm_prediction = clf_svm.score(x_train, y_train)
        # print(svm_prediction)

        print('SVM: ',accuracy_score(y_test ,svm_prediction))
        res.append(accuracy_score(y_test ,svm_prediction))
        # plot_confusion_matrix(clf_svm, x_test, y_test)
        # print(classification_report(y_test, svm_prediction))
        # plt.show()

    print(f"SVM AVG Accuracy: {sum(res)/len(res)}")  

    pickle.dump(clf_svm, open("model/svm.pkl", "wb"))

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

def data_dummy():
    res = []
    news_df = pd.DataFrame({
        "text":["The deadly viruses that vanished without trace","Ndombele give an immediate impact for Spurs", "Virus impact on immune system", "Spurs win on new tactical system"],
        "category":["health","sports","health","sports"]
    })
    news_df['text'] = news_df['text'].apply(lambda x: preprocessing(x))

    count_vectorizer = CountVectorizer()
    x_train_cv = count_vectorizer.fit_transform(news_df['text'])
    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_cv)
    matrix = x_train_tfidf.toarray()

    for i in range(0, len(matrix)):
        N = len(news_df.index) 
        n1 = (news_df['category']==news_df['category'][i]).sum()
        matrix[i] = matrix[i] * math.log2( N/n1)

    x_train_tfidf = csr_matrix(matrix)
    print(pd.DataFrame(x_train_tfidf.toarray(), columns=count_vectorizer.get_feature_names()))

train_svm()
# topic_detection()
# data_dummy()
