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
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn import preprocessing as ppros

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
    # news_df = pd.read_csv('dataset/dataset.csv')
    count_vectorizer = CountVectorizer()
    # x_train_cv = count_vectorizer.fit_transform(news_df['text'])

    x_train_cv = count_vectorizer.fit_transform(news_df['text'])
    pickle.dump(count_vectorizer.vocabulary_, open('model/count_vector.pkl', 'wb'))

    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_cv)
    pickle.dump(tfidf_transformer, open('model/tfidf.pkl', 'wb'))

    x_train, x_test, y_train, y_test = train_test_split(x_train_tfidf, news_df['category'], test_size=0.2)
    #Multinomial Naive Bayes
    # clf = MultinomialNB(alpha=0.12).fit(x_train, y_train) #hyperparameter harus di tune, gotta research more on this 
    # pickle.dump(clf, open('model/multinomial_nb.pkl', 'wb'))
    # load_mnb = pickle.load(open('model/multinomial_nb.pkl', 'rb'))
    # prediction = load_mnb.predict(x_test)

    #SVM
    # clf = svm.LinearSVC().fit(x_train, y_train)
    # pickle.dump(clf, open('model/svm.pkl', 'wb'))
    # load_svm = pickle.load(open('model/svm.pkl', 'rb'))
    # prediction = load_svm.predict(x_test)

    #MLPClassifier
    # clf = MLPClassifier(solver='adam', activation='relu', alpha=3e-4, hidden_layer_sizes=(15,)).fit(x_train, y_train)
    # pickle.dump(clf, open('model/mlp.pkl', 'wb'))
    # load_mlp = pickle.load(open('model/mlp.pkl', 'rb'))
    # prediction = load_mlp.predict(x_test)

    #XGBoost
    encoder = ppros.LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.fit_transform(y_test)
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)
    param = {
        'max_depth': 3,  # the maximum depth of each tree
        'eta': 0.3,  # the training step for each iteration
        'silent': 1,  # logging mode - quiet
        'objective': 'multi:softprob',  # error evaluation for multiclass training
        'num_class': 5}  # the number of classes that exist in this datset
    num_round = 20  # the number of training iterations
    bst = xgb.train(param, dtrain, num_round)
    prediction = bst.predict(dtest)
    best_preds = np.asarray([np.argmax(line) for line in prediction])
    
    result = pd.DataFrame({'actual_label': y_test, 'prediction_label':best_preds})
    result.to_csv('result_dataset_3.csv', sep = ',')

    c_mat = confusion_matrix(y_test, prediction, labels=['sport', 'business', 'politics', 'tech', 'entertainment'])
    acc = accuracy_score(y_test,prediction)
    print("Confusion Matrix: \n", c_mat)
    print("Accuracy: ", acc)

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


train_topic_detection()
# topic_detection()