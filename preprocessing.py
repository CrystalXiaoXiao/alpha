from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import string 
import nltk 

def preprocessing(data):
    article = []
    article = word_tokenize(data)
    article = [char for char in article if char not in string.punctuation] #remove punctuatin
    stop_words = set(stopwords.words('english'))
    article = [word for word in article if word.lower() not in stop_words] #remove stopwords
    article = ' '.join(article)
    article = article.lower()
    article = re.sub('\[\d+\]', '', article) #[1]/[2]/[n] => ''
    article = re.sub('\d+.', '', article) #1. / 2. => ''
    return article    
