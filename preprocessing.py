from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import *
import re
import string 
import nltk 

def preprocessing(data):
    article = word_tokenize(data)
    stop_words = set(stopwords.words('english'))
    snowball_stemmer = SnowballStemmer(language='english')
    stemmed_article = [snowball_stemmer.stem(word) for word in article]
    article = [word for word in stemmed_article if word.lower() not in stop_words] #remove stopwords
    article = ' '.join(article)
    article = article.lower()
    article = article.translate(str.maketrans('', '', string.punctuation)) #remove punctuation
    article = re.sub('\[\d+\]', '', article) #[1]/[2]/[n] => ''
    article = re.sub('\d+', '', article) #1, 2, 3
    article = article.strip()
    return article    
