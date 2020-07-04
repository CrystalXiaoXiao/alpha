from bs4 import BeautifulSoup
from topic_detection import preprocessing
import requests
import re

def scraper_bbc(url):

    html = requests.get(url).text
    soup = BeautifulSoup(html, 'html.parser')

    if '/sport/' in url:
        sportCategory = ['/football/','/tennis/','/rugby/', '/cricket/', '/golf/']
        if any(cat in url for cat in sportCategory):
            page = soup.find('div',  {"class": "story-body sp-story-body gel-body-copy"}) 
        else: #boxing, athletics, formula1, cycling, basketball, nfl, winter-sports, horse-racing
            page = soup.find('div', {"class": "qa-story-body story-body gel-pica gel-10/12@m gel-7/8@l gs-u-ml0@l gs-u-pb++"})
    else:
        page = soup.find('div',  {"class": "story-body__inner"})

    article = ''
    for x in page.findAll('p'):
        article = article + ' ' +   x.text
        article = clean_article(article)
    return article

def scraper_nbc(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, 'html.parser')

    page = soup.find('div',  {"class": "article-body__content"})
    article = ''
        
    for x in page.findAll('p'):
        article = article + ' ' +   x.text
        article = clean_article(article)

    return article

def scraper_cnn(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, 'html.parser')

    page = soup.find('div',  {"class": "l-container"})
    article = ''
    for x in page.findAll('div',{"class":"zn-body__paragraph"}):
        article = article + ' ' +   x.text
        article = clean_article(article)

    return article

def clean_article(article):
    article = article.replace('  ', ' ') 
    article = re.sub('\[\d+\]', '', article) #[1]/[2]/[n] => ''
    article = re.sub('\d+.', '', article) #1. / 2. => ''
    article = re.sub('\"', '', article)
    article = article.replace('\n\n', '\n')
    article = preprocessing(article)

    return article

