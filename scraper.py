from bs4 import BeautifulSoup
import requests

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
    return article

def scraper_nbc(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, 'html.parser')

    page = soup.find('div',  {"class": "article-body__content"})
    article = ''
    for x in page.findAll('p'):
        article = article + ' ' + x.text
    return article

def scraper_cnn(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, 'html.parser')

    page = soup.find('div',  {"class": "l-container"})
    article = ''
    for x in page.findAll('div',{"class":"zn-body__paragraph"}):
        article = article + ' ' + x.text
    return article




# url = "http://www.bbc.co.uk/news/live/world-53039952"
# # # scraper_bbc(url)
# scraper_nbc(url)
# scraper_cnn(url)

