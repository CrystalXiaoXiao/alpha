from newsapi import NewsApiClient
from datetime import datetime, timedelta
import pandas as pd 
from credentials.api_key import key
from scraper import scraper_bbc, scraper_nbc, scraper_cnn
import json

newsapi = NewsApiClient(api_key=key)


def find_news():
    with open('article_collection.json', 'w', encoding='utf-8') as output_json:
        pass
    top_headline = newsapi.get_top_headlines(
                                             sources='bbc-news,nbc-news, cnn',
                                             page_size=10,
                                             page= 1,
    )                                           

    articles = top_headline['articles']
    counter = 0 

    for x, y in enumerate(articles):
        print(y["title"] + ' : ' + y["url"])
        
        if (y["content"] is None) or ('/live/' in y["url"] or '/live-news/'in y["url"]): 
            None
        elif counter < 5:
            if y["source"]["name"] == 'BBC News':
                content = scraper_bbc(y["url"])
                    
            elif y["source"]["name"] == 'NBC News':
                content = scraper_nbc(y["url"])

            elif y["source"]["name"] == 'CNN':
                content = scraper_cnn(y["url"])

            to_json(y["title"], y["url"], y["urlToImage"], content)
            counter = counter + 1

def to_json(title, url, image, content):
    news = []
    news.append({
        'title' : title,
        'url' : url, 
        'image' : image,
        'content' : content 
    })

    with open('article_collection.json', 'a+', encoding='utf-8') as output_json:
        json.dump(news, output_json, ensure_ascii=False, indent=4)



find_news()

    

