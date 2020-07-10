from newsapi import NewsApiClient
from datetime import datetime, timedelta
import pandas as pd 
from credentials.api_key import key
from scraper import scraper_bbc, scraper_nbc, scraper_cnn
from topic_detection import topic_detection
import json

newsapi = NewsApiClient(api_key=key)


def find_news():
    top_headline = newsapi.get_top_headlines(
                                             sources='bbc-news,nbc-news, cnn',
                                             page_size=12,
                                             page= 1,
    )                                           

    articles = top_headline['articles']
    counter = 0 
    news = []

    for x, y in enumerate(articles):
        print(y["title"] + ' : ' + y["url"])
        
        if (y["content"] is None) or ('/live/' in y["url"] or '/live-news/'in y["url"] or '/videos/' in y["url"]): 
            None
        elif counter < 5:
            if y["source"]["name"] == 'BBC News':
                try:
                    content = scraper_bbc(y["url"])
                except:
                    return None
                    
            elif y["source"]["name"] == 'NBC News':
                try:
                    content = scraper_nbc(y["url"])
                except:
                    return None

            elif y["source"]["name"] == 'CNN':
                try:
                    content = scraper_cnn(y["url"])
                except:
                    return None

            news.append({
                'title' : y["title"],
                'url' : y["url"], 
                'image' : y["urlToImage"],
                'content' : content 
            })
            counter = counter + 1
    
    to_json(news)


def to_json(news):
    
    with open('article_collection.json', 'w', encoding='utf-8') as output_json:
        json.dump(news, output_json, ensure_ascii=False, indent=4)



find_news()
topic_detection()
    

