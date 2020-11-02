from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
import torch 
import pandas as pd 
import nltk
import json

bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

news_df = pd.read_json('article_collection.json')

# for text in news_df['content']:
# inputs = bart_tokenizer([text], truncation=False, min_length=0, max_length=1024, return_tensors='pt')
# summary_ids = bart_model.generate(inputs['input_ids'], max_target_length=100)
# print([bart_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])

# generate chunks of text \ sentences <= 1024 tokens
#append terus sentence yang di tokenize sampai lengthnya 1024 (ke sent)
#kalau sudah lebih dari 1024, maka sent diappend ke nested, lalu bikin sent baru
#nested isinya chunk2 yang length < 1024
def nest_sentences(document):
    nested = []
    sent = []
    length = 0
    for sentence in nltk.sent_tokenize(document):
        length += len(sentence)
        if length < 1024:
          sent.append(sentence)
        else:
          nested.append(sent)
          sent = [sentence]
          length = len(sentence)

    if sent:
        nested.append(sent)
    return nested
  
# generate summary on text with <= 1024 tokens
def generate_summary(nested_sentences):
    summaries = []
    for nested in nested_sentences:
        input_tokenized = bart_tokenizer.encode(' '.join(nested), return_tensors='pt')
        # print(nested)
        # print(input_tokenized)
        # input_tokenized = input_tokenized.to(device)
        summary_ids = bart_model.generate(input_tokenized,
                                          min_length=0,
                                          max_length=1024)
        output = [bart_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
        summaries.append(output)

    summaries = [sentence for sublist in summaries for sentence in sublist]
    return summaries

def add_article_summary_to_json_output(article_summaries):
    with open('article_collection.json', 'rb') as input_json:
        data = json.load(input_json)
        for i in range(len(data)):
            data[i]['article_summary'] = article_summaries[i]
    
    with open('article_collection.json', 'w', encoding='utf-8') as output_json:
        json.dump(data, output_json, ensure_ascii=False, indent=4)

article_summaries = []
for text in news_df['content']:
    total_length = len(text)
    while total_length > 500:
        nested = nest_sentences(text)
        summary = generate_summary(nested)
        summarized = ' '.join(summary)
        text = summarized
        total_length = len(summarized)
    
    # print('Summary News: ', summarized)
    article_summaries.append(summarized)

add_article_summary_to_json_output(article_summaries) 
# print(article_summaries)