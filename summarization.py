from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
import torch 
import pandas as pd 


text = """
Tottenham Hotspur boss Jose Mourinho says he trusts England manager Gareth Southgate and Wales counterpart Ryan Giggs to protect his players. 
Spurs will play their eighth game since 13 September when they travel to Manchester United on Sunday. 
The Old Trafford match falls prior to an international break when countries play three games in a week. "You know, I believe that Gareth and [assistant] Steve [Holland], 
they care about the players," said Mourinho. Mourinho worked with Southgate's assistant Holland at Chelsea but says he will not request that England captain Kane is rested 
by his country and is confident the international coaching teams will look after his players' welfare. "I don't think they want to be connected with something that can be 
a consequence of this week and the three international matches, which is obviously too much, especially for my players," added the Portuguese.
"So I don't speak with Gareth or even with Steve, and of course with Steve I am a very good friends. I just let them do the job in the way they want to do it with 
the freedom they deserve. "I didn't speak with Ryan Giggs either. I just leave with them the respect they will have for their players. The players are our players but also theirs." 
Tottenham played Newcastle United on Sunday and Chelsea on Tuesday prior to games against Maccabi Haifa on Thursday and Manchester United on Sunday. 
"Hopefully, Gareth and Steve understand what happened with Tottenham this week and they respect the players," added Mourinho. 
"That's just my hope but I'm not going to call, or ask, or beg. I'm not going to press. I think they deserve their freedom and I have the utmost respect for them."
"""
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

news_df = pd.read_json('article_collection.json')

for text in news_df['content']:
    inputs = tokenizer([text], max_length=1024, return_tensors='pt')
    summary_ids = model.generate(inputs['input_ids'])
    print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])

