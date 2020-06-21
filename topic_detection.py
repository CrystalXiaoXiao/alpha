import pandas as pd 
from sklearn.model_selection import train_test_split

news_df = pd.read_csv('dataset/bbc-text.csv', sep=',')
x_train, x_test, y_train, y_test = train_test_split(news_df['text'], news_df['category'], random_state=1, train_size =0.8)

print("Training dataset: ", x_train.shape[0])
print("Test dataset: ", x_test.shape[0])

# print(news_df.head())