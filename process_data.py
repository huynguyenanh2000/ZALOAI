import pandas as pd 
import os 

train_list = []
test_list = []

for file in os.listdir('data\\train\\0'):
    link = 'data\\train\\0\\'+file
    train_list.append([link, 0])

for file in os.listdir('data\\train\\1'):
    link = 'data\\train\\1\\'+file
    train_list.append([link, 1])

for file in os.listdir('data\\val\\0'):
    link = 'data\\val\\0\\'+file
    test_list.append([link, 0])

for file in os.listdir('data\\val\\1'):
    link = 'data\\val\\1\\'+file
    test_list.append([link, 1])

train_df = pd.DataFrame(train_list, columns=['name', 'label'])
test_df = pd.DataFrame(test_list, columns=['name', 'label'])


train_df.to_csv('train_data.csv')
test_df.to_csv('test_data.csv')