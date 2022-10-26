import os
from socket import VM_SOCKETS_INVALID_VERSION
from unicodedata import numeric
import pandas as pd
import numpy as np 
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, Trainer
os.chdir("C:\\Users\\Hithesh\\Desktop\\Datawise\\ML\\ML_netflix")
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import re
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.dummy import DummyRegressor

df = pd.read_csv("n_movies.csv")
df.head()
df = df.drop(['year', 'certificate', 'duration',  'stars'], axis=1)
df.columns

np.min(df["rating"]) ## 1.7
np.max(df["rating"]) ## 9.9

df["tiers"] = np.ceil(df["rating"])
df = df.dropna()

len(df)
df.head()


# Create class for data preparation
class SimpleDataset:
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts
    
    def __len__(self):
        return len(self.tokenized_texts["input_ids"])
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tokenized_texts.items()}
    
model_name = "j-hartmann/emotion-english-distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
trainer = Trainer(model=model)

#classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True) ##hugging-hub

pred_texts = df["description"].astype('str').tolist()
tokenized_texts = tokenizer(pred_texts,truncation=True,padding=True)
pred_dataset = SimpleDataset(tokenized_texts)
predictions = trainer.predict(pred_dataset)

preds = predictions.predictions.argmax(-1)
labels = pd.Series(preds).map(model.config.id2label)
scores = (np.exp(predictions[0])/np.exp(predictions[0]).sum(-1,keepdims=True)).max(1)

# scores raw
temp = (np.exp(predictions[0])/np.exp(predictions[0]).sum(-1,keepdims=True))

# work in progress
# container
anger = []
disgust = []
fear = []
joy = []
neutral = []
sadness = []
surprise = []

# extract scores (as many entries as exist in pred_texts)
for i in range(len(pred_texts)):
  anger.append(temp[i][0])
  disgust.append(temp[i][1])
  fear.append(temp[i][2])
  joy.append(temp[i][3])
  neutral.append(temp[i][4])
  sadness.append(temp[i][5])
  surprise.append(temp[i][6])
  

df = df.assign(**{'anger':anger, 'disgust':disgust, 'fear':fear, 'joy':joy, 'neutral':neutral, 
             'sadness':sadness, 'surprise':surprise})

new_file = "n_movies_emomods.csv"
df.to_csv(new_file)

df1 = pd.read_csv(new_file)
df1['votes'] = df1['votes'].str.replace(',', '').astype(float)
df1 = df1.drop([df1.columns[0],'year', 'certificate', 'duration',  'stars'], axis=1)
df1 = df1.dropna()
df1.columns
df1.to_csv(new_file)


df1 = pd.read_csv(new_file)
df1 = pd.read_csv(new_file).drop(([df1.columns[0]]), axis = 1)

## Creating Genre Indicators

len(df1)
test = []
for i in np.arange(len(df1)):
    test.append(df1['genre'].replace(',','', regex=True)[i].split( ))
    
df1['genre_list'] = test[1:]

pd.get_dummies(df1['genre_list']apply(pd.Series).stack()).sum(level=0)


genres = df1['genre_list']
genre_indicators = pd.get_dummies(genres.apply(pd.Series).stack()).sum(level=0)

final_df = df1.join(genre_indicators)

col_list = list(final_df.columns)

## Finding most common genres 
top5_genres = final_df[col_list[14:]].sum(axis=0).sort_values(ascending = False)[:5]
all_genres = col_list[14:]
top5_genres = top5_genres.index.tolist()
emo_list = ['neutral', 'joy', 'anger', 'fear', 'disgust', 'sadness', 'surprise']
regressors = all_genres + emo_list[0:] + ['votes']
final_df.to_csv('final_df.csv')

final_df[['Action', 'Drama']]

#### Train Test Split
y = final_df['rating']
X = final_df[regressors]
#X = final_df.drop(['title', 'genre', 'description', 'rating', 'tiers', 'genre_list'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=99)

### Regression Baseline
# create a dummy regressor
dummy_reg = DummyRegressor(strategy='mean')
# fit it on the training set
dummy_reg.fit(X_train, y_train)
# make predictions on the test set
y_pred = dummy_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Dummy RMSE:", rmse)

#### Regression 
model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print('mean_squared_error : ', mean_squared_error(y_test, predictions))
print('RMSE: ', np.sqrt(mean_squared_error(y_test, predictions)))
print('mean_absolute_error : ', mean_absolute_error(y_test, predictions))
print('Coefficients: ', model.coef_)

print('Variance score: {}'.format(model.score(X_test, y_test)))
  
# plot for residual error
  
## setting plot style
plt.style.use('fivethirtyeight')
  
## plotting residual errors in training data
plt.scatter(model.predict(X_train), model.predict(X_train) - y_train,
            color = "green", s = 10, label = 'Train data')
  
## plotting residual errors in test data
plt.scatter(model.predict(X_test), model.predict(X_test) - y_test,
            color = "blue", s = 10, label = 'Test data')
  
## plotting line for zero residual error
plt.hlines(y = 0, xmin = 0, xmax = 10, linewidth = 2)
  
## plotting legend
plt.legend(loc = 'upper right')
  
## plot title
plt.title("Residual errors")
  
## method call for showing the plot
plt.show()



