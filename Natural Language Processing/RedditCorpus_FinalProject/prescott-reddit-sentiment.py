#!/usr/bin/env python
# coding: utf-8

# In[1336]:


import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Model, optimizers, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (Input, Embedding, Bidirectional, LSTM, 
                                     Dense, RepeatVector, Flatten, BatchNormalization, 
                                     LeakyReLU, Dropout)
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import seaborn as sns
nltk.download('vader_lexicon')


# ## Importing Full Corpus

# In[4]:


# Load JSON lines file for the full corpus
corpus = []
with open('redditjson.jl', encoding='utf8') as f:
    for line in f:
        corpus.append(json.loads(line))


# In[1255]:


corpusDF.subreddit.unique()


# ## Select Only Games Subreddit

# In[1350]:


corpusDF = pd.DataFrame(corpus) # Assign corpus to dataframe for ease of review 

reddits = ['Games'] # Subreddits I want to include in my analysis 

reducedDF = corpusDF[corpusDF['subreddit'].isin(reddits)] # Creating a new dataframe consisting of only Games subreddit


# In[1351]:


# Removing empty, deleted, removed, and automoderate utterances
reducedDF = reducedDF[~(reducedDF['text'] == '') & 
                      ~(reducedDF['text'] == '[deleted]') & 
                      ~(reducedDF['text'] == '[removed]') &
                      ~(reducedDF['speaker'] == 'AutoModerator')
                     ]


# ## VADER

# In[1352]:


# Loading the VADER model 
analyzer = SentimentIntensityAnalyzer()


# In[1353]:


# Creating a copy of the Games dataframe and assigning a new column all values of zero
reducedScored = reducedDF.copy()
reducedScored['sentiment'] = 0
comments = reducedScored.text.tolist()


# In[1354]:


len(comments)


# In[1355]:


# Performing sentiment analysis on each utterance and updating the 'sentiment' column score
reducedScored = reducedDF.copy()
reducedScored.reset_index(inplace=True)
reducedScored['sentiment'] = 0
comments = reducedScored.text.tolist()
count = 0
for comment in comments:
    score = analyzer.polarity_scores(comment)
    if score['compound'] > 0.05:
        reducedScored.at[count, 'sentiment'] = 1
    else:
        reducedScored.at[count, 'sentiment'] = 0
    count += 1


# In[1357]:


# Visualizing the sentiment score counts
plt.hist(reducedScored.sentiment)
plt.title('Games Subreddit Sentiment Score Counts')
plt.show()


# ## Tokenization and Padding

# In[1316]:


# Converting the utterances and their VADER sentiment labels to lists
sentences = reducedScored.text.tolist()
labels = np.array(reducedScored.sentiment.tolist())


# In[1317]:


# Reviewing the max sequence lengths to use for variables later
# Some look to be outliers - possibly long URLs. Majority seem to be under 55 words long.
seq_lengths = reducedScored.text.apply(lambda x: len(x.split(' ')))
seq_lengths.describe()


# In[1318]:


# Selecting some values for tokenization and word embeddings.
num_words = 300
max_len = 200
embed_dim = 10


# In[1319]:


# Tokenizing and zero-padding the sequences
tokenizer = Tokenizer(num_words = num_words,
                      split=' ')
tokenizer.fit_on_texts(sentences)
seqs = tokenizer.texts_to_sequences(sentences)
pad_seqs = pad_sequences(seqs, max_len)


# In[1320]:


len(pad_seqs[1])


# ## Word Embeddings Using AutoEncoder

# In[1321]:


# Tripartite splitting of the datasets
X, y = pad_seqs, labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)


# In[1322]:


print('X_train:',X_train.shape,'\n','X_test:',X_test.shape,'\n','X_val:',X_val.shape)


# In[1323]:


# define encoder/decoder
visible = Input(shape=(max_len,))
# encoder layer 1
encoder = Dense(max_len*2)(visible)
encoder = BatchNormalization()(encoder)
encoder = LeakyReLU()(encoder)
# encoder layer 2
encoder = Dense(max_len)(encoder)
encoder = BatchNormalization()(encoder)
encoder = LeakyReLU()(encoder)
# reduction
n_bottleneck = max_len
bottleneck = Dense(n_bottleneck)(encoder)

# decoder layer 1
decoder = Dense(max_len)(bottleneck)
decoder = BatchNormalization()(decoder)
decoder = LeakyReLU()(decoder)
# decoder layer 2
decoder = Dense(max_len*2)(decoder)
decoder = BatchNormalization()(decoder)
decoder = LeakyReLU()(decoder)
# output layer
output = Dense(max_len, activation='linear')(decoder)

gc.collect()

escallback = EarlyStopping(monitor='val_loss', patience=3)

model = Model(inputs=visible, outputs=output)
model.summary()
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, 
          X_train, 
          epochs=100, 
          validation_data=(X_val,X_val), 
          callbacks=[escallback])

encoder = Model(inputs=visible, outputs=bottleneck)
encoder.save('encoder.h5')


# In[1324]:


# Predicting the word embeddings for the X value train/test/validation data
X_train_encode = encoder.predict(X_train)
X_test_encode = encoder.predict(X_test)
X_val_encode = encoder.predict(X_val)


# ## Classification Network

# In[1346]:


# Model used to predict the sentiment scores using the learned word embeddings
model = Sequential()
model.add(Dense(128, input_dim=max_len, activation='relu'))
model.add(Dense(12, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
escallback = EarlyStopping(monitor='val_loss', patience=5)

gc.collect()
model.fit(X_train_encode, 
          y_train,
          validation_data = (X_val_encode,y_val),
          callbacks=[escallback],
          epochs=50)


# ## Sentiment Predictions & Performance

# In[1347]:


# Predicting sentiment scores for the X_Test word embeddings
pred = (model.predict(X_test_encode) > 0.5).astype("int32")
predDF = pd.DataFrame({'actual':y_test.tolist(),'predicted':[item for sublist in pred for item in sublist]})
predDF


# ### ROC Curve

# In[1348]:


# Reviewing ROC AUC score

fpr, tpr, threshold = roc_curve(y_test, pred)
roc_auc = auc(fpr,tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# ### Confusion Matrix

# In[1349]:


# Reviewing the confusion matrix of correctly and incorrectly classified data

cm = confusion_matrix(predDF.actual,predDF.predicted)
plt.figure(figsize=(10,8))
sns.set(font_scale=1.4)
sns.heatmap(cm, annot=True,cmap='Blues',fmt='d')
plt.title('Test Data Sentiment Confusion Matrix', fontsize=20)
plt.show()

