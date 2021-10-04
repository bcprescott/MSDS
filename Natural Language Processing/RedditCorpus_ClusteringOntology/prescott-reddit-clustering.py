#!/usr/bin/env python
# coding: utf-8

# In[160]:


import gc
import json
import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import sample
from matplotlib.pyplot import figure
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralBiclustering
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE


# # Importing Full Corpus and Sampling

# First I'll start by importing the full corpus from the utterance JSON Lines file saved from the last assignment.
# 
# Due to the size of the corpus and being 100 categories and ~240,000 documents, I chose to use 7 categories and 3000 documents as the 'full corpus' and 7 categories and 300 documents as the reduced corpus.

# In[2]:


# Load JSON lines file for the full corpus
corpus = []
with open('utterances.jl', encoding='utf8') as f:
    for line in f:
        corpus.append(json.loads(line))


# In[173]:


corpusDF['subreddit'].unique()


# In[174]:


corpusDF = pd.DataFrame(corpus) # Assign corpus to dataframe for ease of review 

reddits = ['Economics','nfl','programming','Games','AdviceAnimals','Fitness','funny','MovieDetails'] # Subreddits I want to include in my analysis 

fullDF= corpusDF[corpusDF['subreddit'].isin(reddits)].sample(n=3000, replace=False, random_state=11) # Treating as the full term matrix

reducedDF = fullDF.sample(n=300, replace=False, random_state=10).reset_index(drop=True) #Treating as the reduced matrix


# Now that I have my matrices I'm going to create lists with teh tags (used for Doc2Vec), document IDs, and subreddits for later use.

# In[175]:


tags = list() # Tagged documents used for Doc2Vec
ids = list() # Document IDs for later reference
subs = list() # Subreddit names
counter = 0 # Used as the tag ID for each document, just for ease
for row in fullDF['body']:
    tags.append(TaggedDocument(row,[counter]))
    counter += 1
for row in fullDF['id']:
    ids.append(row)
for row in fullDF['subreddit']:
    subs.append(row)


# # Creating Full Matrices

# Generating the word embedding vectors using the Doc2Vec network. I decided to stick with a vector size of 4, which was the vector size used in the last programming assignment. 

# ## Doc2Vec

# In[176]:


gc.collect()
model = Doc2Vec(tags, vector_size = 4, window=2, min_count=1, workers = 8)


# In[177]:


# Creating a dataframe from the generated vectors 
# Appending the document IDs and subreddits 
docDF = pd.DataFrame(model.dv.vectors)
docDF.insert(0, 'id', ids)
docDF['subreddit'] = subs
docDF.head()


# ## TF-IDF

# Recreating the TF-IDF matrix using the 'full' document matrix of 3000 documents.

# In[178]:


# separating out the body text, appending to a list and and joining back into strings for another list
individualDocs = list()
newDocs = list()
for row in fullDF['body']:
    individualDocs.append(row)
for doc in individualDocs:
    strings = ' '.join(doc)
    newDocs.append(strings)


# In[179]:


from sklearn.feature_extraction.text import TfidfVectorizer

gc.collect()
tfidfvectorizer = TfidfVectorizer(analyzer='word',stop_words= 'english')
tfidf_wm = tfidfvectorizer.fit_transform(newDocs)
tfidf_tokens = tfidfvectorizer.get_feature_names()
tfidfDF = pd.DataFrame(data = tfidf_wm.toarray(),columns = tfidf_tokens)
tfidfDF.head()


# # Creating Reduced Matrices

# Now that I have the full matrices created by both TF-IDF and Doc2Vec, I'll do the same and create the reduced matrices keeping the same vector size.

# ## Doc2Vec

# In[180]:


reducedtags = list()
reducedids = list()
reducedsubs = list()
counter = 0
for row in reducedDF['body']:
    reducedtags.append(TaggedDocument(row,[counter]))
    counter += 1
for row in reducedDF['id']:
    reducedids.append(row)
for row in reducedDF['subreddit']:
    reducedsubs.append(row)


# In[181]:


gc.collect()
model2 = Doc2Vec(reducedtags, vector_size = 4, window=2, min_count=1, workers = 8)
reduceddocDF = pd.DataFrame(model2.dv.vectors)
reduceddocDF.insert(0, 'id', reducedids)
reduceddocDF['subreddit'] = reducedsubs
reduceddocDF.head()


# ## TF-IDF

# In[182]:


reducedindividualDocs = list()
reducednewDocs = list()
for row in reducedDF['body']:
    reducedindividualDocs.append(row)
for doc in individualDocs:
    strings = ' '.join(doc)
    reducednewDocs.append(strings)


# In[183]:


gc.collect()
tfidfvectorizer = TfidfVectorizer(analyzer='word',stop_words= 'english')
tfidf_wm = tfidfvectorizer.fit_transform(reducednewDocs)
tfidf_tokens = tfidfvectorizer.get_feature_names()
reducedtfidfDF = pd.DataFrame(data = tfidf_wm.toarray(),columns = tfidf_tokens)
reducedtfidfDF.head()


# # Determining Optimal K-means Cluster Count 

# My next step is to determine the optimal cluster count. I'll be using both the silhouette score of each cluster, as well as the elbow method.
# 
# Based on the results, the most optimal cluster count is 8.

# In[184]:


# Removing ID and subreddit information
reducedDFvec = reduceddocDF.iloc[:,1:-1]


# In[185]:


inertia = [] # will hold inertia values
silscore = dict() # will hold silhouette scores and cluster count

# Loop to test different cluster counts
for k in range(2,16):
    km = KMeans(n_clusters=k, random_state = 50)
    pred = km.fit_predict(reducedDFvec)
    sil = silhouette_score(reducedDFvec,pred)
    silscore[k] = sil
    inertia.append([k,km.inertia_])
clusnum = max(score.items(), key=operator.itemgetter(1))[0] # gets the cluster number with the maximum value in the dictionary
silval = max(score.values()) #retrieves the max score
print("Based on the silhouette score of {}, the optimal number of clusters is {} having an inertia of {}".format(silval,clusnum,inertia[(clusnum-2)][1]))


# In[186]:


plt.figure(figsize=(12,6))
plt.plot(pd.DataFrame(sse)[0], pd.DataFrame(sse)[1], marker='o')
plt.title('Optimal Cluster Count - Elbow Method',  fontsize=18, fontweight='demi')
plt.ylabel('KMeans Inertia')
plt.xlabel('Number of Clusters')
plt.show()


# # K-means Clustering

# Performing K-means clustering using 8 clusters. I am also performing t-SNE on the reduced Doc2Vec dataframe, to reduce the dimensions from 4 to 2 for plotting.

# ## Clustering using original data

# In[251]:


# K-means clustering
kmeans = KMeans(
    init = 'random',
    n_clusters = 8, # determined by earlier tests
    n_init = 10,
    max_iter = 300,
    random_state = 42
)

labels = kmeans.fit_predict(reducedDFvec) # determining clusters

tsne = TSNE(n_components=2,random_state=0) # using t-SNE for dimensionality reduction
tsnePoints = tsne.fit_transform(reducedDFvec) # storing t-SNE values for plotting


# In[252]:


# creating ground truth labels to color points
gtlabels = reduceddocDF.copy()
unique_vals = gtlabels['subreddit'].unique()
gtlabels['subreddit'].replace(to_replace=unique_vals,
           value= list(range(len(unique_vals))),
           inplace=True)


# In[253]:


# ground truth plot
plt.rcParams["figure.figsize"] = [12,8]
plt.scatter(tsnePoints[:,0],tsnePoints[:,1],c=gtlabels['subreddit'])
plt.title('Document Points - Ground Truth by Subreddit', fontsize=18, fontweight='demi')
plt.show()


# When plotting the clusters we can see some distinguishable clusters, but with nearly every cluster having overlap. 

# In[254]:


plt.rcParams["figure.figsize"] = [12,8]
plt.scatter(tsnePoints[:,0],tsnePoints[:,1],c=labels)
plt.title('K-means Clusters - Before t-SNE', fontsize=18, fontweight='demi')
plt.show()


# ## Clustering using t-SNE data

# Clustering after t-SNE has shown to further distinguish clusters, with very little overlap compared to pre t-SNE. Performing t-SNE on the entire dataset beforehand also allows for clusters to be easily plotted.

# In[255]:


# K-means clustering
kmeans = KMeans(
    init = 'random',
    n_clusters = 8, # determined by earlier tests
    n_init = 10,
    max_iter = 300,
    random_state = 42
)

labels = kmeans.fit_predict(tsnePoints) # determining clusters
centroids = kmeans.cluster_centers_


# In[256]:


plt.rcParams["figure.figsize"] = [12,8]
plt.scatter(tsnePoints[:,0],tsnePoints[:,1],c=labels)
plt.scatter(centroids[:,0],centroids[:,1],marker='X',s=100,c='red')
plt.title('K-means Clusters - After t-SNE', fontsize=18, fontweight='demi')
plt.show()


# Creating a plot of subreddits by cluster.

# In[192]:


plt.rcParams["figure.figsize"] = [18,14]
fig, ax = plt.subplots()
sc = ax.scatter(tsnePoints[:,0], tsnePoints[:,1], c=labels)
ax.scatter(centroids[:,0],centroids[:,1],marker='X',s=100,c='red')
ax.set_title('Subreddits by Cluster', fontsize=18, fontweight='demi')
ax.legend(*sc.legend_elements(), title = 'Cluster', loc='upper right')
for i, txt in enumerate(reducedDF['subreddit']):
    ax.annotate(txt, (tsnePoints[:,0][i], tsnePoints[:,1][i]))

plt.show()


# In[193]:


kmeanDF = reduceddocDF.copy()
kmeanDF['labels'] = labels
kmeanDF.head(15)


# # Hierarchical Clustering

# Hierarchical clustering exhibited very similar outcomes to that of K-means. 

# ## Clustering using original data

# In[196]:


cluster = AgglomerativeClustering(n_clusters=8, affinity='euclidean')  
cluster.fit_predict(reducedDFvec)


# In[197]:


plt.figure(figsize=(10, 7))  
plt.scatter(tsnePoints[:,0], tsnePoints[:,1], c=cluster.labels_) 
plt.title('Hierarchical Clusters', fontsize=18, fontweight='demi')
plt.show()


# ## Clustering using t-SNE data

# In[198]:


tsne = TSNE(n_components=2,random_state=0)
tsnePoints = tsne.fit_transform(reducedDFvec)


# In[199]:


cluster = AgglomerativeClustering(n_clusters=8, affinity='euclidean')  
cluster.fit_predict(tsnePoints)


# In[200]:


plt.figure(figsize=(10, 7))  
plt.scatter(tsnePoints[:,0], tsnePoints[:,1], c=cluster.labels_) 
plt.title('Hierarchical Clusters', fontsize=18, fontweight='demi')
plt.show()


# In[201]:


hierDF = reduceddocDF.copy()
hierDF['labels'] = cluster.labels_
hierDF.head(15)


# # Topic Modeling

# The subreddits used in this dataset were: 'Economics','nfl','programming','Games','AdviceAnimals','Fitness','funny','MovieDetails'.
# 
# LDA was ran on the reduced TF-IDF dataset with 5 components. I generated the top 30 words to provide a broader range to potentially identify the subreddits from the topic modeling.

# In[220]:


lda = LatentDirichletAllocation(n_components=5,random_state=50)
lda.fit_transform(reducedtfidfDF)


# In[221]:


for index, topic in enumerate(lda.components_):
    print(f'Top 30 words for Topic #{index}')
    print([tfidfvectorizer.get_feature_names()[i] for i in topic.argsort()[-30:]])
    print('\n')


# # Spectral Biclustering

# Switching to spectral biclustering, it is even more distinguished between different clusters. However, my original Doc2Vec matrix only contained four vectors for each document, which limits the amount of clusters I can use to four. 

# In[224]:


docDFvec = docDF.iloc[:,1:-1]

 # reducing dimensions using t-SNE 
tsne = TSNE(n_components=2,random_state=0)
tsnePoints = tsne.fit_transform(docDFvec)


# In[225]:


# biclustering using four clusters, aligning to the number of vectors for each document
bicluster = SpectralBiclustering(n_clusters=4, random_state=0)
bicluster.fit(docDFvec)
bilabels = bicluster.row_labels_


# In[226]:


plt.figure(figsize=(12, 8))  
plt.scatter(tsnePoints[:,0], tsnePoints[:, 1], c=bilabels)
plt.title('Spectral Biclustering Clusters', fontsize=18, fontweight='demi')


# In[249]:


spectralDF = docDF.copy()
spectralDF['labels'] = bilabels
print(spectralDF.head(20))
print(spectralDF.groupby('subreddit').count())


# In[247]:


spectralDF

