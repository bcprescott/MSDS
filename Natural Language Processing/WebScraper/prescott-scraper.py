#!/usr/bin/env python
# coding: utf-8

from urllib.request import Request, urlopen
import csv
import re
import jsonlines
import json
from bs4 import BeautifulSoup, SoupStrainer
import httplib2
import requests
import pandas as  pd
import urllib
import nltk


data = 'business.csv'


my_headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36 Edg/89.0.774.76", 
          "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9"}

links=[]

with open (data) as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        links.append(row[0])


stop = nltk.corpus.stopwords.words('english')
df = pd.DataFrame(columns=['id','url','title','text'])
counter = 0
for site in links:
    page = requests.get(site, headers=my_headers)
    soup = BeautifulSoup(page.text, 'html.parser')
    sections = soup.find_all('article')
    ptext = []
    for section in sections:
        paragraphs = section.find_all('p')
        for paragraph in paragraphs:
            ptext.append(paragraph.text)
    df = df.append({'id':counter, 'url': page.url, 'title':soup.find('h1').text, 'text':' '.join(ptext)}, ignore_index = True)
    df['text'] = df['text'].apply(lambda x: ' '.join([word.lower() for word in x.split() if word not in (stop)]))
    counter = counter + 1

dfdict = df.to_dict('records')


with jsonlines.open('remotework.jl','w') as writer:
    writer.write_all(dfdict)

