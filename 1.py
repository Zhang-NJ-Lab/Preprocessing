# -*- coding:utf-8 -*-
import logging
import gensim
from gensim.models import word2vec, KeyedVectors
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from chemdataextractor import Document
import pkuseg
import numpy as np
from nltk.tokenize import MWETokenizer
'''stopwords = [line.strip() for line in open('stop_words.txt', encoding='UTF-8').readlines()] #加载自定义停止词

sentence=str()


with open('Sciences.txt', encoding='utf-8') as f: #加载原始数据库并分词
    document = f.read()
    #document_cut = jieba.cut(document)
    tokenizer = MWETokenizer([('solar', 'cell')], separator = '_')
    #seg = pkuseg.pkuseg(user_dict = "userdict.txt")
    text=tokenizer.tokenize(nltk.word_tokenize(document))
    result = ' '.join(text)
    for word in result:
        if word not in stopwords:
            if word != "\t":
                sentence += word

    with open('result.txt', 'w',encoding="utf-8") as f2:
        f2.write(sentence)
'''
'''
sentences = word2vec.LineSentence('result.txt') #正式训练前的格式化

model = word2vec.Word2Vec(sentences,sg=1, hs=0,min_count=1,window=10,vector_size=100)
model.wv.save_word2vec_format("word2vec0607.txt")
'''
model = KeyedVectors.load_word2vec_format('word2vec0607.txt')

#for key in model.similar_by_word('photovoltaic',topn=3000):
    #print(key[0])


with open('0607.txt','r') as f:
    for line in f.readlines():

        doc = Document(line)

        if len(doc.cems) > 0:
            print(doc.cems[0])
            try:
              sm = model.similarity(str(doc.cems[0]), 'photovoltaic')
            except:
               continue
            print(sm)


    print(doc.cems)



'''
f = open('0607.txt', 'rb')
doc = Document.from_file(f)
print(doc.cems)
'''




#中文代码
'''stopwords = [line.strip() for line in open('stop_words.txt', encoding='UTF-8').readlines()] #加载自定义停止词

sentence=str()


with open('Chinese.txt', encoding='utf-8') as f: #加载原始数据库并分词
    document = f.read()
    #document_cut = jieba.cut(document)
    #tokenizer = MWETokenizer([('solar', 'cell')], separator = '_')
    seg = pkuseg.pkuseg(user_dict = "userdict.txt")
    text=seg.cut(document)
    result = ' '.join(text)
    for word in result:
        if word not in stopwords:
            if word != "\t":
                sentence += word

    with open('result1.txt', 'w',encoding="utf-8") as f2:
       f2.write(sentence)
'''
'''
sentences = word2vec.LineSentence('result1.txt') #正式训练前的格式化

model = word2vec.Word2Vec(sentences,sg=1, hs=0,min_count=1,window=10,vector_size=100)
model.wv.save_word2vec_format("word2vec0608.txt")
'''
'''
model = KeyedVectors.load_word2vec_format('word2vec0608.txt')

for key in model.similar_by_word('光伏',topn=3000):
    print(key[0])
'''
'''
doc = Document('砷化镓 光伏 太阳能电池')
print(doc.cems)
'''

