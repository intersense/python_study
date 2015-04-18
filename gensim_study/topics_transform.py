#! /usr/bin/env python2.7
#coding=utf-8
import logging
import os
from gensim import corpora, models, similarities

logging.basicConfig(filename = os.path.join(os.getcwd(), 'log_topic_transform.txt'),format='%(asctime)s : %(levelname)s : %(message)s', level = logging.DEBUG)  

dictionary = corpora.Dictionary.load('deerwester.dict')
corpus = corpora.MmCorpus('deerwester.mm')
logging.info(corpus)

tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model

doc_bow = [(0,1), (1,1)]
print(tfidf[doc_bow]) # step 2 -- use the model to transform vectors

corpus_tfidf = tfidf[corpus]
for doc in corpus_tfidf:
	print(doc)

lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2) # initialize an LSI transformation
corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus: bow -> tfidf -> fold-in-lsi
lsi.print_topics(2)