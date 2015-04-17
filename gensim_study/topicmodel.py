#! /usr/bin/env python2.7
#coding=utf-8

import logging
import os
import string
from gensim import corpora, models, similarities
import jieba

#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.basicConfig(filename = os.path.join(os.getcwd(), 'log.txt'),format='%(asctime)s : %(levelname)s : %(message)s', level = logging.DEBUG)  

# English Text
texts = [line.lower().split() for line in open('mycorpus.txt')]

# Chinese Text 
#texts = [jieba.cut(line) for line in open('mycorpus.txt')]


# remove common words and tokenize
stoplist = set('for a an are i of the and to in on with that this by is using'.split())
#remove the punctuation by string.replace(string.punctuation,"") 
text1 = [[word.replace(string.punctuation,"") for word in wordlist ]for wordlist in texts]
'''for text in texts:
	for token in text:'''

logging.info(text1)
#streaming
dictionary = corpora.Dictionary(texts)
dictionary.save('deerwester.dict')
# remove stop words and words that appear only once
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist
			if stopword in dictionary.token2id]
once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq ==1]
dictionary.filter_tokens(stop_ids + once_ids) #remove stop words and words that appear only once
dictionary.compactify() # remove gaps in id sequence after words that were removed

#remove stop words and words that appear only once

logging.info(dictionary)
print dictionary
logging.info(dictionary.token2id)

#convert tokenized documents to vectors
new_doc = "Human computer interface"
new_vec = dictionary.doc2bow(new_doc.lower().split())
logging.info(new_vec)

#
corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('deerwester.mm', corpus)
logging.info(corpus)


#Streaming
class MyCorpus(object):
	def __iter__(self):
		for line in open('mycorpus.txt'):
			yield dictionary.doc2bow(line.lower().split())
corpus_memory_friendly = MyCorpus() # doesn't load the corpus into memory
for vector in corpus_memory_friendly:
	print (vector)

print ("编码")