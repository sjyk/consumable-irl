from gensim import corpora, models, similarities
from gensim.models import ldamodel
import nltk
import numpy as np
from collections import *
from decimal import *


def gen_R(gen_source):
	bag_o_words=get_bag_of_words(gen_source)
	delimiters=get_delimiters(gen_source)
	R_prime=OrderedDict()
    #R_prime['source']=gen_source
    # R_prime['delimiters']=delimiters
    # R_prime['bag_o_words']=bag_o_words
   #  for delimiter in delimiters:
   #  	R_prime[delimiter]=[]
   #      delimited_source=delimit_source(gen_source,delimiter)
   #      for i,word in enumerate(bag_o_words):
   #      	if word in delimited_source:
   #          	R_prime[delimiter][i]=1
			# else:
			# 	R_prime[delimiter][i]=0
	bag_o_words=get_bag_of_words(gen_source)
	delimitted=delimit_source(gen_source,"")
	for word in bag_o_words:
		R_prime[word]=[0]*len(delimitted)
	for i,word in enumerate(delimitted):
		if word in R_prime:
			R_prime[word][i]=1
	return R_prime,len(R_prime.keys())

def get_lda_dist(lda,data,dict_):
	# bag_o_words=OrderedDict()
	# for key in R_prime.keys():
	# 	bag_o_words[key]=sum(R_prime[key])
	# words=[[key]*bag_o_words[key] for key in bag_o_words.keys()]
	# dictionary=corpora.Dictionary(words)

	# corpus=[dictionary.doc2bow(text) for text in words]
	# lda=ldamodel.LdaModel(corpus,id2word=dictionary,num_topics=100)
	# # topiclist=lda.get_document_topics()
	# topiclist=lda[]
	topiclist=lda[dict_.doc2bow(data)]
	return topiclist
	


def gen_V_lda(R_prime,num_topics=10,trained_lda=None,dictionary=None):
	# delimiters=R_prime['delimiters']
	# bag_o_words=R_prime['bag_o_words']
	#source=R_prime['source']
	topics=[]
	topic_probability=[]
	bag_o_words=OrderedDict()
	for key in R_prime.keys():
		bag_o_words[key]=sum(R_prime[key])
	words=[[key]*bag_o_words[key] for key in bag_o_words.keys()]
	# dictionary=corpora.Dictionary(words)
	# corpus=[dictionary.doc2bow(text) for text in words]

	if trained_lda!=None:
		assert isinstance(trained_lda,ldamodel.LdaModel)
		assert isinstance(dictionary,corpora.Dictionary)
		corpus=[dictionary.doc2bow(text) for text in words]
		trained_lda.update(corpus)
		topics=lda.show_topics(num_topics=num_topics,num_words=len(dictionary),formatted=False)
	else:
		dictionary=corpora.Dictionary(words)
		corpus=[dictionary.doc2bow(text) for text in words]
		lda=ldamodel.LdaModel(corpus,id2word=dictionary,num_topics=100)
		topics=lda.show_topics(num_topics=num_topics,num_words=len(dictionary),formatted=False)
	V_prime=OrderedDict()
	for word in R_prime.keys():
		V_prime[word]=[0]*num_topics

	for i,topic in enumerate(topics):
		for entry in topic:
			V_prime[entry[1]][i]=entry[0]

	return V_prime
	# for topic_id,topic_prob in trained_lda.get_document_topics(source):
	# 	topics.append(topic_id)
	# 	topic_probability.append(topic_prob)

# def gen_V_lda_corpus(R_prime,num_topics=10,trained_lda=None):
	

def gen_V_keywords(R_prime,prune=True):
	words=R_prime.keys()
	tagged=nltk.pos_tag(words)
	pruned_words=[]
	for tag in tagged:
		if tag[1].find('NN')!=-1:
			pruned_words.append(tag[0])
	V_prime={}
	if not prune:
		for word in words:#not sure about this part ask sanjay if this won't statistically fuck our results
			V_prime[word]=0
	for word in pruned_words:
		V_prime[word]=sum(R_prime[word])
	return V_prime

def get_rank(dictb4mat):
	mat=dict2mat(dictb4mat)
	return np.linalg.matrix_rank(mat)

def get_shannon_entropy(dictb4mat):
	mat=dict2mat(dictb4mat)
	Hx=np.zeros(len(dictb4mat.keys())).reshape((len(dictb4mat.keys()),1))
	for i,row in enumerate(mat):
		summed=0
		normalized=row/row.sum()
		for y in normalized.nonzero()[0]:
			summed=normalized[y]*float(Decimal(normalized[y]).ln())
		summed=(0-summed)
		Hx[i]=summed
	return Hx

def get_von_neumann_entropy(dictb4mat):
	mat=dict2mat(dictb4mat)
	assert mat.shape[0]==mat.shape[1]
	eigenvalues=np.linalg.eig(mat)[0]
	summed=0
	normalized=eigenvalues/eigenvalues.sum()
	for y in normalized.nonzero()[0]:
		summed=normalized[y]*float(Decimal(normalized[y]).ln())
	return summed

def get_bag_of_words(source):
	return delimit_source(source,"")

def get_bag_of_actions(source):
	"""TODO still"""
	pass

def get_delimiters(source):
	pass


def dict2mat(dict_):
	r=len(dict_.keys())
	c=len(dict_[dict_.keys()[0]])
	toreturn=np.zeros((r,c))
	for r,key in enumerate(dict_.keys()):
		for c,item in enumerate(dict_[key]):
			toreturn[r,c]=item

	return toreturn

def delimit_source(source,delimiter):
	source=source.replace("."," ")
	source=source.replace(","," ")
	source=source.replace(";"," ")
	# source=source.replace(":"," ")
	source=source.replace('"'," ")
	source=source.replace(")"," ")
	source=source.replace("("," ")
	source=source.replace("{"," ")
	source=source.replace("}"," ")
	# source=source.replace("<"," ")
	# source=source.replace(">"," ")
	source=source.replace("?"," ")
	source=source.replace("\\"," ")
	# source=source.replace("/"," ")
	source=source.replace("|"," ")
	source=source.replace("\"",'')
	return source.split()

