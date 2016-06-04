import numpy as np
from operator import itemgetter
from math import *
from R_to_V import get_bag_of_words


def partition(data):
	pass

def map_blocks(blocks):
	covar_mat={}
	for block in blocks:
		covar_mat[block]={}
		for other_block in blocks:
			covar_mat[block][other_block]=calc_covariance(block,other_block)
	return covar_mat

def map_blocks_similarity(blocks,word_list):
	covar_mat={}
	for block in blocks:
		covar_mat[block.item]={}
		for other_block in blocks:
			if other_block==block:
				continue
			# print 'hi'
			covar_mat[block.item][other_block.item]=calc_word_similarity(block,other_block,word_list)
	# print covar_mat
	return covar_mat

def filter_edges(edge_dict,threshold=lambda x:x<=0):
	pruned_covar_mat={}
	for key in edge_dict.keys():
		pruned_covar_mat[key]={}
		for other_key in edge_dict[key].keys():
			edge_weight=edge_dict[key][other_key]
			# if (other_key==key) or (edge_weight<0) or (t_test(edge_weight)>threshold):
			if (other_key==key) or threshold(edge_weight):
				continue
			pruned_covar_mat[key][other_key]=edge_weight
	return pruned_covar_mat


def reduce_blocks(passed_in):
	for key in passed_in.keys():
		total=0
		for edge in passed_in[key].keys():
			total+=passed_in[key][edge]
		passed_in[key]['cumulative weight']=total
	return passed_in

def map_edges(edges,blocks):
	normalized_edges={}
	key_list={}
	for i,block in enumerate(blocks):
		key_list[block]=i
	W=np.zeros((len(blocks),len(blocks)))
	for key in edges:
		normalized_edges[key]={}
		to_normalize=edges[key]["cumulative weight"]
		for edge in edges[key]:
			W[key_list[key],key_list[edge]]=edges[key][edge]
			normalized_edges[key][edge]=(edges[key][edge]/to_normalize)

	return normalized_edges,W

def map_edges_tree(edges,blocks):
	normalized_edges={}
	key_list={}
	blocks=map(lambda x:x.item,blocks)
	for i,block in enumerate(blocks):
		key_list[block]=i
	W=np.zeros((len(blocks),len(blocks)))
	count=0
	for key in edges.keys():
		if key not in blocks:
			count+=1
	for key in edges.keys():
		normalized_edges[key]={}
		#to_normalize=edges[key]["cumulative weight"]
		for edge in edges[key].keys():
			# print key_list[key]
			# print key_list[edge]
			W[key_list[key],key_list[edge]]=edges[key][edge]
		#	normalized_edges[key][edge]=(edges[key][edge]/to_normalize)

	return normalized_edges,W

def get_laplacian(W):
	D=np.zeros((W.shape[0],W.shape[1]))
	for i,row in enumerate(W):
		D[i,i]=row.sum()
	return (D-W)
	
def calc_eigen(W,k):
	val,vec=np.linalg.eig(W)
	sorted_val=sorted(enumerate(val),key=itemgetter(1),reverse=True)
	sorted_vec=np.zeros((vec.shape[0],vec.shape[1]))
	x=0
	for i,val in sorted_val:
		sorted_vec[:,x]=vec[:,i]
		x+=1
	return sorted_vec[:,1:1+k]

def calc_covariance(original,other):
	org_avg=get_average(original)
	other_avg=get_average(other)
	org_s2=get_s2(original,org_avg)
	other_s2=get_s2(other,other_avg)
	org_=sqrt(org_s2)
	other_=sqrt(other_s2)
	cov=0
	for i,x in enumerate(original):
		cov+=(((ord(x)-org_avg)/org_)*((ord(other[i])-other_avg)/other_))
	return (cov/len(original)-1)

def calc_word_similarity(original,other,word_list):
	word_dict={}
	for word in word_list:
		word_dict[word]=0
	# print word_dic
	for word in get_bag_of_words(original.item):
		word_dict[word]+=1
	for word in get_bag_of_words(other.item):
		word_dict[word]+=1
	return len(filter(lambda x:word_dict[x]==2,word_dict))

def get_average(data):
	summed=0
	for x in data:
		summed+=ord(x)
	return summed/len(data)

def get_s2(data,avg):
	s2=0
	for x in data:
		(s2)+=(ord(x)-avg)**2
	return s2/(len(data)-1)