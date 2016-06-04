import ete2
import math
from R_to_V import *
import numpy as np
from itertools import product
from clustering_funcs import get_lowest_ed_pair,cluster_segmentation_data_affinity
import pickle as pkl

def getnextchars(curr_char):
	"""used to assign names to each node"""
	last_char_val=curr_char[-1]
	if last_char_val=='Z':
		i=len(curr_char)-1
		carry=True
		temp=''
		while i>=0:
			ch=curr_char[i]
			if carry:
				if ch=='Z':
					if i==0:
						temp='AA'+temp
					else:
						temp='A'+temp
				else:
					carry=False
					temp=chr(ord(ch)+1)+temp
			else:
				temp=ch+temp
			i-=1
		return temp

	else:
		return curr_char[:-1]+chr(ord(last_char_val)+1)	

def get_tree(data,clustering_func,intersect_meth=lambda x,y:intersection(x,y),threshold=lambda x:x<=0,cutoff=lambda x:len(x)<=1,num_groups=lambda groups:int(math.ceil(len(groups)/4.0)),join=False,names=None):
	"""returns the hierarchical clustering tree"""
	curr_trees=[]
	return_dict={}
	level_dict={}
	curr_char=chr(ord('A')-1)
	for i,datum in enumerate(data):
		
		if names:
			temp=names[i]
			# print temp
			curr_char=getnextchars(curr_char)
			return_dict[curr_char]=[temp]+datum
			curr_trees.append(Tree(datum,node_name=curr_char))
			
		else:
			curr_char=getnextchars(curr_char)
			return_dict[curr_char]=datum
			curr_trees.append(Tree(datum,node_name=curr_char))


	level=0
	while True :#set stop depth with function
		print 'cluster_lvl: '+str(level)
		clustered=perform_clustering(curr_trees,clustering_func,threshold,num_groups)
		print 'done clustering, starting merging'
		curr_trees=[]
		#print clustered[0]
		removed_words=[]
		level_dict[level]={}
		no_root=True
		i=0
		clusterlen_list=[]
		for group in clustered:
			print 'cluster id:',len(group),'size:'i
			i+=1
			clusterlen_list.append((len(group),i))
			if len(group)==1:
				curr_char=getnextchars(curr_char)
				curr_trees.append(Tree(group[0].item,subtree=group,node_name=getnextchars(curr_char)))
				continue
			no_root=False
			subtrees=[]
			for doc in group:			
				subtrees.append(doc)
			intersect=intersect_meth(group,len(clustered))
			if intersect==None:
				continue
			lost=intersect[1]
			intersect=intersect[0]
			curr_char=getnextchars(curr_char)
			level_dict[level][curr_char]=lost
			return_dict[curr_char]=intersect
			if join:
				curr_trees.append(Tree(' '.join(intersect),subtree=subtrees,node_name=curr_char,lost_vals=lost))
			else:
				curr_trees.append(Tree(intersect,subtree=subtrees,node_name=curr_char,lost_vals=lost))
		
		if no_root or cutoff(curr_trees):

			if len(curr_trees)==1:
				break
			curr_char=getnextchars(curr_char)
			intersect=intersect_meth(curr_trees,len(curr_trees))
			lost=intersect[1]
			intersect=intersect[0]
			level_dict[level]['0_unrooted']=lost
			return_dict['0_unrooted']=intersect
			if join:
				curr_trees=[(Tree(' '.join(intersect),subtree=curr_trees,node_name='0_unrooted',lost_vals=lost))]
			else:
				curr_trees=[(Tree(intersect,subtree=curr_trees,node_name='0_unrooted',lost_vals=lost))]
			break
		# pkl.dump(clusterlen_list,open('cbst2_'+str(level)+'.pkl','wb'))
		level+=1
	return curr_trees[0],return_dict,level_dict


def perform_clustering(groups,clustering_func,threshold=lambda x:x<=0,num_groups=lambda groups:int(math.ceil(len(groups)/4.0))):
	return clustering_func(groups,threshold,num_groups=num_groups)


def intersection(group,cluster_size):
	temp_dict={}
	# if len(group)==1 and cluster_size!=1:
	# 	print 'hi'+'\n'*50
	# 	return None
	for item in group:
		if isinstance(item,Tree):
			for word in get_bag_of_words(item.item):
				if word not in temp_dict:
					temp_dict[word]=0
				temp_dict[word]+=1
		else:
			for word in get_bag_of_words(item):
				if word not in temp_dict:
					temp_dict[word]=0
				temp_dict[word]+=1
	group_size=len(group)
	intersection_words=[item for item in temp_dict if temp_dict[item]>=group_size]
	lost_words=[item for item in temp_dict if temp_dict[item]<group_size]
	# return apply_update(group,intersection_words)
	return intersection_words,lost_words

def robo_data_intersection(group,cluster_size):
	temp_dict={}
	for item in group:
		if isinstance(item,Tree):
			for action in get_bag_of_actions(item.item):
				if action not in temp_dict:
					temp_dict[action]=0
				temp_dict[action]+=1
		else:
			for action in get_bag_of_actions(item):
				if action not in temp_dict:
					temp_dict[action]=0
				temp_dict[action]+=1
	group_size=len(group)
	intersection_words=[item for item in temp_dict if temp_dict[item]>=group_size]
	lost_words=[item for item in temp_dict if temp_dict[item]<group_size]
	# return apply_update(group,intersection_words)
	return intersection_words,lost_words

def LCS_binary_intersect(group,cluster_size):
	cluster_size=len(group)
	if cluster_size==1:
		return group[0].item,[]
	elif cluster_size==2:
		seq,lost=run_LCS(group)
		return seq,lost
	else:
		return None,None

def LCS_intersect(group,cluster_size):
	cluster_size=len(group)
	if cluster_size==1:
		return group[0].item,[]
	elif cluster_size==0:
		return None,None
	else:
		seq,lost=run_greedy_LCS(group)
		return seq,lost

def run_greedy_LCS(group):	
	tempgroup=list(group)
	while len(tempgroup)>1:
		pair=get_lowest_ed_pair(tempgroup)
		tempgroup.remove(pair[0])
		tempgroup.remove(pair[1])
		seq=run_LCS(pair)[0]
		tempgroup.append(Tree(seq))

	return tempgroup[0].item,None


def run_LCS(group):
	mat=np.zeros(tuple(map(lambda x:len(x.item)+1,group)))
	arrs=map(lambda x:xrange(len(x.item)+1),group)
	the_gen=product(*arrs)
	group_len=len(group)
	origin=[0]*group_len
	val_list=[]
	set_mat(mat,origin,[Null_string()],val_list)
	i=0
	for index in the_gen:
		if i==0:
			i+=1
			continue
		equal=True
		eq_val=None
		max_=float('-inf')
		maxval=None
		for x,i in enumerate(index):
			if i==0:
				equal=False
				continue
			if equal:
				if eq_val==None:
					if isinstance(eq_val,segment):
						eq_val=eq_val.merge(group[x].item[i-1])
					else:
						eq_val=group[x].item[i-1]
				elif eq_val!=group[x].item[i-1]:
					equal=False
		#check splicing
			if x!=group_len-1:
				sp=index[:x]+tuple([i-1])+index[x+1:]	
			else:
				sp=index[:x]+tuple([i-1])
			val=get_mat(mat,sp,val_list)
			if maxval==None:
				maxval=val
			elif len(maxval)<len(val):
				maxval=val
		if equal:
			sp=map(lambda x:x-1,index)
			val=get_mat(mat,sp,val_list)
			if maxval==None:
				maxval
			elif len(maxval)<=len(val)+1:
				maxval=val+[eq_val]

		set_mat(mat,index,maxval,val_list)
		i+=1
	
	return get_mat(mat,map(lambda x:len(x.item),group),val_list)[1:],None

def run_LCS_length(group):
	mat=np.zeros(tuple(map(lambda x:len(x.item)+1,group)))
	arrs=map(lambda x:xrange(len(x.item)+1),group)
	the_gen=product(*arrs)
	group_len=len(group)
	origin=[0]*group_len
	
	set_mat(mat,origin,0)
	i=0
	for index in the_gen:
		if i==0:
			i+=1
			continue
		equal=True
		eq_val=None
		max_=float('-inf')
		maxval=None
		for x,i in enumerate(index):
			if i==0:
				equal=False
				continue
			if equal:
				if eq_val==None:
					eq_val=group[x].item[i-1]
				elif eq_val!=group[x].item[i-1]:
					equal=False
		#check splicing
			if x!=group_len-1:
				sp=index[:x]+tuple([i-1])+index[x+1:]	
			else:
				sp=index[:x]+tuple([i-1])
			val=get_mat(mat,sp)
			maxval=max(maxval,val)
		if equal:
			sp=map(lambda x:x-1,index)
			val=get_mat(mat,sp)
			maxval=max(maxval,val+1)

		set_mat(mat,index,maxval)
		i+=1
	
	return LCS_backtrack(mat,group),None


def get_mat(mat,index,val_list=None):
	tv=mat

	for i in index:
		tv=tv[i]
	if val_list==None:
		return tv
	return val_list[int(tv)]

def set_mat(mat,index,val,val_list=None):
	tv=mat
	for x,i in enumerate(index):
		if x==len(index)-1:
			if val_list==None:
				tv[i]=val
			else:
				val_list.append(val)
				tv[i]=len(val_list)-1
		else:
			tv=tv[i]

def LCS_backtrack(mat,group):
	index=map(lambda x:len(x.item),group)
	length=get_mat(mat,index)
	rtn=[]
	while 0 not in index:
		equal=True
		eq_val=None
		for x,i in enumerate(index):
			if eq_val==None:
				eq_val=group[x].item[i-1]
			elif eq_val!=group[x].item[i-1]:
				equal=False
				break
		if equal:
			rtn=[eq_val]+rtn
			index=map(lambda x:x-1,index)
		else:
			val_=0
			mv=None
			w=0
			while w<len(index):
				if w==len(index)-1:
					sp=index[:w]+[index[w]-1]
				else:
					sp=index[:w]+[index[w]-1]+index[w+1:]
				tv=get_mat(mat,sp)
				if tv>val_:
					val=tv
					mv=sp
			index=mv

	return rtn


class Null_string:
	def __init__(self):
		pass
	def __eq__(self,other):
		if isinstance(other,Null_string):
			return True
		return False

	def __str__(self):
		return '_'
	def __repr__(self):
		return '_'

def apply_update(group,words):
	new_group=[]
	for item in group:
		new_group.append(item.update(words))
	return new_group

def make_newick(treeobject):
	if treeobject.subtrees==None:
		i=0
		duplicate=str(treeobject.node_name)
		# while i<len(duplicate):
		# 	if duplicate[i]==')':
		# 		duplicate.insert(i,'\\')
		# 		i+=1
		# 	elif dublicate[i]=='(':
		# 		dublicate.insert(i,'\\')
		# 		i+=1
		# 	i+=1
		return duplicate
	tempstring=''
	first=True
	for treeobj in treeobject.subtrees:
		comma=','
		if first:
			comma=''
			first=False
		tempstring+=comma+make_newick(treeobj)

	return '('+tempstring+')'+str(treeobject.node_name)

def make_json(treeobject):
	toreturn={}
	toreturn['name']=treeobject.node_name
	toreturn['data']=treeobject.item
	toreturn['children']=[]
	toreturn['lost']=''
	if treeobject.lost_vals:
		toreturn['lost']=' '.join(treeobject.lost_vals)
	if treeobject.subtrees!=None:
		for subtree in treeobject.subtrees:
			toreturn['children'].append(make_json(subtree))
	return toreturn
	# toreturn='{id : "'+treeobject.node_name+'",'
	# toreturn+='text : "'+treeobject.item+'",'
	# toreturn+='state : {opened : boolean,disabled : boolean,selected : boolean},children : ['
	# if treeobject.subtrees==None:
	# 	first=True
	# 	for subtree in treeobject.subtrees:
	# 		if not first:
	# 			toreturn+=','
	# 		toreturn+=make_json(subtree)
	# 		first=False
	# return toreturn+']}'

	# return toreturn

class segment:
	def __init__(self,segment_point,time,demo_id):
		self.id=[(demo_id,time)]
		self.label=segment_point

	def __str__(self):
		return str(self.id+[self.label])

	def __repr__(self):
		return str(self)

	def __eq__(self,other):
		if isinstance(other,segment):
			return self.label==other.label
		return False

	def merge(self,other):
		if isinstance(other,segment):
			self.id+=segment.id
		return self

class Tree:
	def __init__(self,entry,subtree=None,node_name='',lost_vals=None):
		if isinstance(entry,Tree):
			self=entry
		else:
			self.item=entry
			self.subtrees=subtree
			self.node_name=node_name
			self.lost_vals=lost_vals

	# def __str__(self):	
	# 	tempstring=''
	# 	printlist=[self]
	# 	while printlist!=[]:
	# 		to_add=[]
	# 		for item in printlist:
	# 			to_add.extend(item.subtrees)
	# 			tempstring+=item.item+','
	# 		tempstring+='\n'
	# 		printlist=to_add
	# 	return tempstring

	# def __repr__(self):
	# 	tempstring=''
	# 	printlist=[self]
	# 	while printlist!=[]:
	# 		to_add=[]
	# 		for item in printlist:
	# 			to_add.extend(item.subtrees)
	# 			tempstring+=item.item+','
	# 		tempstring+='\n'
	# 		printlist=to_add
	# 	return tempstring

	def __str__(self):
		return str(self.item)

	def __repr__(self):
		return str(self.item)

	def visualize(self,savepath='tree.txt',write_perm='False'):
		newick=make_newick(self)+';'
		self.newick=ete2.Tree(newick,format=1)
		print self.newick
		if write_perm:
			f=open(savepath,'w')
			f.write(str(self.newick))
			f.close()

	def save_tree_to_file(self,filepath):
		newick=make_newick(self)+';'
		# countleft=0
		# countright=0
		# for char in newick:
		# 	if char=='(':
		# 		countleft+=1
		# 	elif char==')':
		# 		countright+=1
		# print countleft,' ',countright
		# print newick
		self.newick=ete2.Tree(newick,format=1)
		ts=ete2.TreeStyle()
		ts.rotation=90
		#self.newick.show(tree_style=ts)
		self.newick.render(filepath,w=500,tree_style=ts)
		# print self.newick

	def render_tree(self):
		newick=make_newick(self)+';'
		# countleft=0
		# countright=0
		# for char in newick:
		# 	if char=='(':
		# 		countleft+=1
		# 	elif char==')':
		# 		countright+=1
		# print countleft,' ',countright
		# print newick
		self.newick=ete2.Tree(newick,format=8)
		ts=ete2.TreeStyle()
		ts.rotation=90
		#self.newick.show(tree_style=ts)
		self.newick.show(tree_style=ts)

	def return_json(self):
		self.json_=make_json(self)
		return self.json_

	def update(self,words):
		toreturn=[]
		for word in words:
			i=0
			while i<len(self.item):
				if self.item[i:i+len(word)]==word:
					toreturn.append((i,word))
					i+=len(word)
				else:
					i+=1
		sorted_return=sorted(toreturn,key=lambda x:x[0])
		found_ones=[]
		i=0
		while i<len(sorted_return):
			indice=thing[0]
			x=1
			new_thing=thing[1]
			found=1
			while sorted_return[i+x][0]==indice:
				if len(sorted_return[i+x][1])>len(new_thing):
					new_thing=sorted_return[i+x][1]
					found=x
				x+=1
			found_ones.append(new_thing)
			i+=x
		self.item=' '.join([x[1] for x in found_ones])



	
	


