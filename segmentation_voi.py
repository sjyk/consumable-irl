from domains.RCIRL import RCIRL
from tsc.tsc import TransitionStateClustering
from mp.goalpaths import GoalPathPlanner
import numpy as np
from utils.utils import *
import TSH.clustering_tree as tree_gen
from TSH.clustering_tree import segment
import TSH.clustering_funcs as funcs
from rlpy.Domains import RCCar
from rlpy.Agents import Q_Learning
from rlpy.Representations import Tabular, IncrementalTabular, RBF
from rlpy.Policies import eGreedy
from utils.ConsumableExperiment import ConsumableExperiment as Experiment
#from rlpy.Tools import __rlpy_location__, findElemArray1D, perms
import os,sys,inspect 
import random
from sklearn import linear_model
import pickle as pkl
# import IPython
import matplotlib.pyplot as plt


REVERSE_LEVELS=False

def get_demonstrations(demonstration_per_policy,max_policy_iter,num_policy_demo_checks,agent):
	opt = {}
	opt["exp_id"] = 1
#    opt["path"] = "./Results/gridworld2"
	opt["checks_per_policy"] = 10
	opt["max_steps"] = 100000
	opt["num_policy_checks"] = 100
	exp = 0.3
	discretization = 20
	walls = [(-1, -0.3, 0.1, 0.3)]
	domain = RCIRL([(-0.1, -0.25)],
					  wallArray=walls,
					  noise=0, rewardFunction=RCIRL.rcreward)
	domain.episodeCap = 200
	# Representation 10
	representation = RBF(domain, num_rbfs=1000,resolution_max=25, resolution_min=25,
						 const_feature=False, normalize=True, seed=1) #discretization=discretization)
	# Policy
	policy = eGreedy(representation, epsilon=0.3)

	# Agent
	opt["agent"]=agent
	opt["agent"] = Q_Learning(representation=representation, policy=policy,
					   discount_factor=domain.discount_factor,
					   initial_learn_rate=0.7,
					   learn_rate_decay_mode="boyan", boyan_N0=700,
					   lambda_=0.)
	
	opt["domain"] = domain 


	pdomain = RCIRL([(-0.1, -0.25)],
					  wallArray=walls,
					  noise=0)

	experiment = Experiment(**opt)
	experiment.run(visualize_steps=False,
					   performance_domain = pdomain,
					   visualize_learning=False,
					   visualize_performance=1)
	# return experiment
	return map(lambda x:map(lambda y:np.array(y),x),experiment.all_experiment_list)
	# return map(lambda x:map(lambda y:np.array(map(eval,y)),x),experiment.result['all_steps'])



def get_tsc_labels(demos):
	transitions=TransitionStateClustering(window_size=3)
	actual_demos=[]
	name_list=[]
	for i,demo_set in enumerate(demos):
		ind=random.choice(range(len(demo_set)))
		demo=demo_set[ind]
		"""This deletion is only for the car sim"""
		temp=np.delete(demo,4,1)
		transitions.addDemonstration(temp)
		actual_demos.append(temp)
		name_list.append(str(i)+'_'+str(ind))
	transitions.fit(pruning=0)
	rtn=[[] for x in actual_demos]
	for transition in transitions.task_segmentation:
		# wall=walltype[transition[0]]
		rtn[transition[0]].append(segment(transition[2],transition[1],transition[0]))
	return rtn,actual_demos,name_list

def Traverse_BFS(our_tree):
	stack=[our_tree]
	rtn=[]
	i=0
	x=0
	while stack:
		x+=1
		if stack[0].node_name=='0_unrooted':
			stack=stack[0].subtrees
			continue
		new_level=[]
		# sum_=0
		# tempnum=0
		templist=[]
		for tree in stack:
			if len(tree.item)==0:
				if tree.subtrees:
					new_level+=tree.subtrees
				continue
			# if i%20:
			# 	print i, tree.node_name
			i+=1
			# tempnum+=1
			children=tree.subtrees
			# print tree
			# print children
			if children:
				new_level+=children
			# if isinstance(length,int):
			# 	diff=abs(len(tree.item)-length)
			# else:
			# 	diff=funcs.edit_distance(tree,tree_gen.Tree(length))
			templist.append(get_info(tree.item))
		
		stack=new_level
		rtn.append(templist)
	if REVERSE_LEVELS:
		return list(reversed(rtn))
	else:
		return rtn

def get_info(item):
	holdingdict={}
	# try:
	# 	for x in item[0].id:
	# 		holdingdict[x[0]]=[]
	# except:
	# 	IPython.embed()
	for x in item[0].id:
			holdingdict[x[0]]=[]
	rtn=[[] for x in item[0].id]
	for segments in item:
		for x,metadata in enumerate(segments.id):
			holdingdict[metadata[0]].append((metadata[1],segments.label))
	i=0
	for key in holdingdict:
		list_=holdingdict[key]
		rtn[i]=[key,list_]
		i+=1
	return rtn



"""
get_info should return the following data structure
{'associated':thingy}
where thingy is a list of all segmented demonstrations belonging to childrens of that node
each entry in the list has the following format
[demo index,a list of the segmentations]
where the list of segmentations is a list of tuples of the form (time stamp,segmentation change point)
"""

def featurize_tree(demos,list_,corpora):
	demolist=[[] for demo in demos]
	for item in list_:
		assoc=item
		for ass in assoc:
			id_=ass[0]
			order=ass[1]
			if len(order)==0:
				continue
			d=demos[id_]
			featurized_ass=[]
			# print order
			for x,step in enumerate(d):
				if len(order)>1:
					if x>(order[0][0]+order[1][0])/2:
						order=order[1:]
				tempstep=step.ravel().tolist()
				tempstep.append(corpora[order[0][1]])
				featurized_ass.append(tempstep)
			demolist[id_]=np.array(featurized_ass)
	for x,d in enumerate(demolist):
		if d!=[]:
			continue
		temp=[]
		for y,step in enumerate(demos[x]):
			tempstep=step.ravel().tolist()
			tempstep.append(float(corpora['no_seg']))
			temp.append(tempstep)
		demolist[x]=np.array(temp)
	return demolist

def featurize(demos,list_,corpora):
	demolist=[[] for demo in demos]
	for item in list_:
		temp=list(item)
		if len(temp)==0:
			continue
		d=demos[temp[0].id[0][0]]
		tlist=[]
		# print temp
		for x,step in enumerate(d):
			if len(temp)>1:
					if x>(temp[0].id[0][1]+temp[0].id[0][1])/2:
						temp=temp[1:]
			# if x>temp[0].id[1]:
			# 	temp=temp[1:]
			tempstep=step.ravel().tolist()
			tempstep.append(corpora[temp[0].label])
			tlist.append(tempstep)
		# print temp[0].id[0]
		demolist[temp[0].id[0][0]]=np.array(tlist)
	for x,d in enumerate(demolist):
		if d!=[]:
			continue
		temp=[]
		for y,step in enumerate(demos[x]):
			tempstep=step.ravel().tolist()
			tempstep.append(float(corpora['no_seg']))
			temp.append(tempstep)
		demolist[x]=np.array(temp)
	return demolist

# class segment:
# 	def __init__(self,segment_point,time,demo_id):
# 		self.id=[(demo_id,time)]
# 		self.label=segment_point

# 	def __str__(self):
# 		return str(self.id+[self.label])

# 	def __repr__(self):
# 		return str(self)

# 	def __eq__(self,other):
# 		if isinstance(other,segment):
# 			return self.label==other.label
# 		return False

# 	def merge(self,other):
# 		if isinstance(other,segment):
# 			self.id+=segment.id
# 		return self


def TSH_labeling(demos,level=None):
	rtn,actual,name_list=get_tsc_labels(demos)
	# print map(lambda x: x.shape,actual)
	intersect=lambda x,y:tree_gen.LCS_intersect(x,y)
	clusterer=lambda x,y,num_groups:funcs.cluster_segmentation_data_k_means(x)
	our_tree,node_dict,lost_dict=tree_gen.get_tree(rtn,clusterer,intersect_meth=intersect,names=name_list)
	corpora=get_corpora_tree(our_tree)
	BFS_traversal_list=Traverse_BFS(our_tree)
	if level:
		return actual,featurize_tree(actual,BFS_traversal_list[level],corpora)
	else:
		rtn=[]
		for level in range(len(BFS_traversal_list)):
			rtn.append(featurize_tree(actual,BFS_traversal_list[level],corpora))
		return actual,rtn

def TSC_labeling(demos):
	rtn,actual,name_list=get_tsc_labels(demos)
	# intersect=lambda x,y:tree_gen.LCS_intersect(x,y)
	# clusterer=lambda x,y,num_groups:funcs.cluster_segmentation_data_k_means(x)
	# our_tree,node_dict,lost_dict=tree_gen.get_tree(rtn,clusterer,intersect_meth=intersect,names=name_list)
	corpora=get_corpora(rtn)
	return actual,featurize(actual,rtn,corpora)



def label(demos):
	rtn=[]
	for demo in demos:
		templist=[]
		for ind in range(len(demo)-1):
			curr=demo[ind]
			next=demo[ind+1]
			delta=next-curr
			templist.append(delta.ravel().tolist())
		temp=templist[-1]
		t=[]
		for x in temp:
			t.append(0.0)
		templist.append(t)
		rtn.append(np.array(templist))
	return rtn

def get_errors(features,labels,model):
	# IPython.embed()
	feat=np.concatenate(features)
	lab=np.concatenate(labels)
	model.fit(feat,lab)
	values=model.predict(feat)
	rtn=[]
	for i in range(len(values)):
		t=values[i]-lab[i]
		rtn.append(t.T.dot(t))
	return np.array(rtn)

def rewards(actions,states,domain):

	eps_length = 0
	eps_return = 0
	eps_discount_return = 0
	eps_term = 0

	domain.s0()
	for a in actions:

		r, ns, eps_term, p_actions = domain.step_dx(a)
		s = ns
		eps_return += r
		eps_discount_return += domain.discount_factor ** eps_length * \
			r
		eps_length += 1
		if eps_term:
			break
	return eps_discount_return

def get_actions(features,labels,model):
	feat=np.concatenate(features)
	lab=np.concatenate(labels)
	model.fit(feat,lab)
	values=model.predict(feat)
	actionlist=[]
	for demo in features:
		len_=len(demo)
		actionlist.append(values[:len_])
		values=values[len_:]
	return actionlist

def get_all_rewards(actionlist):
	rewardlist=[]
	walls = [(-1, -0.3, 0.1, 0.3)]
	domain = RCIRL([(-0.1, -0.25)],
					  wallArray=walls,
					  noise=0)
	for actions in actionlist:
		rewardlist.append(rewards(actions,None,domain))
	return rewardlist

def get_corpora_tree(tree_):
	corpora={}
	stack=[tree_]
	ind=0
	while stack:
		if stack[0].node_name=='0_unrooted':
			stack=stack[0].subtrees
			continue
		new_level=[]
		# sum_=0
		# tempnum=0
		templist=[]
		for tree in stack:
			# tempnum+=1
			children=tree.subtrees
			# print tree
			# print children
			if children:
				new_level+=children
			# if isinstance(length,int):
			# 	diff=abs(len(tree.item)-length)
			# else:
			# 	diff=funcs.edit_distance(tree,tree_gen.Tree(length))
			for seg in tree.item:
				if seg.label not in corpora:
					corpora[seg.label]=ind
					ind+=1
			# templist.append(get_info(tree.item))
		# rtn.append(templist)
		stack=new_level
	corpora['no_seg']=ind
	return corpora

def get_corpora(segment_lists):
	corpora={}
	ind=0
	for segmentation in segment_lists:
		for seg in segmentation:
			if seg.label not in corpora:
				corpora[seg.label]=ind
				ind+=1
	corpora['no_seg']=ind
	return corpora

def VOI(errors1,errors2):
	tot1=0
	tot2=0
	# for i,errorlist in enumerate(errors1):
	# 	errorlist2=errors2[i]
	# 	tot1+=sum(errorlist)/float(len(errorlist))
	# 	tot2+=sum(errorlist2)/float(len(errorlist2))
	tot1=np.sum(errors1)/float(len(errors1))
	tot2=np.sum(errors2)/float(len(errors2))
	return -(tot2-tot1)



# experiment=get_demonstrations(0,0,0,0)
demos=pkl.load(open('temp_demos.pkl','rb'))
# IPython.embed()
# pkl.dump(demos,open('temp_demos.pkl','wb'))


actual,demo_w_tsh=TSH_labeling(demos)
labels=label(actual)
model=linear_model.LinearRegression()
# actual=np.array(actual)
# err1=get_errors(actual,labels,model)
act1=get_actions(actual,labels,model)
rew1=get_all_rewards(act1)
voi1=[]
for d in demo_w_tsh:
	# err2=get_errors(d,labels,model)
	act2=get_actions(d,labels,model)
	rew2=get_all_rewards(act2)
	# voi1.append(VOI(err1,err2))
	voi1.append(VOI(rew1,rew2))

actual,demo_w_tsc=TSC_labeling(demos)
labels=label(actual)
model=linear_model.LinearRegression()
# err1=get_errors(actual,labels,model)
# err2=get_errors(demo_w_tsc,labels,model)
act2=get_actions(demo_w_tsc,labels,model)
rew2=get_all_rewards(act2)
# voi2=VOI(err1,err2)
voi2=VOI(rew1,rew2)
print voi1
print voi2
plt.plot(voi1)
plt.axhline(voi2)
plt.show()