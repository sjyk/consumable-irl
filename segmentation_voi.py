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
from sklearn import linear_model#,kernel_ridge
# import sklearn
import pickle as pkl
import IPython
import matplotlib.pyplot as plt
import sys
sys.path.append("../python-segmentation-benchmark/")
from generate.TrajectoryDataGenerator import *
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import FeedForwardNetwork,LinearLayer, SigmoidLayer,FullConnection
from pybrain.supervised.trainers import BackpropTrainer

REVERSE_LEVELS=False
RLPY_DEMOS=False

def get_demonstrations(demonstration_per_policy,max_policy_iter,num_policy_demo_checks,agent):
	"""return demonstrations generated from the parallel parking car rlpy simulator"""
	opt = {}
	opt["exp_id"] = 1
#    opt["path"] = "./Results/gridworld2"
	opt["checks_per_policy"] = 5
	opt["max_steps"] = 1000000
	opt["num_policy_checks"] = 1000
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
	# opt["agent"]=agent
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

def get_bezier_demos():
	"""returns a set of demonstrations that consist of random bezier curves"""
	params = {'k':10,'dims':2, 'observation':[0.15,0.15], 'resonance':[0.35,0.35], 'drift':[0,0]}
	system=createNewDemonstrationSystem(k=params['k'],
									   dims=params['dims'], 
									   observation=params['observation'], 
									   resonance=params['resonance'], 
									   drift=params['drift'])
	num_demos=100
	rtn=[]
	gtlist=[]
	initalcond=np.ones((2,1))
	for j in range(num_demos):
		print j
		t=sampleDemonstrationFromSystem(system,initalcond,lm=1.0, dp=0.0)
		rtn.append([np.squeeze(t[0])])
		gtlist.append([t[1]])

	return rtn,gtlist

def get_tsc_labels(demos):
	"""performs TSC on a set of demonstrations and returns the segmentation of each"""
	transitions=TransitionStateClustering(window_size=8)
	actual_demos=[]
	name_list=[]
	for i,demo_set in enumerate(demos):
		# ind=random.choice(range(len(demo_set)))
		ind=0
		demo=demo_set[ind]
		"""This deletion is only for the car sim"""
		if RLPY_DEMOS:
			temp=np.delete(demo,4,1)
		else:
			temp=demo
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
	"""gets segmentation of demonstrations at each level of the tree"""
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
	"""
	get_info is called on node.item 
	should return a list of all segmented demonstrations belonging to childrens of that node 
	each entry in the list has the following format
	[demo index,a list of the segmentations]
	where the list of segmentations is a list of tuples of the form (time stamp,segmentation change point)
	"""
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





def featurize_tree(demos,list_,corpora):
	"""adds segmentations from certain level of tree to demonstrations as an extra feature"""
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
	"""deprecated"""
	demolist=[[] for demo in demos]
	for item in list_:
		temp=list(item)
		if len(temp)==0:
			continue
		d=demos[temp[0].id[0][0]]
		if temp[-1].id[0][1]!=len(d)-1:
			temp.append(segment('end of demo',len(d)-1,temp[0].id[0][0]))
		else:
			temp[-1].label='end of demo'
		tlist=[]
		# print temp
		for x,step in enumerate(d):
			if len(temp)>1:
					if x>temp[0].id[0][1]:
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


####moved to clustering_tree.py

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
	"""Gets changepoints given demonstrations and then performs clustering. 
	Returns tsc segmentation for each level. (levels returned with index 0= top of tree)"""
	rtn,actual,name_list=get_tsc_labels(demos)
	intersect=lambda x,y:tree_gen.LCS_intersect(x,y)
	#clusterer=lambda x,y,num_groups:funcs.cluster_segmentation_data_k_means(x)#really just spectral clustering
	clusterer=lambda x,y,num_groups:funcs.cluster_segmentation_data_affinity(x)
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
	"""deprecated"""
	rtn,actual,name_list=get_tsc_labels(demos)
	corpora=get_corpora(rtn)
	return actual,featurize(actual,rtn,corpora)



def label(demos):
	"""returns labels for each timestep in a list of demonstrations (label=x(i+1)-xi)"""
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
	"""given a set of either demos+changepoint or just demos as features, uses linear regression to predict action given state.
	then computes prediction error of predicted_action-label"""
	# IPython.embed()
	feat=np.concatenate(features)
	lab=np.concatenate(labels)
	model.fit(feat,lab)
	values=model.predict(feat)
	# inLayer = LinearLayer(feat.shape[-1])
	# hiddenLayer = SigmoidLayer(120)
	# outLayer = LinearLayer(lab.shape[-1])
	# in_to_hidden = FullConnection(inLayer, hiddenLayer)
	# hidden_to_out = FullConnection(hiddenLayer, outLayer)
	# n = FeedForwardNetwork()
	# n.addInputModule(inLayer)
	# n.addModule(hiddenLayer)
	# n.addOutputModule(outLayer)
	# n.addConnection(in_to_hidden)
	# n.addConnection(hidden_to_out)
	# n.sortModules()
	# trndata=SupervisedDataSet( feat.shape[-1], lab.shape[-1] )
	# for i,x in enumerate(feat):
	# 	trndata.addSample(x,lab[i])
	# trainer = BackpropTrainer( n, dataset=trndata, lrdecay=.99, weightdecay=0.01)
	# trainer.trainEpochs(1)
	# values=activateOnDataset(trndata)
	rtn=[]
	for i in range(len(lab)):
		t=values[i]-lab[i]
		rtn.append(t.T.dot(t))
	return np.array(rtn)

def rewards(actions,states,domain):
	"""deprecated"""
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

def rewards2(actions,states,domain):
	"""deprecated"""
	eps_length = 0
	eps_return = 0
	eps_discount_return = 0
	eps_term = 0

	domain.s0()
	for i,a in enumerate(actions):

		r, ns, eps_term, p_actions = domain.step_dx_s(states[i],a)
		s = ns
		eps_return += r
		eps_discount_return += domain.discount_factor ** eps_length * \
			r
		eps_length += 1
		if eps_term:
			break
	return eps_discount_return

def get_actions(features,labels,model):
	""""""
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
	"""deprecated"""
	rewardlist=[]
	walls = [(-1, -0.3, 0.1, 0.3)]
	domain = RCIRL([(-0.1, -0.25)],
					  wallArray=walls,
					  noise=0)
	for actions in actionlist:
		rewardlist.append(rewards(actions,None,domain))
	return rewardlist

def get_all_rewards_2(actionlist,demos):
	"""deprecated"""
	rewardlist=[]
	walls = [(-1, -0.3, 0.1, 0.3)]
	domain = RCIRL([(-0.1, -0.25)],
					  wallArray=walls,
					  noise=0)
	for i,actions in enumerate(actionlist):
		rewardlist.append(rewards2(actions,demos[i],domain))
	return rewardlist

def get_corpora_tree(tree_):
	"""builds a mapping to assign change point labels to a number, so it may be used as an extra feature in a np matrix"""
	corpora={}
	stack=[tree_]
	ind=0
	while stack:
		if stack[0].node_name=='0_unrooted':
			stack=stack[0].subtrees
			continue
		new_level=[]
		templist=[]
		for tree in stack:
			children=tree.subtrees
			if children:
				new_level+=children
			for seg in tree.item:
				if seg.label not in corpora:
					corpora[seg.label]=ind
					ind+=1
		stack=new_level
	corpora['no_seg']=ind
	corpora['end of demo']=ind+1
	return corpora

def get_corpora(segment_lists):
	"""deprecated"""
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
	"""deprecated"""
	tot1=0
	tot2=0

	tot1=np.sum(errors1)/float(len(errors1))
	tot2=np.sum(errors2)/float(len(errors2))
	voitotal=(tot2-tot1)
	sum_=0
	for i,err in enumerate(errors1):
		err2=errors2[i]
		tvoi=err2-err
		sum_+=tvoi-voitotal
	return voitotal,sum_/float(len(errors1))

def VOI2(errors1,errors2):
	"""gets value of information by computing difference in expected error"""
	tot1=0
	tot2=0

	tot1=np.sum(errors1)/float(len(errors1))
	tot2=np.sum(errors2)/float(len(errors2))
	voitotal=-(tot2-tot1)
	sum_=0
	for i,err in enumerate(errors1):
		err2=errors2[i]
		tvoi=-(err2-err)
		sum_+=tvoi-voitotal
	return voitotal,sum_/float(len(errors1))

def test_clustering_changepoints():
	"""generates changepoints to test whether or not the clustering algorithm is working properly"""
	return [[segment(random.randint(0,3),0,x)] for x in range(100)]+[[segment(5,0,x+100)] for x in range(50)]+[[segment(10,0,x+150)] for x in range(50)],[],[]

def test_changepoint_demos():
	"""generates very simple demonstrations that should have near deterministic results 
	when input into changepoint algorithm. Used to debug tsc algo"""
	return [[np.array([[1],[random.randint(1,4)]])] for x in range(100)]+[[np.array([[1],[5]])+x] for x in range(50)]+[[np.array([[1],[10]])+x] for x in range(50)]



if __name__=='__main__':
	# demos=get_demonstrations(0,0,0,0)
	# demos,wtf=get_bezier_demos()
	# del wtf
	demos=map(lambda x:map(lambda y:y[::5],x), pkl.load(open('bezier_obs_demos.pkl','rb')))
	# demos=test_changepoint_demos()

	actual,demo_w_tsh=TSH_labeling(demos)
	labels=label(actual)
	model=linear_model.LinearRegression()

	err1=get_errors(actual,labels,model)
	voi1=[]
	std1=[]
	for d in demo_w_tsh:
		err2=get_errors(d,labels,model)
		tempvoi,tempstd=VOI2(err1,err2)
		
		voi1.append(tempvoi)
		std1.append(tempstd)

	print voi1
	print std1

	plt.bar(range(len(voi1)),list(reversed(voi1)),yerr=std1)
	plt.show()
	IPython.embed()

