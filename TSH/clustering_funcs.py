# from clustering_tree import Tree
from R_to_V import get_bag_of_words
from spectral_clustering import *
import random
import math
import IPython
from itertools import combinations
import numpy as np
import sklearn.cluster

def cluster_spectral_k_means(groups,threshold=lambda x:x<=0,num_groups=lambda x:int(math.ceil(len(x)/4.0)),num_dims=2,opt_cutoff=.5):
	word_dict={}
	for group in groups:
		# if isinstance(group.item,list):
		# 	words=group.item
		# else:
		words=get_bag_of_words(group.item)
        # print ','.join(group.item)

		# group.item=' '.join(group.item)
		for word in words:
			word_dict[word]=1
	word_list=word_dict.keys()
	# print word_list
	for i,group in enumerate(groups):
		if group==None:
			print i

	similarity_mat=map_blocks_similarity(groups,word_list)
	filtered_mat=filter_edges(similarity_mat,threshold)
	W=map_edges_tree(filtered_mat,groups)[1]
	laplacian=get_laplacian(W)
	eigen=calc_eigen(laplacian,num_dims)
	# with open("tempy.txt","w") as f:
	# 	f.write(str(eigen))
	# print len(eigen)
	list_of_points=map(lambda x:Point(x[1].tolist(),groups[x[0]],x),enumerate(eigen))
	print len(groups),' ',len(groups)/4.0, ' ',math.ceil(len(groups)/4.0),' ',int(math.ceil(len(groups)/4.0)),' ',num_groups(groups) 

	kmean_cluster=kmeans(list_of_points,num_groups(groups),opt_cutoff)
	clusters=map(lambda x:map(lambda y:y.block,x.points),kmean_cluster)
	# print map(lambda x:map(lambda y:y.node_name,x),clusters)
	# print map(lambda x:x.node_name,groups)
	# print clusters
	return clusters

def cluster_segmentation_data_k_means(groups,threshold=lambda x:x<=0,num_groups=lambda x:int(math.ceil(len(x)/4.0)),num_dims=2,opt_cutoff=.5):
    similarity_mat=get_ed_mat(groups)
    cluster_o_matic=sklearn.cluster.SpectralClustering(n_clusters=num_groups(groups),affinity='precomputed')
    # cluster_o_matic=sklearn.cluster.AffinityPropagation(affinity='precomputed')
    labs=cluster_o_matic.fit_predict(similarity_mat)
    # labs=cluster_o_matic.predict(similarity_mat)
    rtn_dict={}
    for i,lab in enumerate(labs):
        if lab not in rtn_dict:
            rtn_dict[lab]=[]
        rtn_dict[lab].append(groups[i])
    return [x for x in rtn_dict.values()]

def cluster_segmentation_data_affinity(groups,threshold=lambda x:x<=0,num_groups=lambda x:int(math.ceil(len(x)/4.0)),num_dims=2,opt_cutoff=.5):
    similarity_mat=get_lcs_mat(groups)
    cluster_o_matic=sklearn.cluster.AffinityPropagation(affinity='precomputed')
    labs=cluster_o_matic.fit_predict(similarity_mat)
    rtn_dict={}
    for i,lab in enumerate(labs):
        if lab not in rtn_dict:
            rtn_dict[lab]=[]
        rtn_dict[lab].append(groups[i])
    return [x for x in rtn_dict.values()]

def relational_cluster(groups,garbage_threshold,similarity_metric=lambda x,y,match:edit_distance(x,y,match),element_thresh=0,group_thresh=0,num_clusters=1,num_groups=1,point_thresh=None,match=lambda x,y:x==y):
    
    G=map(lambda i:GROUP(set([i[0]])),enumerate(groups))
    print len(G)
    column_similarity=get_col_sim(groups,similarity_metric,point_thresh,match)
    i=0
    while len(G)>num_clusters:
        pair=get_max_auto_sim(G,column_similarity,group_thresh,element_thresh)
        if pair ==None:
            break
        else:
            i+=1
            print 'made pair: '+str(i)
            G.remove(pair[0])
            G.remove(pair[1])
            G.append(pair[0].merge(pair[1]))
    
    return [retrieve_cluster_from_indices(groups,x.subtrees) for x in G]

def retrieve_cluster_from_indices(groups, set_o_indices):
    return [groups[i] for i in set_o_indices]

def get_lcs_mat(groups):
    from clustering_tree import LCS_binary_intersect
    len_=len(groups)
    ed_mat=np.zeros((len_,len_))
    for i in range(len_):
        for j in range(len_):
            ed_mat[i,j]=len(LCS_binary_intersect([groups[i],groups[j]],2)[0])
            # ed_mat[i,j]=edit_distance(groups[i],groups[j])
    return ed_mat
def get_ed_mat(groups):
    len_=len(groups)
    ed_mat=np.zeros((len_,len_))
    for i in range(len_):
        for j in range(len_):
            ed_mat[i,j]=edit_distance(groups[i],groups[j])
    return ed_mat

def get_col_sim(groups,sim_metric,point_thresh,match):
    len_=len(groups)
    ed_mat=np.zeros((len_,len_))
    print 'making ed_mat'
    for i in range(len_):
        print i
        for j in range(len_):
            if point_thresh==None:
                ed_mat[i,j]=sim_metric(groups[i],groups[j],match=match)
            else:
                ed_mat[i,j]=sim_metric(groups[i],groups[j],match=lambda x,y:match(x,y,thresh=point_thresh))
    print 'making col sim'
    col_sim=np.zeros((len_,len_))
    for i in range(len_):
        i_row=ed_mat[i]
        print i
        for j in range(len_):
            sum_=0
            j_row=ed_mat[j]
            # for k in range(len_):
            #     sum_+=abs(i_row[k]-j_row[k])/len_
            col_sim[i,j]=1-(sum(abs(i_row-j_row))/len_)
    # print col_sim
    return col_sim

def get_max_auto_sim(G,col_sim,g_thresh,e_thresh):
    combs=list(combinations(G,2))
    combs=sorted([(comb,auto_sim(comb[0],comb[1],col_sim)) for comb in combs],key=lambda x:x[1])
    i=0
    while len(combs)>0:
        
        maxval,max_=combs.pop()
        i+=1
        # print max_
        if i==100:
            print 'made it to 100'
        if max_<=g_thresh:
            continue
        elif not check_elem(maxval,e_thresh,col_sim):
            continue
        else:
            return maxval

    return None

def check_elem(val,e_thresh,col_sim):
    temp=val[0].merge(val[1]).subtrees
    for i,j in combinations(temp,2):
        if col_sim[i,j]<=e_thresh:
            return False
    return True

def auto_sim(v1,v2,col_sim):
    temp=v1.merge(v2).subtrees
    len_=len(temp)
    sum_=0
    for i,j in combinations(temp,2):
        sum_+=col_sim[i,j]

    return 2*(len_-1)*sum_/len_

def binary_segmentation_cluster(groups,threshold=lambda x:x<=0,num_groups=None,num_dims=2,match=lambda x,y:x==y):
    pair=get_lowest_ed_pair(groups,match)
    
    groups.remove(pair[0])
    groups.remove(pair[1])
    # return [pair]+map(lambda x:Tree(x.item,subtree=[x]),groups)
    return [pair]+map(lambda x:[x],groups)

def get_lowest_ed_pair(groups,match=lambda x,y:x==y):

    gen=combinations(groups,2)
    min_=float('inf')
    mv=None
    for pair in gen:

        val=edit_distance(pair[0],pair[1],match)
        # print val
        if val<min_:
            min_=val
            mv=pair
    return mv

def edit_distance(c1,c2,match=lambda x,y:x==y):
    mat=np.zeros((len(c1.item)+1,len(c2.item)+1))
    i=0
    while i<=len(c1.item):
        j=0
        while j<=len(c2.item):
            # print i,j
            if i==0:
                if j==0:
                    j+=1
                    continue

                mat[i,j]=mat[i,j-1]+1
            elif j==0:
                
                mat[i,j]=mat[i-1,j]+1
            else:
                if match(c1.item[i-1],c2.item[j-1]):

                    diff=0
                else:

                    diff=1

                mat[i,j]=min(mat[i,j-1]+1,mat[i-1,j]+1,mat[i-1,j-1]+diff)
            j+=1
        
        i+=1
    return mat[-1,-1]

def EDR_match(c1,c2,thresh=0):
    return 0 if abs(c1.x-c2.x)<=thresh and abs(c1.y-c2.y)<=thresh else 1

def DTW(c1,c2,dist=lambda x,y:abs(x-y)):
    mat=np.zeros((len(c1.item)+1,len(c2.item)+1))
    for i in xrange(1,len(c1.item)+1):
        mat[i,0]=float('inf')
    for i in xrange(1,len(c2.item)+1):
        mat[0,i]=float('inf')
    mat[0,0]=0
    for i in xrange(1,len(c1.item)+1):
        for j in xrange(1,len(c2.item)+1):
            cost=dist(c1.item[i-1],c2.item[j-1])
            mat[i,j]=cost+min(mat[i-1,j],mat[i,j-1],mat[i-1,j-1])

    return mat[-1,-1]




class GROUP:
    def __init__(self,subtrees):
        self.subtrees=subtrees
    
    def merge(self,g2):
        return GROUP(self.subtrees|g2.subtrees)


class Point:
    '''
    An point in n dimensional space
    '''
    def __init__(self, coords,block=None,tempy=None):
        '''
        coords - A list of values, one per dimension
        '''
        
        self.coords = coords
        self.n = len(coords)
        self.block=block
        self.deb=tempy
        if tempy==None:
            print 'bi'
        
    def __repr__(self):
        return str(self.coords)

class Cluster:
    '''
    A set of points and their centroid
    '''
    
    def __init__(self, points):
        '''
        points - A list of point objects
        '''
        
        if len(points) == 0: raise Exception("ILLEGAL: empty cluster")
        # The points that belong to this cluster
        self.points = points
        
        # The dimensionality of the points in this cluster
        self.n = points[0].n
        
        # Assert that all points are of the same dimensionality
        for p in points:
            if p.n != self.n: raise Exception("ILLEGAL: wrong dimensions")
            
        # Set up the initial centroid (this is usually based off one point)
        self.centroid = self.calculateCentroid()
        
    def __repr__(self):
        '''
        String representation of this object
        '''
        return str(self.points)
    
    def update(self, points):
        '''
        Returns the distance between the previous centroid and the new after
        recalculating and storing the new centroid.
        '''
        old_centroid = self.centroid
        self.points = points
        self.centroid = self.calculateCentroid()
        print 'update'
        shift = getDistance(old_centroid, self.centroid,self) 
        return shift
    
    def calculateCentroid(self):
        '''
        Finds a virtual center point for a group of n-dimensional points
        '''
        numPoints = len(self.points)
        # Get a list of all coordinates in this cluster
        coords = [p.coords for p in self.points]
        # Reformat that so all x's are together, all y'z etc.
        unzipped = zip(*coords)
        # Calculate the mean for each dimension
        centroid_coords = [math.fsum(dList)/numPoints for dList in unzipped]
        if centroid_coords==[]:
            print coords
        return Point(centroid_coords)

def kmeans(points, k, cutoff):
    # print k
    # Pick out k random points to use as our initial centroids
    initial = random.sample(points, k)
    print initial
    # Create k clusters using those centroids
    clusters = [Cluster([p]) for p in initial]
    # print points
    # print '\n'
    # print clusters
    # print '\n'
    # print k
    
    # Loop through the dataset until the clusters stabilize
    loopCounter = 0
    while True:
        # Create a list of lists to hold the points in each cluster
        lists = [ [] for c in clusters]
        clusterCount = len(clusters)
        
        # Start counting loops
        loopCounter += 1
        # For every point in the dataset ...
        for p in points:
            # Get the distance between that point and the centroid of the first
            # cluster.
            smallest_distance = getDistance(p, clusters[0].centroid)

        
            # Set the cluster this point belongs to
            clusterIndex = 0
        
            # For the remainder of the clusters ...
            for i in range(clusterCount - 1):
                # calculate the distance of that point to each other cluster's
                # centroid.
                distance = getDistance(p, clusters[i+1].centroid,cnum=i+1)
                # If it's closer to that cluster's centroid update what we
                # think the smallest distance is, and set the point to belong
                # to that cluster
                if distance < smallest_distance:
                    smallest_distance = distance
                    clusterIndex = i+1
            lists[clusterIndex].append(p)
        
        # Set our biggest_shift to zero for this iteration
        biggest_shift = 0.0
        
        # As many times as there are clusters ...
        for i in range(clusterCount):
            # Calculate how far the centroid moved in this iteration
            if lists[i]==[]:
                continue
            shift = clusters[i].update(lists[i])
            # Keep track of the largest move from all cluster centroid updates
            biggest_shift = max(biggest_shift, shift)
        
        # If the centroids have stopped moving much, say we're done!
        if biggest_shift < cutoff:
            print "Converged after %s iterations" % loopCounter
            break
    if len(points)!=sum(map(lambda x:len(x.points),clusters)):
    	print 'hi'
    	print clusters
    return clusters

def getDistance(a, b,cluster1=None,cnum=0,lc=0):
    '''
    Euclidean distance between two n-dimensional points.
    Note: This can be very slow and does not scale well
    '''
    if a.n != b.n:
    	print a.n,' ',b.n
        print a.deb,' ',b.deb
        if a.n==0:
            print a.coords
            print a.block
            # print a.deb
        else:
            print b.coords
            print b.block
            # print b.deb
    	print cluster1.points
        print cnum
        print lc
        raise Exception("ILLEGAL: non comparable points")
    
    ret = reduce(lambda x,y: x + pow((a.coords[y]-b.coords[y]), 2),range(a.n),0.0)
    return math.sqrt(ret)



