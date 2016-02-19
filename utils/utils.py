import numpy as np

"""
Resolving discrete clusters to points
"""
def discrete2DClustersToPoints(model, dist, radius=1):
	result = []
	ac = [(round(a.means_[0][0]),round(a.means_[0][1])) for a in model]
	shape = np.shape(dist)

	print dist

	for a in ac:
		tmp  = []
		for i in range(-radius,radius+1):
			for j in range(-radius,radius+1):
				if a[0]+i >=0 and a[0]+i<shape[0] and a[1]+j >=0 and a[1]+j<shape[1]:
					tmp.append((a[0]+i,a[1]+j))

		maxtup = tmp[0]
		maxval = dist[tmp[0][0],tmp[0][1]]
		for i in tmp:
			print i
			if dist[i[0],i[1]] > maxval:
				maxval = dist[i[0],i[1]]
				maxtup = i

		result.append(maxtup)
	return result

def calculateStateDist(dims, trajs):
	counts = np.zeros((dims[0],dims[1]))
	for t in trajs:
		t1 = [tuple(a.tolist()) for a in t]
		for k in t1:
			counts[k[0],k[1]] = counts[k[0],k[1]] + 1
    
	return counts/np.sum(np.sum(counts))

def calculateStateTemporalDist(dims, trajs, waypoints):
	
	distArray = []

	for i in range(0, len(waypoints)):
		counts = np.zeros((dims[0],dims[1]))
		for t in trajs:
			segmentp = segmentTrajectory(t, waypoints)

			if segmentp != None:
				segment = segmentp[0]
				for k in segment:
					counts[k[0],k[1]] = counts[k[0],k[1]] + 1
    
		distArray.append(counts/np.sum(np.sum(counts)))
	return distArray

def segmentTrajectory(traj, waypoints):
	result = []
	
	tuptraj = [tuple(a.tolist()) for a in traj]
	running = []
	wpindex = 0
	for k in tuptraj:
		#print k, waypoints[wpindex]
		if k[0:2] == waypoints[wpindex]:
			running.append(k)
			result.append(running)
			running = []
			wpindex = wpindex + 1
		else:
			running.append(k)

	if wpindex != len(waypoints):
		return None
	else:
		return result

