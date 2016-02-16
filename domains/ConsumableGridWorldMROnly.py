"""
This class implements a GridWorld with
consumable rewards
"""
import os,sys,inspect
from rlpy.Tools import __rlpy_location__, findElemArray1D, perms
from rlpy.Domains.Domain import Domain
from ConsumableGridWorld import ConsumableGridWorld
import numpy as np
from rlpy.Tools import plt, FONTSIZE, linearMap

"""
This class allows for configurable reward functions
that are history-dependent.
"""
class ConsumableGridWorldMROnly(ConsumableGridWorld): 
    #__metaclass__ = Domain
    #default paths
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    default_map_dir = os.path.join(currentdir,"ConsumableGridWorldMaps")

    """
    A reward function is a function over ALL of the previous states, 
    the set of goal states,
    the step reward constant, 
    and the goal reward constant.
    """
    def __init__(self, 
    			 goalArray,
    			 rewardFunction=None, 
    			 mapname=os.path.join(default_map_dir, "4x5.txt"),
                 noise=.1, 
                 episodeCap=None):

    	self.prev_states = []
    	self.rewardFunction = rewardFunction
       
        super(ConsumableGridWorldMROnly, self).__init__(goalArray, mapname, noise, episodeCap)

    def step(self, a):
    	result = super(ConsumableGridWorldMROnly, self).step(a)
        self.prev_states.append((result[1],a))
        
        if self.rewardFunction == None:
            return result
        else:
        	r = self.rewardFunction(self.prev_states, 
        							self.goalArray, 
        							self.STEP_REWARD, 
        							self.GOAL_REWARD)
        	return r,result[1],result[2],result[3]

    """
    Override to also initialize history to null
    """
    def s0(self):
    	self.prev_states = []
    	return super(ConsumableGridWorldMROnly, self).s0()

    """
    Popular reward functions that you could use
    """
    @staticmethod
    def allMarkovReward(ps,ga, sr, gr):
    	r = sr
    	last_state = ps[len(ps)-1][0]
    	if (last_state[0],last_state[1]) in ga:
    		r = gr
    	return r

    implicitReward = None
