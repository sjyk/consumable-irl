from rlpy.Domains import Pinball
from rlpy.Agents import Q_Learning
from rlpy.Representations import Tabular
from rlpy.Policies import eGreedy
from rlpy.Experiments import Experiment
#from rlpy.Tools import __rlpy_location__, findElemArray1D, perms
import os
import numpy as np


"""
This class implements a path planner for rlpy
environments. It does so by rolling out
trajectories of policies
"""
class GoalPathPlanner:

	"""
	This class takes a domain as a parameter
	and learns a model to completion.
	"""
	def __init__(self, domain, representation, policy,steps=100000):
		
		opt = {}
		opt["domain"] = domain
		# Agent
		opt["agent"] = Q_Learning(representation=representation, policy=policy,
                       discount_factor=domain.discount_factor,
                       initial_learn_rate=0.1,
                       learn_rate_decay_mode="boyan", boyan_N0=100,
                       lambda_=0.)
    
		opt["checks_per_policy"] = 10
		opt["max_steps"] = steps
		opt["num_policy_checks"] = 20
		experiment = Experiment(**opt)
		experiment.run()
		self.policy = opt["agent"].policy
		self.domain = domain

	"""
	Using the policy this generates a set
	of trajectories
	"""
	def generateTrajectories(self, N=10):
		demonstrations = []
		for i in range(0, N):
			traj = []
			cur_state = self.domain.s0()[0]
			for j in range(0, self.domain.episodeCap):
				traj.append(cur_state)
				terminal = self.domain.isTerminal()

				if terminal:
					break

				a = self.policy.pi(cur_state, terminal,self.domain.possibleActions())
				cur_state = self.domain.step(a)[1]
				
			demonstrations.append(traj)
		return demonstrations



