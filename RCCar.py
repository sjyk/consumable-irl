#!/usr/bin/env python
#from domains.ConsumableGridWorld import ConsumableGridWorld
from domains.RCIRL import RCIRL
from tsc.tsc import TransitionStateClustering
from mp.goalpaths import GoalPathPlanner
import numpy as np
from utils.utils import *

from rlpy.Domains import RCCar
from rlpy.Agents import Q_Learning
from rlpy.Representations import Tabular, IncrementalTabular, RBF
from rlpy.Policies import eGreedy
from utils.ConsumableExperiment import ConsumableExperiment as Experiment
#from rlpy.Tools import __rlpy_location__, findElemArray1D, perms
import os,sys,inspect 

if __name__ == '__main__':
    opt = {}
    opt["exp_id"] = 1
    opt["path"] = "./Results/gridworld2"
    opt["checks_per_policy"] = 10
    opt["max_steps"] = 1000000
    opt["num_policy_checks"] = 100
    exp = 0.3
    discretization = 20
    domain = RCIRL([(-0.1, -0.25)],noise=0, rewardFunction=RCIRL.rcreward)
    domain.episodeCap = 200
    # Representation 10
    representation = RBF(domain, num_rbfs=1000,resolution_max=25, resolution_min=25,
                         const_feature=False, normalize=True, seed=1) #discretization=discretization)
    # Policy
    policy = eGreedy(representation, epsilon=0.3)

    # Agent
    opt["agent"] = Q_Learning(representation=representation, policy=policy,
                       discount_factor=domain.discount_factor,
                       initial_learn_rate=0.7,
                       learn_rate_decay_mode="boyan", boyan_N0=700,
                       lambda_=0.)
    opt["domain"] = domain 


    pdomain = RCIRL([(-0.1, -0.25)],noise=0)

    experiment = Experiment(**opt)
    experiment.run(visualize_steps=False,
                       performance_domain = pdomain,
                       visualize_learning=False,
                       visualize_performance=1)
    experiment.save()
    