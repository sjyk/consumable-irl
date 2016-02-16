"""
This class implements a path planner for rlpy
environments. It does so by rolling out
trajectories of policies
"""

__author__ = "Robert H. Klein"

from domains.ConsumableGridWorld import ConsumableGridWorld
from domains.ConsumableGridWorldIRL import ConsumableGridWorldIRL
from tsc.tsc import TransitionStateClustering
from mp.goalpaths import GoalPathPlanner
import numpy as np
from utils.utils import *

from rlpy.Domains import Pinball
from rlpy.Agents import Q_Learning
from rlpy.Representations import Tabular, IncrementalTabular
from rlpy.Policies import eGreedy
from utils.ConsumableExperiment import ConsumableExperiment as Experiment
#from rlpy.Tools import __rlpy_location__, findElemArray1D, perms
import os,sys,inspect 


def grid_world1_markov(exp_id=1, path="./Results/gridworld1"):
    opt = {}
    opt["exp_id"] = exp_id
    opt["path"] = path
    opt["checks_per_policy"] = 10
    opt["max_steps"] = 150000
    opt["num_policy_checks"] = 20
    noise = 0.1
    exp = 0.3
    discretization = 400

    maze = os.path.join(ConsumableGridWorld.default_map_dir, '10x7-ACC2011.txt')
    domain = ConsumableGridWorldIRL([(7,5), (1,2)], 
                                    mapname=maze, 
                                    encodingFunction= lambda x: ConsumableGridWorldIRL.allMarkovEncoding(x), 
                                    noise=noise)
    
    opt["domain"] = domain

    # Representation
    representation = Tabular(domain, discretization=discretization)

    # Policy
    policy = eGreedy(representation, epsilon=exp)

    # Agent
    opt["agent"] = Q_Learning(representation=representation, policy=policy,
                       discount_factor=domain.discount_factor,
                       initial_learn_rate=0.1,
                       learn_rate_decay_mode="boyan", boyan_N0=100,
                       lambda_=0.)

    experiment = Experiment(**opt)
    experiment.run(visualize_steps=False,
                   visualize_learning=False,
                   visualize_performance=0)
    experiment.save()
    return np.max(experiment.result["return"]),np.sum(experiment.result["return"])

def grid_world1_reward(exp_id=2, path="./Results/gridworld1"):
    opt = {}
    opt["exp_id"] = exp_id
    opt["path"] = path
    opt["checks_per_policy"] = 10
    opt["max_steps"] = 150000
    opt["num_policy_checks"] = 20
    noise = 0.1
    exp = 0.3
    discretization = 400

    maze = os.path.join(ConsumableGridWorld.default_map_dir, '10x7-ACC2011.txt')
    domain = ConsumableGridWorldIRL([(7,5), (1,2)], 
                                    mapname=maze, 
                                    encodingFunction= lambda x: ConsumableGridWorldIRL.stateVisitEncoding(x,[(7,5)]), 
                                    noise=noise,
                                    binary=True)
    
    opt["domain"] = domain

    # Representation
    representation = Tabular(domain, discretization=discretization)

    # Policy
    policy = eGreedy(representation, epsilon=exp)

    # Agent
    opt["agent"] = Q_Learning(representation=representation, policy=policy,
                       discount_factor=domain.discount_factor,
                       initial_learn_rate=0.1,
                       learn_rate_decay_mode="boyan", boyan_N0=100,
                       lambda_=0.)

    experiment = Experiment(**opt)
    experiment.run(visualize_steps=False,
                   visualize_learning=False,
                   visualize_performance=0)
    experiment.save()
    return np.max(experiment.result["return"]),np.sum(experiment.result["return"])

def grid_world1_sliding(exp_id=3, path="./Results/gridworld1"):
    opt = {}
    opt["exp_id"] = exp_id
    opt["path"] = path
    opt["checks_per_policy"] = 10
    opt["max_steps"] = 150000
    opt["num_policy_checks"] = 20
    noise = 0.1
    exp = 0.3
    discretization = 400

    maze = os.path.join(ConsumableGridWorld.default_map_dir, '10x7-ACC2011.txt')
    domain = ConsumableGridWorldIRL([(7,5), (1,2)], 
                                    mapname=maze, 
                                    encodingFunction= lambda x: ConsumableGridWorldIRL.slidingWindowEncoding(x,3), 
                                    noise=noise)
    
    opt["domain"] = domain

    # Representation
    representation = IncrementalTabular(domain, discretization=discretization)

    # Policy
    policy = eGreedy(representation, epsilon=exp)

    # Agent
    opt["agent"] = Q_Learning(representation=representation, policy=policy,
                       discount_factor=domain.discount_factor,
                       initial_learn_rate=0.1,
                       learn_rate_decay_mode="boyan", boyan_N0=100,
                       lambda_=0.)

    experiment = Experiment(**opt)
    experiment.run(visualize_steps=False,
                   visualize_learning=False,
                   visualize_performance=0)
    experiment.save()
    return np.max(experiment.result["return"]),np.sum(experiment.result["return"])

def grid_world1_trp(exp_id=4, path="./Results/gridworld1"):
    opt = {}
    opt["exp_id"] = exp_id
    opt["path"] = path
    opt["checks_per_policy"] = 10
    opt["max_steps"] = 150000
    opt["num_policy_checks"] = 20
    noise = 0.1
    exp = 0.3
    discretization = 20

    # Domain:
    maze = os.path.join(ConsumableGridWorld.default_map_dir, '10x7-ACC2011.txt')
    domain = ConsumableGridWorldIRL([(7,5), (1,2)],
                                    mapname=maze, 
                                    encodingFunction= lambda x: ConsumableGridWorldIRL.stateVisitEncoding(x,[(7,5)]),
                                    binary=True, 
                                    noise=noise)
    #domain = Pinball(noise=0.3)
    
    # Representation
    representation = Tabular(domain, discretization=discretization)

    # Policy
    policy = eGreedy(representation, epsilon=0.3)

    d = GoalPathPlanner(domain, representation,policy)
    trajs = d.generateTrajectories(N=5)
    a = TransitionStateClustering(window_size=2)
    for t in trajs:
        N = len(t)
        demo = np.zeros((N,2))
        for i in range(0,N):
            demo[i,:] = t[i][0:2]
        a.addDemonstration(demo)
    a.fit(normalize=False, pruning=0.5)
    ac = [(round(a.means_[0][0]),round(a.means_[0][1])) for a in a.model]

    print ac

    #reinitialize
    domain = ConsumableGridWorldIRL([(7,5), (1,2)],
                                    mapname=maze, 
                                    encodingFunction= lambda x: ConsumableGridWorldIRL.statePassageEncoding(x,ac,5), noise=noise)
    representation = IncrementalTabular(domain, discretization=discretization)
    policy = eGreedy(representation, epsilon=0.3)
    opt["agent"] = Q_Learning(representation=representation, policy=policy,
                       discount_factor=domain.discount_factor,
                       initial_learn_rate=0.1,
                       learn_rate_decay_mode="boyan", boyan_N0=100,
                       lambda_=0.)

    opt["domain"] = domain

    experiment = Experiment(**opt)
    experiment.run(visualize_steps=False,
                   visualize_learning=False,
                   visualize_performance=0)
    experiment.save()
    return np.max(experiment.result["return"]),np.sum(experiment.result["return"])

def grid_world1_trb(exp_id=6, path="./Results/gridworld1"):
    opt = {}
    opt["exp_id"] = exp_id
    opt["path"] = path
    opt["checks_per_policy"] = 10
    opt["max_steps"] = 150000
    opt["num_policy_checks"] = 20
    noise = 0.1
    exp = 0.3
    discretization = 20

    # Domain:
    maze = os.path.join(ConsumableGridWorld.default_map_dir, '10x7-ACC2011.txt')
    domain = ConsumableGridWorldIRL([(7,5), (1,2)],
                                    mapname=maze, 
                                    encodingFunction= lambda x: ConsumableGridWorldIRL.stateVisitEncoding(x,[(7,5)]),
                                    binary=True, 
                                    noise=noise)
    #domain = Pinball(noise=0.3)
    
    # Representation
    representation = Tabular(domain, discretization=discretization)

    # Policy
    policy = eGreedy(representation, epsilon=0.3)

    d = GoalPathPlanner(domain, representation,policy)
    trajs = d.generateTrajectories(N=5)
    a = TransitionStateClustering(window_size=2)
    for t in trajs:
        N = len(t)
        demo = np.zeros((N,2))
        for i in range(0,N):
            demo[i,:] = t[i][0:2]
        a.addDemonstration(demo)
    a.fit(normalize=False, pruning=0.5)
    dist = calculateStateDist((10,7), trajs) 
    ac = discrete2DClustersToPoints(a.model, dist, radius=1)

    #ac = [(round(a.means_[0][0]),round(a.means_[0][1])) for a in a.model]

    print ac

    #reinitialize
    domain = ConsumableGridWorldIRL([(7,5), (1,2)],
                                    mapname=maze, 
                                    encodingFunction= lambda x: ConsumableGridWorldIRL.stateVisitEncoding(x,ac), noise=noise,binary=True)
    representation = IncrementalTabular(domain, discretization=discretization)
    policy = eGreedy(representation, epsilon=0.3)
    opt["agent"] = Q_Learning(representation=representation, policy=policy,
                       discount_factor=domain.discount_factor,
                       initial_learn_rate=0.1,
                       learn_rate_decay_mode="boyan", boyan_N0=100,
                       lambda_=0.)

    opt["domain"] = domain

    experiment = Experiment(**opt)
    experiment.run(visualize_steps=False,
                   visualize_learning=False,
                   visualize_performance=0)
    experiment.save()
    return np.max(experiment.result["return"]),np.sum(experiment.result["return"])

def gridworld1_irl(exp_id=5, path="./Results/gridworld1"):
    opt = {}
    opt["exp_id"] = exp_id
    opt["path"] = path
    opt["checks_per_policy"] = 10
    opt["max_steps"] = 150000
    opt["num_policy_checks"] = 20
    noise = 0.1
    exp = 0.3
    discretization = 400

    # Domain:
    maze = os.path.join(ConsumableGridWorld.default_map_dir, '10x7-ACC2011.txt')
    domain = ConsumableGridWorldIRL([(7,5), (1,2)],
                                    mapname=maze, 
                                    encodingFunction= lambda x: ConsumableGridWorldIRL.stateVisitEncoding(x,[(7,5)]), 
                                    noise=noise,
                                    binary=True)
    #domain = Pinball(noise=0.3)

    # Representation
    representation = Tabular(domain, discretization=discretization)

    # Policy
    policy = eGreedy(representation, epsilon=0.3)

    # Agent
    opt["agent"] = Q_Learning(representation=representation, policy=policy,
                       discount_factor=domain.discount_factor,
                       initial_learn_rate=0.1,
                       learn_rate_decay_mode="boyan", boyan_N0=100,
                       lambda_=0.)
    
    opt["checks_per_policy"] = 10
    opt["max_steps"] = 150000
    opt["num_policy_checks"] = 20
    

    d = GoalPathPlanner(domain, representation,policy)
    trajs = d.generateTrajectories(N=5) 
    dist = calculateStateDist((10,7), trajs)        

    # Policy reset
    policy = eGreedy(representation, epsilon=0.3)
    representation = Tabular(domain, discretization=discretization)

    opt["agent"] = Q_Learning(representation=representation, policy=policy,
                       discount_factor=domain.discount_factor,
                       initial_learn_rate=0.1,
                       learn_rate_decay_mode="boyan", boyan_N0=100,
                       lambda_=0.)

    domain = ConsumableGridWorldIRL([(7,5), (1,2)],
                                    mapname=maze, 
                                    encodingFunction= lambda x: ConsumableGridWorldIRL.allMarkovEncoding(x),
                                    rewardFunction= lambda x,y,z,w: ConsumableGridWorldIRL.maxEntReward(x,y,z,w,dist),
                                    noise=noise)
    
    pdomain = ConsumableGridWorldIRL([(7,5), (1,2)],
                                    mapname=maze, 
                                    encodingFunction= lambda x: ConsumableGridWorldIRL.allMarkovEncoding(x),
                                    noise=noise)

    opt["domain"] = domain
    experiment = Experiment(**opt)
    experiment.run(visualize_steps=False,
                   performance_domain = pdomain,
                   visualize_learning=False,
                   visualize_performance=0)
    experiment.save()

    return np.max(experiment.result["return"]),np.sum(experiment.result["return"])

def gridworld1_rirl(exp_id=6, path="./Results/gridworld1"):
    opt = {}
    opt["exp_id"] = exp_id
    opt["path"] = path
    opt["checks_per_policy"] = 10
    opt["max_steps"] = 150000
    opt["num_policy_checks"] = 20
    noise = 0.1
    exp = 0.3
    discretization = 400

    # Domain:
    maze = os.path.join(ConsumableGridWorld.default_map_dir, '10x7-ACC2011.txt')
    domain = ConsumableGridWorldIRL([(7,5), (1,2)],
                                    mapname=maze, 
                                    encodingFunction= lambda x: ConsumableGridWorldIRL.stateVisitEncoding(x,[(7,5)]), 
                                    noise=noise,
                                    binary=True)
    #domain = Pinball(noise=0.3)

    # Representation
    representation = Tabular(domain, discretization=discretization)

    # Policy
    policy = eGreedy(representation, epsilon=0.3)

    # Agent
    opt["agent"] = Q_Learning(representation=representation, policy=policy,
                       discount_factor=domain.discount_factor,
                       initial_learn_rate=0.1,
                       learn_rate_decay_mode="boyan", boyan_N0=100,
                       lambda_=0.)
    
    opt["checks_per_policy"] = 10
    opt["max_steps"] = 150000
    opt["num_policy_checks"] = 20
    

    d = GoalPathPlanner(domain, representation,policy)
    trajs = d.generateTrajectories(N=5) 
    dist = calculateStateDist((10,7), trajs)        

    # Policy reset
    policy = eGreedy(representation, epsilon=0.3)
    representation = Tabular(domain, discretization=discretization)

    opt["agent"] = Q_Learning(representation=representation, policy=policy,
                       discount_factor=domain.discount_factor,
                       initial_learn_rate=0.1,
                       learn_rate_decay_mode="boyan", boyan_N0=100,
                       lambda_=0.)

    domain = ConsumableGridWorldIRL([(7,5), (1,2)],
                                    mapname=maze, 
                                    encodingFunction= lambda x: ConsumableGridWorldIRL.stateVisitEncoding(x,[(7,5)]),
                                    rewardFunction= lambda x,y,z,w: ConsumableGridWorldIRL.rewardIRL(x,y,z,w,dist),
                                    noise=noise)
    
    pdomain = ConsumableGridWorldIRL([(7,5), (1,2)],
                                    mapname=maze, 
                                    encodingFunction= lambda x: ConsumableGridWorldIRL.stateVisitEncoding(x,[(7,5)]),
                                    noise=noise)

    opt["domain"] = domain
    experiment = Experiment(**opt)
    experiment.run(visualize_steps=False,
                   performance_domain = pdomain,
                   visualize_learning=False,
                   visualize_performance=0)
    experiment.save()

    return np.max(experiment.result["return"]),np.sum(experiment.result["return"])

def gridworld1_tirl(exp_id=7, path="./Results/gridworld1"):
    opt = {}
    opt["exp_id"] = exp_id
    opt["path"] = path
    opt["checks_per_policy"] = 10
    opt["max_steps"] = 150000
    opt["num_policy_checks"] = 20
    noise = 0.1
    exp = 0.3
    discretization = 400

    # Domain:
    maze = os.path.join(ConsumableGridWorld.default_map_dir, '10x7-ACC2011.txt')
    domain = ConsumableGridWorldIRL([(7,5), (1,2)],
                                    mapname=maze, 
                                    encodingFunction= lambda x: ConsumableGridWorldIRL.stateVisitEncoding(x,[(7,5)]), 
                                    noise=noise,
                                    binary=True)
    #domain = Pinball(noise=0.3)

    # Representation
    representation = Tabular(domain, discretization=discretization)

    # Policy
    policy = eGreedy(representation, epsilon=0.3)

    # Agent
    opt["agent"] = Q_Learning(representation=representation, policy=policy,
                       discount_factor=domain.discount_factor,
                       initial_learn_rate=0.1,
                       learn_rate_decay_mode="boyan", boyan_N0=100,
                       lambda_=0.)
    
    opt["checks_per_policy"] = 10
    opt["max_steps"] = 150000
    opt["num_policy_checks"] = 20
    

    d = GoalPathPlanner(domain, representation,policy)
    trajs = d.generateTrajectories(N=5) 
    dist = calculateStateDist((10,7), trajs)   
    a = TransitionStateClustering(window_size=2)
    for t in trajs:
        N = len(t)
        demo = np.zeros((N,2))
        for i in range(0,N):
            demo[i,:] = t[i][0:2]
        a.addDemonstration(demo)
    a.fit(normalize=False, pruning=0.5)
    dist = calculateStateDist((10,7), trajs) 
    ac = discrete2DClustersToPoints(a.model, dist, radius=1)     

    # Policy reset
    policy = eGreedy(representation, epsilon=0.3)
    representation = Tabular(domain, discretization=discretization)

    opt["agent"] = Q_Learning(representation=representation, policy=policy,
                       discount_factor=domain.discount_factor,
                       initial_learn_rate=0.1,
                       learn_rate_decay_mode="boyan", boyan_N0=100,
                       lambda_=0.)

    domain = ConsumableGridWorldIRL([(7,5), (1,2)],
                                    mapname=maze, 
                                    encodingFunction= lambda x: ConsumableGridWorldIRL.stateVisitEncoding(x,[(7,5)]),
                                    rewardFunction= lambda x,y,z,w: ConsumableGridWorldIRL.tRewardIRL(x,y,z,w,dist,[(7,5)]),
                                    noise=noise)
    
    pdomain = ConsumableGridWorldIRL([(7,5), (1,2)],
                                    mapname=maze, 
                                    encodingFunction= lambda x: ConsumableGridWorldIRL.stateVisitEncoding(x,[(7,5)]),
                                    noise=noise)

    opt["domain"] = domain
    experiment = Experiment(**opt)
    experiment.run(visualize_steps=False,
                   performance_domain = pdomain,
                   visualize_learning=False,
                   visualize_performance=0)
    experiment.save()

    return np.max(experiment.result["return"]),np.sum(experiment.result["return"])

if __name__ == '__main__':
    #print grid_world1_markov()
    #print grid_world1_reward()
    print grid_world1_sliding()
    #print grid_world1_trb()
    #print gridworld1_rirl()
    #print gridworld1_tirl()