"""
This class implements an experimental framework for 
executing gridworld experiments
"""
import os,sys,inspect 
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

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
from copy import deepcopy


class GridWorld(object):

    def __init__(self,
                 path="./Results/gridworld1", 
                 max_steps=150000,
                 noise = 0.1,
                 exp = 0.3,
                 discretization = 400,
                 num_policy_checks=20,
                 checks_per_policy=10,
                 mapname='10x7-ACC2011.txt',
                 consumable=[(7,5), (1,2)]):

        self.opt_template = {}
        self.opt_template["path"] = path
        self.opt_template["checks_per_policy"] = checks_per_policy
        self.opt_template["max_steps"] = max_steps
        self.opt_template["num_policy_checks"] = num_policy_checks

        self.env_template = {}
        self.env_template["noise"] = noise
        self.env_template["exp"] = exp
        self.env_template["discretization"] = discretization
        self.env_template["map"] = os.path.join(ConsumableGridWorld.default_map_dir, mapname)
        self.env_template["consumable"] = consumable

    ##Helper methods to create the various domains
    
    def createMarkovDomain(self, k=1, rewardFunction=None):
        return ConsumableGridWorldIRL(self.env_template["consumable"], 
                               mapname=self.env_template["map"], 
                               encodingFunction= lambda x: ConsumableGridWorldIRL.allMarkovEncoding(x,k), 
                               rewardFunction = rewardFunction,
                               noise=self.env_template["noise"],
                               binary=True)

    def createStateDomain(self,waypoints, rewardFunction=None):
        return ConsumableGridWorldIRL(self.env_template["consumable"], 
                                    mapname=self.env_template["map"], 
                                    encodingFunction= lambda x: ConsumableGridWorldIRL.stateVisitEncoding(x,waypoints), 
                                    rewardFunction = rewardFunction,
                                    noise=self.env_template["noise"],
                                    binary=True)

    def createSlidingDomain(self,k):
        return ConsumableGridWorldIRL(self.env_template["consumable"], 
                                    mapname=self.env_template["map"], 
                                    encodingFunction= lambda x: ConsumableGridWorldIRL.slidingWindowEncoding(x,k), 
                                    noise=self.env_template["noise"])

    def createPassageDomain(self,selfwaypoints,k=5):
        return ConsumableGridWorldIRL(self.env_template["consumable"], 
                                    mapname=self.env_template["map"], 
                                    encodingFunction= lambda x: ConsumableGridWorldIRL.statePassageEncoding(x,waypoints,k), 
                                    noise=self.env_template["noise"],
                                    binary=True)

    def getTSCWaypoints(self,N=5, w=2, pruning=0.5):
        sd = self.createStateDomain(self.env_template["consumable"])
        representation = IncrementalTabular(sd, discretization=self.env_template["discretization"])
        policy = eGreedy(representation, epsilon=self.env_template["exp"])
        d = GoalPathPlanner(sd,representation,policy, steps=self.opt_template["max_steps"])
        trajs = d.generateTrajectories(N=N)
        a = TransitionStateClustering(window_size=w)
        
        for t in trajs:
            N = len(t)
            demo = np.zeros((N,2))
            for i in range(0,N):
                demo[i,:] = t[i][0:2]
            a.addDemonstration(demo)

        a.fit(normalize=False, pruning=pruning)
        dist = calculateStateDist(np.shape(sd.map), trajs) 
        return discrete2DClustersToPoints(a.model, dist, radius=1)

    def getIRLDist(self,N=5, rand=False):
        sd = self.createStateDomain(self.env_template["consumable"])
        representation = IncrementalTabular(sd, discretization=self.env_template["discretization"])
        policy = eGreedy(representation, epsilon=self.env_template["exp"])
        
        if not rand:
            d = GoalPathPlanner(sd,representation,policy, steps=self.opt_template["max_steps"])
        else:
            d = GoalPathPlanner(sd,representation,policy, steps=100)

        trajs = d.generateTrajectories(N=N)
        return calculateStateDist(np.shape(sd.map), trajs) 

    def getIRLTDist(self,waypoints,N=5, rand=False):
        sd = self.createStateDomain(self.env_template["consumable"])
        representation = IncrementalTabular(sd, discretization=self.env_template["discretization"])
        policy = eGreedy(representation, epsilon=self.env_template["exp"])
        
        if not rand:
            d = GoalPathPlanner(sd,representation,policy, steps=self.opt_template["max_steps"])
        else:
            d = GoalPathPlanner(sd,representation,policy, steps=100)

        trajs = d.generateTrajectories(N=N)
        return calculateStateTemporalDist(np.shape(sd.map), trajs, waypoints) 

    def runRewardVisit(self):
        opt = deepcopy(self.opt_template)
        domain = self.createStateDomain(self.env_template["consumable"])
        opt["domain"] = domain
        representation = IncrementalTabular(domain, discretization=self.env_template["discretization"])
        policy = eGreedy(representation, epsilon=self.env_template["exp"])
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

    def runMarkov(self):
        opt = deepcopy(self.opt_template)
        domain = self.createMarkovDomain()
        opt["domain"] = domain
        representation = IncrementalTabular(domain, discretization=self.env_template["discretization"])
        policy = eGreedy(representation, epsilon=self.env_template["exp"])
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

    def runSliding(self,k=3):
        opt = deepcopy(self.opt_template)
        domain = self.createSlidingDomain(k)
        opt["domain"] = domain
        representation = IncrementalTabular(domain, discretization=self.env_template["discretization"])
        policy = eGreedy(representation, epsilon=self.env_template["exp"])
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

    def runTSCVisit(self,N=5, w=2, pruning=0.5):
        opt = deepcopy(self.opt_template)
        ac = self.getTSCWaypoints(N, w, pruning)
        domain = self.createStateDomain(ac)
        opt["domain"] = domain
        representation = IncrementalTabular(domain, discretization=self.env_template["discretization"])
        policy = eGreedy(representation, epsilon=self.env_template["exp"])
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

    def runIRL(self,N=5):
        opt = deepcopy(self.opt_template)

        dist = self.getIRLDist(N=N)
        bdist = self.getIRLDist(N=N,rand=True)

        #print dist-bdist

        domain = self.createMarkovDomain(rewardFunction= lambda x,y,z,w: ConsumableGridWorldIRL.maxEntReward(x,y,z,w,dist-bdist))
        opt["domain"] = domain

        representation = IncrementalTabular(domain, discretization=self.env_template["discretization"])
        policy = eGreedy(representation, epsilon=self.env_template["exp"])
        opt["agent"] = Q_Learning(representation=representation, policy=policy,
                       discount_factor=domain.discount_factor,
                       initial_learn_rate=0.1,
                       learn_rate_decay_mode="boyan", boyan_N0=100,
                       lambda_=0.)

        performance_domain = self.createMarkovDomain()

        experiment = Experiment(**opt)
        experiment.run(visualize_steps=False,
                       performance_domain = performance_domain,
                       visualize_learning=False,
                       visualize_performance=0)
        experiment.save()
        
        return np.max(experiment.result["return"]),np.sum(experiment.result["return"])

    def runRewardIRL(self,N=5):
        opt = deepcopy(self.opt_template)
        dist = self.getIRLTDist(self.env_template["consumable"],N=N)
        bdist = self.getIRLDist(N=N, rand=True)
        dist = [d-bdist for d in dist]

        print dist

        domain = self.createStateDomain(waypoints=self.env_template["consumable"],
                                        rewardFunction=lambda x,y,z,w: ConsumableGridWorldIRL.rewardIRL(x,y,z,w,dist,self.env_template["consumable"]))
        
        opt["domain"] = domain
        representation = IncrementalTabular(domain, discretization=self.env_template["discretization"])
        policy = eGreedy(representation, epsilon=self.env_template["exp"])
        opt["agent"] = Q_Learning(representation=representation, policy=policy,
                       discount_factor=domain.discount_factor,
                       initial_learn_rate=0.1,
                       learn_rate_decay_mode="boyan", boyan_N0=100,
                       lambda_=0.)

        experiment = Experiment(**opt)
        experiment.run(visualize_steps=False,
                       performance_domain = self.createStateDomain(self.env_template["consumable"]),
                       visualize_learning=False,
                       visualize_performance=0)
        experiment.save()

        
        return np.max(experiment.result["return"]),np.sum(experiment.result["return"])

    def runTIRL(self,N=5, w=2, pruning=0.5):
        opt = deepcopy(self.opt_template)
        dist = self.getIRLDist(N=N)
        ac = self.getTSCWaypoints(N, w, pruning)
        domain = self.createStateDomain(waypoints=ac, 
                                        rewardFunction=lambda x,y,z,w: ConsumableGridWorldIRL.rewardIRL(x,y,z,w,dist))
        opt["domain"] = domain
        representation = IncrementalTabular(domain, discretization=self.env_template["discretization"])
        policy = eGreedy(representation, epsilon=self.env_template["exp"])
        opt["agent"] = Q_Learning(representation=representation, policy=policy,
                       discount_factor=domain.discount_factor,
                       initial_learn_rate=0.1,
                       learn_rate_decay_mode="boyan", boyan_N0=100,
                       lambda_=0.)

        experiment = Experiment(**opt)
        experiment.run(visualize_steps=False,
                       performance_domain = self.createStateDomain(waypoints=self.env_template["consumable"]),
                       visualize_learning=False,
                       visualize_performance=0)
        experiment.save()
        
        return np.max(experiment.result["return"]),np.sum(experiment.result["return"])

if __name__ == '__main__':
    #print grid_world1_markov()
    #print grid_world1_reward()
    #print grid_world1_sliding()
    #print grid_world1_trb()
    #print gridworld1_rirl()
    #print gridworld1_tirl()
    g = GridWorld(max_steps=100000,mapname="10x10-12ftml.txt", consumable=[(0,9)])
    g.runRewardVisit()