"""
This class implements a path planner for rlpy
environments. It does so by rolling out
trajectories of policies
"""


__author__ = "Robert H. Klein"

from domains.ConsumableGridWorld import ConsumableGridWorld
from domains.ConsumableGridWorldIRL import ConsumableGridWorldIRL
from mp.goalpaths import GoalPathPlanner
from rlpy.Domains import Pinball
from rlpy.Agents import Q_Learning
from rlpy.Representations import Tabular
from rlpy.Policies import eGreedy
from rlpy.Experiments import Experiment
#from rlpy.Tools import __rlpy_location__, findElemArray1D, perms
import os


def make_experiment(exp_id=2, path="./Results/Tutorial/gridworld-qlearning"):
    """
    Each file specifying an experimental setup should contain a
    make_experiment function which returns an instance of the Experiment
    class with everything set up.

    @param id: number used to seed the random number generators
    @param path: output directory where logs and results are stored
    """
    opt = {}
    opt["exp_id"] = exp_id
    opt["path"] = path

    # Domain:
    maze = os.path.join(ConsumableGridWorld.default_map_dir, '11x11-Rooms.txt')
    domain = ConsumableGridWorldIRL([(3,8), (1,3)],mapname=maze, encodingFunction= lambda x: ConsumableGridWorldIRL.allMarkovEncoding(x), noise=0.3)
    #domain = Pinball(noise=0.3)
    opt["domain"] = domain

    # Representation
    representation = Tabular(domain, discretization=400)

    # Policy
    policy = eGreedy(representation, epsilon=0.3)

    # Agent
    opt["agent"] = Q_Learning(representation=representation, policy=policy,
                       discount_factor=domain.discount_factor,
                       initial_learn_rate=0.1,
                       learn_rate_decay_mode="boyan", boyan_N0=100,
                       lambda_=0.)
    
    opt["checks_per_policy"] = 10
    opt["max_steps"] = 10000
    opt["num_policy_checks"] = 20
    experiment = Experiment(**opt)

    d = GoalPathPlanner(domain, representation,policy)
    print d.generateTrajectories(N=10)
   
    """
    opt["checks_per_policy"] = 10
    opt["max_steps"] = 100
    opt["num_policy_checks"] = 10
    experiment = Experiment(**opt)
    """
    
    return experiment

if __name__ == '__main__':
    experiment = make_experiment(1)
    #experiment.run(visualize_steps=False,  # should each learning step be shown?
    #               visualize_learning=False,  # show policy / value function?
    #               visualize_performance=1)  # show performance runs?
    #experiment.plot()
    #experiment.save()
    
