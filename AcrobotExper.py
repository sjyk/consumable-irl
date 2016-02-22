#!/usr/bin/env python
"""
Runs experiment with custom domain
"""
__author__ = "Richard Liaw"
import rlpy
from rlpy.Tools import deltaT, clock, hhmmss, getTimeStr
# from .. import visualize_trajectories as visual
import os
import yaml
import shutil
import inspect
import numpy as np
from rlpy.CustomDomains import RCIRL, Encoding, allMarkovReward
import domains


def load_yaml(file_path):
    with open(file_path, 'r') as f:
        ret_val = yaml.load(f)
    return ret_val

def run_experiment_params(param_path='./params.yaml'):
    params = type("Parameters", (), load_yaml(param_path))


    def goalfn(s, goal):
        return -np.cos(s[0]) - np.cos(s[1] + s[0]) > goal

    
    def allMarkovReward(ps,ga, sr, gr):
        r = sr
        last_state = ps[len(ps)-1]
        if any([goalfn(last_state, g) for g in ga]):
            r = gr
        return r


    # # Load domain
    def encode_trial():
        rewards = list(params.domain_params['goalArray'])
        print rewards
        encode = Encoding(rewards, goalfn)
        return encode.strict_encoding

    params.domain_params['goalfn'] = goalfn
    params.domain_params['rewardFunction'] = allMarkovReward
    params.domain_params['encodingFunction'] = encode_trial()
    # params.domain_params['goalArray'] = params.domain_params['goalArray'][::4]
    domain = eval(params.domain)(**params.domain_params) #only for acrobot

    #Load Representation
    representation = eval(params.representation)(
                domain, 
                **params.representation_params)
    policy = eval(params.policy)(
                representation, 
                **params.policy_params)
    agent = eval(params.agent)(
                policy, 
                representation,
                discount_factor=domain.discount_factor, 
                **params.agent_params)

    opt = {}
    opt["exp_id"] = params.exp_id
    opt["path"] = params.results_path + getTimeStr() + "/"
    opt["max_steps"] = params.max_steps
    # opt["max_eps"] = params.max_eps

    opt["num_policy_checks"] = params.num_policy_checks
    opt["checks_per_policy"] = params.checks_per_policy

    opt["domain"] = domain
    opt["agent"] = agent

    if not os.path.exists(opt["path"]):
        os.makedirs(opt["path"])

    shutil.copy(param_path, opt["path"] + "params.yml")
    shutil.copy(inspect.getfile(eval(params.domain)), opt["path"] + "domain.py")
    shutil.copy(inspect.getfile(inspect.currentframe()), opt["path"] + "exper.py")


    return eval(params.experiment)(**opt)


if __name__ == '__main__':
    import sys
    experiment = run_experiment_params(sys.argv[1])
    import ipdb; ipdb.set_trace()
    experiment.run(visualize_steps=0,  # should each learning step be shown?
                   visualize_learning=False,
                   visualize_performance=0)  # show policy / value function?
                   # saveTrajectories=False)  # show performance runs?
    
    # experiment.domain.showLearning(experiment.agent.representation)

    # experiment.plotTrials(save=True)
    # experiment.plot(save=True, x = "learning_episode") #, y="reward")
    experiment.save()

