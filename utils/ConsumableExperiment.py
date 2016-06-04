"""Standard Experiment for Learning Control in RL."""

import logging
from rlpy.Tools import plt
import numpy as np
from copy import deepcopy
import re
import argparse
from rlpy.Tools import deltaT, clock, hhmmss
from rlpy.Tools import className, checkNCreateDirectory
from rlpy.Tools import printClass
import rlpy.Tools.results
from rlpy.Tools import lower
from rlpy.Experiments import Experiment
import os
import rlpy.Tools.ipshell
import json
from collections import defaultdict

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"


class ConsumableExperiment(Experiment):


    def __init__(self, agent, domain, exp_id=1, max_steps=1000,
                 config_logging=True, num_policy_checks=10, log_interval=1,
                 path='Results/Temp',
                 checks_per_policy=1, stat_bins_per_state_dim=0, **kwargs):
        """
        :param agent: the :py:class:`~Agents.Agent.Agent` to use for learning the task.
        :param domain: the problem :py:class:`~Domains.Domain.Domain` to learn
        :param exp_id: ID of this experiment (main seed used for calls to np.rand)
        :param max_steps: Total number of interactions (steps) before experiment termination.

        .. note::
            ``max_steps`` is distinct from ``episodeCap``; ``episodeCap`` defines the
            the largest number of interactions which can occur in a single
            episode / trajectory, while ``max_steps`` limits the sum of all
            interactions over all episodes which can occur in an experiment.

        :param num_policy_checks: Number of Performance Checks uniformly
            scattered along timesteps of the experiment
        :param log_interval: Number of seconds between log prints to console
        :param path: Path to the directory to be used for results storage
            (Results are stored in ``path/output_filename``)
        :param checks_per_policy: defines how many episodes should be run to
            estimate the performance of a single policy

        """
        super(ConsumableExperiment,self).__init__(agent, domain, exp_id, max_steps,config_logging, num_policy_checks, log_interval, path, checks_per_policy, stat_bins_per_state_dim, **kwargs)

    def run(self, performance_domain=None, visualize_performance=0, visualize_learning=False,
            visualize_steps=False, debug_on_sigurg=False):
        """
        Run the experiment and collect statistics / generate the results

        :param visualize_performance: (int)
            determines whether a visualization of the steps taken in
            performance runs are shown. 0 means no visualization is shown.
            A value n > 0 means that only the first n performance runs for a
            specific policy are shown (i.e., for n < checks_per_policy, not all
            performance runs are shown)
        :param visualize_learning: (boolean)
            show some visualization of the learning status before each
            performance evaluation (e.g. Value function)
        :param visualize_steps: (boolean)
            visualize all steps taken during learning
        :param debug_on_sigurg: (boolean)
            if true, the ipdb debugger is opened when the python process
            receives a SIGURG signal. This allows to enter a debugger at any
            time, e.g. to view data interactively or actual debugging.
            The feature works only in Unix systems. The signal can be sent
            with the kill command:

                kill -URG pid

            where pid is the process id of the python interpreter running this
            function.

        """

        if debug_on_sigurg:
            rlpy.Tools.ipshell.ipdb_on_SIGURG()

        if performance_domain == None:
            self.performance_domain = deepcopy(self.domain)
        else:
            self.performance_domain = deepcopy(performance_domain)
            
        self.seed_components()

        self.result = defaultdict(list)
        self.result["seed"] = self.exp_id
        total_steps = 0
        eps_steps = 0
        eps_return = 0
        episode_number = 0

        # show policy or value function of initial policy
        if visualize_learning:
            self.domain.showLearning(self.agent.representation)

        # Used to bound the number of logs in the file
        start_log_time = clock()
        # Used to show the total time took the process
        self.start_time = clock()
        self.elapsed_time = 0
        # do a first evaluation to get the quality of the inital policy
        self.all_experiment_list=[]
        self.evaluate(total_steps, episode_number, visualize_performance)
        self.total_eval_time = 0.
        terminal = True
        curr_experiment_list=[]
        while total_steps < self.max_steps:
            if terminal or eps_steps >= self.domain.episodeCap:
                # if curr_experiment_list!=[]:
                #     self.all_experiment_list.append(curr_experiment_list)
                curr_experiment_list=[]
                s, terminal, p_actions = self.domain.s0()
                a = self.agent.policy.pi(s, terminal, p_actions)
                # Visual
                if visualize_steps:
                    self.domain.show(a, self.agent.representation)

                # Output the current status if certain amount of time has been
                # passed
                eps_return = 0
                eps_steps = 0
                episode_number += 1
            # Act,Step
            curr_experiment_list.append((str(list(s)),str(a)))
            r, ns, terminal, np_actions = self.domain.step(a)

            self._gather_transition_statistics(s, a, ns, r, learning=True)
            na = self.agent.policy.pi(ns, terminal, np_actions)

            total_steps += 1
            eps_steps += 1
            eps_return += r

            # Print Current performance
            if (terminal or eps_steps == self.domain.episodeCap) and deltaT(start_log_time) > self.log_interval:
                start_log_time = clock()
                elapsedTime = deltaT(self.start_time)
                self.logger.info(
                    self.log_template.format(total_steps=total_steps,
                                             elapsed=hhmmss(
                                                 elapsedTime),
                                             remaining=hhmmss(
                                                 elapsedTime * (
                                                     self.max_steps - total_steps) / total_steps),
                                             totreturn=eps_return,
                                             steps=eps_steps,
                                             num_feat=self.agent.representation.features_num))

            # learning
            self.agent.learn(s, p_actions, a, r, ns, np_actions, na, terminal)
            s, a, p_actions = ns, na, np_actions
            # Visual
            if visualize_steps:
                self.domain.show(a, self.agent.representation)

            # Check Performance
            if total_steps % (self.max_steps / self.num_policy_checks) == 0:
                self.elapsed_time = deltaT(
                    self.start_time) - self.total_eval_time

                # show policy or value function
                if visualize_learning:
                    self.domain.showLearning(self.agent.representation)

                self.evaluate(
                    total_steps,
                    episode_number,
                    visualize_performance)
                self.total_eval_time += deltaT(self.start_time) - \
                    self.elapsed_time - \
                    self.total_eval_time
                start_log_time = clock()

        # Visual
        if visualize_steps:
            self.domain.show(a, self.agent.representation)
        self.logger.info("Total Experiment Duration %s" % (hhmmss(deltaT(self.start_time))))

    def evaluate(self, total_steps, episode_number, visualize=0):
        """
        Evaluate the current agent within an experiment

        :param total_steps: (int)
                     number of steps used in learning so far
        :param episode_number: (int)
                        number of episodes used in learning so far
        """
        # TODO resolve this hack
        if className(self.agent) == 'PolicyEvaluation':
            # Policy Evaluation Case
            self.result = self.agent.STATS
            return

        random_state = np.random.get_state()
        #random_state_domain = copy(self.domain.random_state)
        elapsedTime = deltaT(self.start_time)
        performance_return = 0.
        performance_steps = 0.
        performance_term = 0.
        performance_discounted_return = 0.
        runs=[]
        for j in xrange(self.checks_per_policy):
            p_ret, p_step, p_term, p_dret,run = self.performanceRun(
                total_steps, visualize=visualize > j)
            performance_return += p_ret
            performance_steps += p_step
            performance_term += p_term
            performance_discounted_return += p_dret
            runs.append(run)
        if runs!=[]:
            self.all_experiment_list.append(runs)
        performance_return /= self.checks_per_policy
        performance_steps /= self.checks_per_policy
        performance_term /= self.checks_per_policy
        performance_discounted_return /= self.checks_per_policy
        self.result["learning_steps"].append(total_steps)
        self.result["return"].append(performance_return)
        self.result["learning_time"].append(self.elapsed_time)
        self.result["num_features"].append(self.agent.representation.features_num)
        self.result["steps"].append(performance_steps)
        self.result["terminated"].append(performance_term)
        self.result["learning_episode"].append(episode_number)
        self.result["discounted_return"].append(performance_discounted_return)
        # reset start time such that performanceRuns don't count
        self.start_time = clock() - elapsedTime
        if total_steps > 0:
            remaining = hhmmss(
                elapsedTime * (self.max_steps - total_steps) / total_steps)
        else:
            remaining = "?"
        self.logger.info(
            self.performance_log_template.format(total_steps=total_steps,
                                                 elapsed=hhmmss(
                                                     elapsedTime),
                                                 remaining=remaining,
                                                 totreturn=performance_return,
                                                 steps=performance_steps,
                                                 num_feat=self.agent.representation.features_num))

        np.random.set_state(random_state)
        #self.domain.rand_state = random_state_domain

    def performanceRun(self, total_steps, visualize=False):
        """
        Execute a single episode using the current policy to evaluate its
        performance. No exploration or learning is enabled.

        :param total_steps: int
            maximum number of steps of the episode to peform
        :param visualize: boolean, optional
            defines whether to show each step or not (if implemented by the domain)
        """

        # Set Exploration to zero and sample one episode from the domain
        eps_length = 0
        eps_return = 0
        eps_discount_return = 0
        eps_term = 0

        self.agent.policy.turnOffExploration()

        s, eps_term, p_actions = self.performance_domain.s0()
        experiment_list=[]
        while not eps_term and eps_length < self.domain.episodeCap:
            a = self.agent.policy.pi(s, eps_term, p_actions)
            if visualize:
                self.performance_domain.showDomain(a)
            experiment_list.append(list(s))
            # experiment_list.append(str(list(s)+[a]))

            r, ns, eps_term, p_actions = self.performance_domain.step(a)
            self._gather_transition_statistics(s, a, ns, r, learning=False)
            s = ns
            eps_return += r
            eps_discount_return += self.performance_domain.discount_factor ** eps_length * \
                r
            eps_length += 1
        if visualize:
            self.performance_domain.showDomain(a)
        self.agent.policy.turnOnExploration()
        # This hidden state is for domains (such as the noise in the helicopter domain) that include unobservable elements that are evolving over time
        # Ideally the domain should be formulated as a POMDP but we are trying
        # to accomodate them as an MDP

        return eps_return, eps_length, eps_term, eps_discount_return, experiment_list

    def save(self):
        """Saves the experimental results to the ``results.json`` file
        """
        results_fn = os.path.join(self.full_path, self.output_filename)
        # self.result["all_steps"]=self.all_experiment_list
        self.result["walls"]=self.domain.wallArray.tolist()
        if not os.path.exists(self.full_path):
            os.makedirs(self.full_path)
        with open(results_fn, "w") as f:
            json.dump(self.result, f, indent=4, sort_keys=True)

    def load(self):
        """loads the experimental results from the ``results.txt`` file
        If the results could not be found, the function returns ``None``
        and the results array otherwise.
        """
        results_fn = os.path.join(self.full_path, self.output_filename)
        self.results = rlpy.Tools.results.load_single(results_fn)
        return self.results

    def plot(self, y="return", x="learning_steps", save=False):
        """Plots the performance of the experiment
        This function has only limited capabilities.
        For more advanced plotting of results consider
        :py:class:`Tools.Merger.Merger`.
        """
        labels = rlpy.Tools.results.default_labels
        performance_fig = plt.figure("Performance")
        res = self.result
        plt.plot(res[x], res[y], '-bo', lw=3, markersize=10)
        plt.xlim(0, res[x][-1] * 1.01)
        y_arr = np.array(res[y])
        m = y_arr.min()
        M = y_arr.max()
        delta = M - m
        if delta > 0:
            plt.ylim(m - .1 * delta - .1, M + .1 * delta + .1)
        xlabel = labels[x] if x in labels else x
        ylabel = labels[y] if y in labels else y
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        if save:
            path = os.path.join(
                self.full_path,
                "{:3}-performance.pdf".format(self.exp_id))
            performance_fig.savefig(path, transparent=True, pad_inches=.1)
        plt.ioff()
        plt.show()

    def compile_path(self, path):
        """
        An experiment path can be specified with placeholders. For
        example, ``Results/Temp/{domain}/{agent}/{representation}``.
        This functions replaces the placeholders with actual values.
        """
        variables = re.findall("{([^}]*)}", path)
        replacements = {}
        for v in variables:
            if lower(v).startswith('representation') or lower(v).startswith('policy'):
                obj = 'self.agent.' + v
            else:
                obj = 'self.' + v

            if len([x for x in ['self.domain', 'self.agent', 'self.agent.policy', 'self.agent.representation'] if x == lower(obj)]):
                replacements[v] = eval('className(%s)' % obj)
            else:
                try:
                    replacements[v] = str(eval('%s' % v))
                except:
                    print "Warning: Could not interpret path variable", repr(v)

        return path.format(**replacements)
