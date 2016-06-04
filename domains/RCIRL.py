"""
This class implements a GridWorld with
consumable rewards
"""
import os,sys,inspect
from rlpy.Tools import __rlpy_location__, findElemArray1D, perms
from rlpy.Domains.Domain import Domain
from ConsumableGridWorldIRL import ConsumableGridWorldIRL
from rlpy.Tools import plt, bound, wrap, mpatches, id2vec
import matplotlib as mpl
from copy import deepcopy
import numpy as np
import pickle
from rlpy.Tools import plt, FONTSIZE, linearMap  
import math

allMarkovEncoding = ConsumableGridWorldIRL.allMarkovEncoding

class RCIRL(Domain): 
    # #default paths
    # currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    # default_map_dir = os.path.join(currentdir,"ConsumableGridWorldMaps")

    actions_num = 9
    state_space_dims = 4
    continuous_dims = np.arange(state_space_dims)

    ROOM_WIDTH = 3  # in meters
    ROOM_HEIGHT = 2  # in meters
    XMIN = -ROOM_WIDTH / 2.0
    XMAX = ROOM_WIDTH / 2.0
    YMIN = -ROOM_HEIGHT / 2.0
    YMAX = ROOM_HEIGHT / 2.0
    ACCELERATION = .1
    TURN_ANGLE = np.pi / 6
    SPEEDMIN = -.3
    SPEEDMAX = .3
    HEADINGMIN = -np.pi
    HEADINGMAX = np.pi
    INIT_STATE = np.array([0.0, 0.0, 0.0, 0.0])
    STEP_REWARD = -1
    GOAL_REWARD = 50
    # GOAL = [.5, .5]
    GOAL_RADIUS = .15
    HEADBOUND = 0.10000
    SPEEDBOUND = 0.02

    actions = np.outer([-1, 0, 1], [-1, 0, 1])
    discount_factor = .9
    episodeCap = 10000
    delta_t = .3  # time between steps
    CAR_LENGTH = .3  # L on the webpage
    CAR_WIDTH = .15
    # The location of rear wheels if the car facing right with heading 0
    REAR_WHEEL_RELATIVE_LOC = .05
    # Used for visual stuff:
    domain_fig = None
    X_discretization = 20
    Y_discretization = 20
    SPEED_discretization = 5
    HEADING_discretization = 3
    ARROW_LENGTH = .2
    car_fig = None
    wallArray = None


    # #an encoding function maps a set of previous states to a fixed
    # #width encoding
    # """
    # A reward function is a function over ALL of the previous states, 
    # the set of goal states,
    # the step reward constant, 
    # and the goal reward constant.
    # """
    def __init__(self, 
                 goalArray, 
                 wallArray=None,
                 encodingFunction=allMarkovEncoding, # TODO
                 rewardFunction=None,
                 goalfn=lambda state, goal: state == goal,
                 episodeCap=None,
                 goal_radius=None, 
                 headbound=None,
                 noise=.1,
                 step_reward=None):
        self.debug = False
        #setup consumable rewards
        self.statespace_limits = np.array(
            [[self.XMIN,
              self.XMAX],
             [self.YMIN,
              self.YMAX],
                [self.SPEEDMIN,
                 self.SPEEDMAX],
                [self.HEADINGMIN,
                 self.HEADINGMAX]])

        self.goalArray0 = np.array(goalArray)
        self.goalArray = np.array(goalArray)
        self.prev_states = []

        self.encodingFunction = encodingFunction
        self.rewardFunction = rewardFunction
        self.goalfn = goalfn

        # should convert to bins? or leave discrete
        self.start_state = self.augment_state(self.INIT_STATE)

        if noise:
            self.NOISE = noise
        else:
            self.NOISE = 0

        if wallArray is not None:
            for elem in wallArray: # (left corner) x, y, dx, dy
                assert len(elem) == 4
            self.wallArray = np.array(wallArray)
        if episodeCap:
            self.episodeCap = episodeCap
        if goal_radius:
            self.GOAL_RADIUS = goal_radius
        if headbound:
            self.HEADBOUND = headbound
        if step_reward:
            self.STEP_REWARD = step_reward
        #remove goals for existing maps

        # set given goals

        encodingLimits = []
        for i in range(0,len(self.encodingFunction(self.prev_states))):
            encodingLimits.append([0,1])

        self.statespace_limits = np.vstack((self.statespace_limits, encodingLimits))
        self.state_space_dims = len(self.statespace_limits)
        continuous_dims = np.arange(self.state_space_dims)

        self.DimNames = ["Dim: "+str(k) for k in range(0,2+len(self.encodingFunction(self.prev_states)))]

        self.terminal_state = []

        super(RCIRL, self).__init__()

    #def possibleActions(self):
    #    return np.array([0, 1, 3, 4, 6, 7])

    def step(self, a):
        r = self.STEP_REWARD
        ga = self.goalArray.copy()

        if self.random_state.random_sample() < self.NOISE:
            # Random Move
            a = self.random_state.choice(self.possibleActions())

    #     # Take action
    #     statesize = np.shape(self.state)
    #     actionsize = np.shape(self.ACTIONS[a])
    #    # print statesize[0]-actionsize[0]

        ns = self.augment_state(self.simulate_step(self.state[:4], a))
        # print ns
        # if any(ns):
            # import ipdb; ipdb.set_trace()

        self.prev_states.append(ns[:4])
    #     #print ns[0], ns[1],ga[0][0],ga[0][1]

        terminal = self.isTerminal() # TODO: Check - get terminal after?

        if self.bumped(self.state):
            print "Bumped"

        if self.collided(self.state) or self.bumped(self.state):
            # r = (self.episodeCap - len(self.prev_states)) * self.STEP_REWARD
            r = -self.episodeCap*(-self.STEP_REWARD) #make sure that the car does not get rewarded for colliding
            terminal = True

        if not terminal and self.at_goal(ns,self.state,ga[0]): 
            r = self.GOAL_REWARD #(len(self.goalArray0) - len(self.goalArray)) ** 1.3
            self.goalArray = ga[1:]
            if self.debug:
                print "New Goal Reached!", ns, r, len(self.prev_states)

        if self.rewardFunction != None:
            r = self.rewardFunction(self.prev_states, 
                                    self.goalArray, 
                                    self.STEP_REWARD, 
                                    self.GOAL_REWARD,
                                    self.collided(self.state))
        
        ## DEBUG
        if len(self.prev_states) == 11:
            self.terminal_state.append(self.state)   
        ######### 

        self.state = ns.copy()

        return r, ns, terminal, self.possibleActions()

    def step_dx(self, dx):
        r = self.STEP_REWARD
        ga = self.goalArray.copy()

        if self.random_state.random_sample() < self.NOISE:
            # Random Move
            a = self.random_state.choice(self.possibleActions())

    #     # Take action
    #     statesize = np.shape(self.state)
    #     actionsize = np.shape(self.ACTIONS[a])
    #    # print statesize[0]-actionsize[0]

        ns = self.augment_state(self.state[:4]+dx)
        # print ns
        # if any(ns):
            # import ipdb; ipdb.set_trace()

        self.prev_states.append(ns[:4])
    #     #print ns[0], ns[1],ga[0][0],ga[0][1]

        terminal = self.isTerminal() # TODO: Check - get terminal after?

        if self.bumped(self.state):
            print "Bumped"

        if self.collided(self.state) or self.bumped(self.state):
            # r = (self.episodeCap - len(self.prev_states)) * self.STEP_REWARD
            r = -self.episodeCap*(-self.STEP_REWARD) #make sure that the car does not get rewarded for colliding
            terminal = True

        if not terminal and self.at_goal(ns,self.state,ga[0]): 
            r = self.GOAL_REWARD #(len(self.goalArray0) - len(self.goalArray)) ** 1.3
            self.goalArray = ga[1:]
            if self.debug:
                print "New Goal Reached!", ns, r, len(self.prev_states)

        if self.rewardFunction != None:
            r = self.rewardFunction(self.prev_states, 
                                    self.goalArray, 
                                    self.STEP_REWARD, 
                                    self.GOAL_REWARD,
                                    self.collided(self.state))
        
        ## DEBUG
        if len(self.prev_states) == 11:
            self.terminal_state.append(self.state)   
        ######### 

        self.state = ns.copy()

        return r, ns, terminal, self.possibleActions()

    def augment_state(self, state):
        return np.concatenate((state, 
                            self.encodingFunction(self.prev_states)))

    def collided(self, state):
        x, y = state[:2]
        return x == self.XMIN or x == self.XMAX or y == self.YMIN or y == self.YMAX

    def bumped(self, state):
        if self.wallArray is None:
            return False
        # assumes the box will be big enough such that it won't be skipped
        x, y = state[:2]
        for wallx, wally, dx, dy in self.wallArray:
            left =  wallx - self.CAR_WIDTH
            right =  wallx + dx + self.CAR_WIDTH
            bottom =  wally - self.CAR_WIDTH
            top =  wally + dy + self.CAR_WIDTH

            if ((left <= x <= right) and
                (bottom <= y <= top)):
                return True
        return False

    def simulate_step(self, state, a):
        x, y, speed, heading = state

        # Map a number between [0,8] to a pair. The first element is
        # acceleration direction. The second one is the indicator for the wheel
        acc, turn = id2vec(a, [3, 3])
        acc -= 1                # Mapping acc to [-1, 0 1]
        turn -= 1                # Mapping turn to [-1, 0 1]

        # Calculate next state
        nx = x + speed * np.cos(heading) * self.delta_t
        ny = y + speed * np.sin(heading) * self.delta_t
        nspeed = speed + acc * self.ACCELERATION * self.delta_t
        nheading    = heading + speed / self.CAR_LENGTH * \
            np.tan(turn * self.TURN_ANGLE) * self.delta_t

        # Bound values
        nx = bound(nx, self.XMIN, self.XMAX)
        ny = bound(ny, self.YMIN, self.YMAX)
        nspeed = bound(nspeed, self.SPEEDMIN, self.SPEEDMAX)
        nheading = wrap(nheading, self.HEADINGMIN, self.HEADINGMAX)

        # Collision to wall => set the speed to zero
        if nx == self.XMIN or nx == self.XMAX or ny == self.YMIN or ny == self.YMAX:
            nspeed = 0

        return np.array([nx, ny, nspeed, nheading])

    def s0(self):

        if self.debug:
            print "#############################"
            print "######## NEW TRIAL ##########"
            print "#############################"

        self.prev_states = []
        self.state = self.augment_state(self.INIT_STATE.copy())
        self.goalArray = np.array(self.goalArray0)
        
        return self.state.copy(), self.isTerminal(), self.possibleActions()

    def isTerminal(self, s=None):
        # TODO
        # termination condition should include max_steps?
        if s is None:
            s = self.state
        if len(self.goalArray) == 0:
            return True

        return False

    def at_goal(self, state=None, oldstate=None, goal=None):
         """Check if current state is at goal"""
         state = state if state is not None else self.state
         oldstate = oldstate if oldstate is not None else self.state
         goal = goal if goal is not None else self.GOAL

         if state[0] == oldstate[0] and state[1] == oldstate[1]:
            return (np.linalg.norm(goal[:2] - state[:2]) < self.GOAL_RADIUS)

         #circle line intersection
         c = self.dist(oldstate[0],oldstate[1],state[0],state[1],goal[0],goal[1])
         #print c, oldstate[:2],state[:2]

         
         return (np.linalg.norm(c - goal[:2]) < self.GOAL_RADIUS)
             #and (abs(state[2] - goal[2]) < self.SPEEDBOUND)
             #and abs(state[3] - goal[3]) < self.HEADBOUND)

    def dist(self,x1,y1, x2,y2, x3,y3): # x3,y3 is the point
        px = x2-x1
        py = y2-y1
    
        s = px*px + py*py
    
        u =  ((x3 - x1) * px + (y3 - y1) * py) / s
    
        if u > 1:
           u = 1
        elif u < 0:
           u = 0
    
        x = x1 + u * px
        y = y1 + u * py
    
        dx = x - x3
        dy = y - y3
    
        return [x,y]


    def showDomain(self, a):
        s = self.state
        # Plot the car
        x, y, speed, heading = s[:4]
        car_xmin = x - self.REAR_WHEEL_RELATIVE_LOC
        car_ymin = y - self.CAR_WIDTH / 2.
        if self.domain_fig is None:  # Need to initialize the figure
            self.domain_fig = plt.figure()
            if self.wallArray is not None:
                for xmin, ymin, dx, dy in self.wallArray:
                    plt.gca().add_patch(
                        mpatches.Rectangle(
                            [xmin,
                             ymin],
                            dx,
                            dy,
                            color='b',
                            alpha=.4)
                    )

            # Goal
            for goal in self.goalArray:
                plt.gca(
                ).add_patch(
                    plt.Circle(
                        goal[:2],
                        radius=self.GOAL_RADIUS,
                        color='g',
                        alpha=.4))
                plt.xlim([self.XMIN, self.XMAX])
                plt.ylim([self.YMIN, self.YMAX])
                plt.gca().set_aspect('1')
        # Car
        if self.car_fig is not None:
            plt.gca().patches.remove(self.car_fig)

        self.car_fig = mpatches.Rectangle(
            [car_xmin,
             car_ymin],
            self.CAR_LENGTH,
            self.CAR_WIDTH,
            alpha=.4)
        rotation = mpl.transforms.Affine2D().rotate_deg_around(
            x, y, heading * 180 / np.pi) + plt.gca().transData
        self.car_fig.set_transform(rotation)
        plt.gca().add_patch(self.car_fig)

        plt.draw()

    """
    Popular reward functions that you could use
    """
    @staticmethod
    def rcreward(ps, ga, sr, gr, collided):
        last_state = ps[len(ps)-1]
        if len(ga) == 0:
            return 1e6

        if collided:
            return -1e6

        return -np.linalg.norm(last_state[0:2]-ga[0][0:2])#+10*sr
