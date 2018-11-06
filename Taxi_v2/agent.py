import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA = 6, gamma=0.999, alpha = 0.06, seed = 0):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = 1
        self.gamma = gamma
        self.alpha = alpha
        np.random.seed(seed)

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if np.random.rand() > self.epsilon:
        	return np.argmax(self.Q[state])
        else:
        	return np.random.choice(self.nA)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        Qsa = self.Q[state][action] 
        self.Q[state][action] = Qsa + (self.alpha * (reward + (self.gamma \
                                   * np.max(self.Q[next_state])) - Qsa))   
        self.update_epsilon()

    def update_epsilon(self, decay = 0.99998, min_eps = 0.0001):
        """ obtains the action probabilities corresponding to epsilon-greedy policy """
        self.epsilon = max(self.epsilon*decay, min_eps)
