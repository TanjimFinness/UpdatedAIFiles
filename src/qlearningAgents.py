# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random, util, math
import numpy as np


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        # "*** YOUR CODE HERE ***"
        # self.q_table = np.zeros([])
        self.q_table = dict()

    # ok
    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        # "*** YOUR CODE HERE ***"

        # TODO:  Computar Q(state,action)

        q_value = self.q_table.get((state, action))
        if q_value is None:
            q_value = 0
            self.q_table.update([((state, action), q_value)])
        return q_value

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no leglsal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        # "*** YOUR CODE HERE ***"
        # TODO: V(s) = max_{a in actions} Q(s,a)

        best_value = 0.

        actions = self.getLegalActions(state)
        if len(actions) == 0:
            return 0.0

        for action in actions:
            if self.getQValue(state, action) > best_value:
                best_value = self.getQValue(state, action)
        #print(f'best_value: {best_value}')
        return best_value


    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        # "*** YOUR CODE HERE ***"
        # TODO: computar policy(s) = arg_max_{a in actions} Q(s,a)
        # Em caso de valores iguais, escolhe aleatorio

        legalActions = self.getLegalActions(state)
        best_action = None
        best_action_val = -9999

        if len(legalActions) != 0:
            random.shuffle(legalActions)
            for act in legalActions:
                act_val = self.getQValue(state, act)
                if act_val is None:
                    act_val = 0.

                if act_val > best_action_val:
                    best_action_val = act_val
                    best_action = act

                if act_val == best_action_val:
                    choice = random.choice([act, best_action])
                    if choice == act:
                        best_action = act
                        best_action_val = act_val

        return best_action


    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        if util.flipCoin(self.epsilon):
            return random.choice(self.getLegalActions(state))
        action = self.getPolicy(state)
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        # "*** YOUR CODE HERE ***"
        #print(state, action, nextState, reward)
        q_old = self.getQValue(state, action)
        opt_future_value_estimate = self.computeValueFromQValues(nextState)
        learned = reward + self.discount * opt_future_value_estimate

        #print(f'q_old: {q_old} | reward: {reward} | opt_future: {opt_future_value_estimate} | learned : {learned}')

        q_updated = (1 - self.alpha) * q_old + self.alpha * learned
        q_updated = q_old + self.alpha * (learned - q_old)
        #print(f'q_updated : {q_updated}')

        self.q_table.update([((state, action), q_updated)])
        #print(f'updated :: q_value : state {state} | action {action} | nextState {nextState} : {q_updated}')
        #print(f'q_table :: \n{self.q_table}')


    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
