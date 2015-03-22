# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        closestGhost = min([manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])
        distance2Ghost = -10 / closestGhost if closestGhost else -1000
        foodList = newFood.asList()
        closestFood = min([manhattanDistance(newPos, food) for food in foodList]) if foodList else 0

        return -2 * closestFood + distance2Ghost - 50 * len(foodList)


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def adSearch(state, agentIdx, ply):

            if agentIdx == state.getNumAgents():

                if ply == self.depth:
                    return self.evaluationFunction(state)
                else:
                    return adSearch(state, 0, ply + 1)

            else:
                legalMoves = state.getLegalActions(agentIdx)

                if len(legalMoves) == 0:
                    return self.evaluationFunction(state)

                next = ( adSearch(state.generateSuccessor(agentIdx, m), agentIdx + 1, ply) for m in legalMoves)

                return max(next) if agentIdx == 0 else min(next)


        result = max(gameState.getLegalActions(0), key = lambda x: adSearch(gameState.generateSuccessor(0, x), 1, 1))
        #print result
        return result


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """

        def ABSearch(state):
            value, bestAction = None, None
            a, b = None, None

            for action in state.getLegalActions(0):
                value = max(value,
                            minValue(state.generateSuccessor(0, action), 1,1, a, b)
                )

                if a is None:
                    a = value
                    bestAction = action
                else:
                    a, bestAction = max(value, a), action if value > a else bestAction
            return bestAction

        def minValue(state, agentIdx, ply, a, b):

            if agentIdx == state.getNumAgents():
                return maxValue(state, 0, ply + 1, a, b)

            value = None

            for action in state.getLegalActions(agentIdx):
                succ = minValue(state.generateSuccessor(agentIdx, action),
                                agentIdx + 1,
                                ply,
                                a,
                                b)
                value = succ if value is None else min(value, succ)
                if a is not None and value < a:
                    return value

                b = value if b is None else min(b, value)

            if value is not None:
                return value
            else:
                return self.evaluationFunction(state)


        def maxValue(state, agentIdx, ply, a, b):
            if ply > self.depth:
                return self.evaluationFunction(state)

            value = None

            for action in state.getLegalActions(agentIdx):
                succ = minValue(state.generateSuccessor(agentIdx, action),
                                agentIdx + 1,
                                ply,
                                a,
                                b)
                value = max(value, succ)
                if b is not None and value > b:
                    return value
                a = max(a, value)

            if value is not None:
                return value
            else:
                return self.evaluationFunction(state)

        action = ABSearch(gameState)

        return action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def adSearch(state, agentIdx, ply):

            if agentIdx == state.getNumAgents():

                if ply == self.depth:
                    return self.evaluationFunction(state)
                else:
                    return adSearch(state, 0, ply + 1)

            else:
                legalMoves = state.getLegalActions(agentIdx)

                if len(legalMoves) == 0:
                    return self.evaluationFunction(state)

                next = ( adSearch(state.generateSuccessor(agentIdx, m), agentIdx + 1, ply) for m in legalMoves)

                if agentIdx == 0:
                    return max(next)
                else:
                    l = list(next)
                    return sum(l) / len(l)



        result = max(gameState.getLegalActions(0), key = lambda x: adSearch(gameState.generateSuccessor(0, x), 1, 1))

        return result


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: it`s code from q1 with small modification
    """
    "*** YOUR CODE HERE ***"

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newCapsules = currentGameState.getCapsules()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    closestGhost = min([manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])
    closestCapsule = min([manhattanDistance(newPos, caps) for caps in newCapsules]) if newCapsules else 0
    distance2Capsule = -3 / closestCapsule if  closestCapsule else 100
    distance2Ghost = -2 / closestGhost if closestGhost else -500
    foodList = newFood.asList()
    closestFood = min([manhattanDistance(newPos, food) for food in foodList]) if foodList else 0

    return -2 * closestFood  + distance2Ghost - 10 * len(foodList) + distance2Capsule

# Abbreviation
better = betterEvaluationFunction

