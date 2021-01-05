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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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
        score = successorGameState.getScore()
        food = currentGameState.getFood().asList()
        closest_food = 0
        if len(food) > 0:
            closest_food = (min(manhattanDistance(newPos, f) for f in food))
        score -= closest_food
        if len(newGhostStates) > 0:
            closest_ghost = 0
            for i in range(len(newGhostStates)):
                ghost_distance = manhattanDistance(newGhostStates[i].getPosition(), newPos)
                if newScaredTimes[i] > ghost_distance:
                    closest_ghost = -ghost_distance
                elif ghost_distance <= 1:
                    return -999999
                if closest_ghost == 0 or closest_ghost > ghost_distance:
                    closest_ghost = ghost_distance
        score -= 1/closest_ghost
        if closest_food == 0:
            score += 1000
        elif closest_food == 1:
            score += 100
        return score

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
        best_move = self.MiniMax(gameState, 0, 0)[1]
        return best_move

    def MiniMax(self, gameState, agentIndex, depth):
        best_move = None
        if (depth == self.depth) or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), best_move
        if agentIndex == 0:
            value = float("-inf")
        else:
            value = float("inf")
        for action in gameState.getLegalActions(agentIndex):
            nxt_pos = gameState.generateSuccessor(agentIndex, action)
            if agentIndex == 0:
                nxt_val, nxt_move = self.MiniMax(nxt_pos, 1, depth)
                if value < nxt_val:
                    value, best_move = nxt_val, action
            elif agentIndex == gameState.getNumAgents() - 1:
                nxt_val, nxt_move = self.MiniMax(nxt_pos, 0, depth+1)
                if value > nxt_val:
                    value, best_move = nxt_val, action
            else:
                nxt_val, nxt_move = self.MiniMax(nxt_pos, agentIndex+1, depth)
                if value > nxt_val:
                    value, best_move = nxt_val, action
        return value, best_move

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        best_move = self.alphaBeta(gameState, 0, 0, float("-inf"), float("inf"))[1]
        return best_move

    def alphaBeta(self, gameState, agentIndex, depth, alpha, beta):
        best_move = None
        if (depth == self.depth) or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), best_move
        if agentIndex == 0:
            value = float("-inf")
        else:
            value = float("inf")
        for action in gameState.getLegalActions(agentIndex):
            nxt_pos = gameState.generateSuccessor(agentIndex, action)
            if agentIndex == 0:
                nxt_val, nxt_move = self.alphaBeta(nxt_pos, 1, depth, alpha, beta)
                if value < nxt_val:
                    value, best_move = nxt_val, action
                if value >= beta:
                    return value, best_move
                alpha = max(alpha, value)
            elif agentIndex == gameState.getNumAgents() - 1:
                nxt_val, nxt_move = self.alphaBeta(nxt_pos, 0, depth+1, alpha, beta)
                if value > nxt_val:
                    value, best_move = nxt_val, action
                if value <= alpha:
                    return value, best_move
                beta = min(beta, value)
            else:
                nxt_val, nxt_move = self.alphaBeta(nxt_pos, agentIndex+1, depth, alpha, beta)
                if value > nxt_val:
                    value, best_move = nxt_val, action
                if value <= alpha:
                    return value, best_move
                beta = min(beta, value)
        return value, best_move

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
        best_move = self.expectedMax(gameState, 0, 0)[1]
        return best_move

    def expectedMax(self, gameState, agentIndex, depth):
        best_move = None
        if (depth == self.depth) or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), best_move
        if agentIndex == 0:
            value = float("-inf")
        else:
            value = 0
        legalActions = gameState.getLegalActions(agentIndex)
        for action in legalActions:
            nxt_pos = gameState.generateSuccessor(agentIndex, action)
            if agentIndex == 0:
                nxt_val, nxt_move = self.expectedMax(nxt_pos, 1, depth)
                if value < nxt_val:
                    value, best_move = nxt_val, action
            else:
                probability = 1.0/len(legalActions)
                if agentIndex == gameState.getNumAgents() - 1:
                    nxt_val, nxt_move = self.expectedMax(nxt_pos, 0, depth + 1)
                    value += nxt_val
                else:
                    nxt_val, nxt_move = self.expectedMax(nxt_pos, agentIndex + 1, depth)
                    value += nxt_val
        return value, best_move


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isLose():
        # Prevent losing
        return float("-inf")
    if currentGameState.isWin():
        # Award winning
        return float("inf")
    score = currentGameState.getScore()
    food = currentGameState.getFood().asList()
    GhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
    pos = currentGameState.getPacmanPosition()
    capsules = currentGameState.getCapsules()
    # Increase more score if closest food is far away.
    if len(GhostStates) > 0:
        # Less score for getting close to ghosts
        closest_ghost = 0
        for i in range(len(GhostStates)):
            ghost_distance = manhattanDistance(GhostStates[i].getPosition(), pos)
            if ScaredTimes[i] > ghost_distance:
                closest_ghost = -ghost_distance
            elif ghost_distance <= 1:
                # Prevent bumping into ghost
                return float("-inf")
            if closest_ghost == 0 or closest_ghost > ghost_distance:
                closest_ghost = ghost_distance
        score -= 1 / closest_ghost
        if len(capsules) > 0:
            # Going to capsule increase points
            closest_capsule = (min(manhattanDistance(pos, cap) for cap in capsules))
            if closest_capsule < closest_ghost:
                score += 1/closest_capsule
    closest_food = 0.1
    temp_food = []
    if len(food) > 1:
        for f in food:
            temp_food.append(manhattanDistance(pos, f))
        closest_food = min(temp_food)
    score += 1/closest_food
    # Want as few food as possible
    score -= 10 * len(food)
    return score

# Abbreviation
better = betterEvaluationFunction
