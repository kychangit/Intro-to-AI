# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    open = util.Stack()
    open.push((problem.getStartState(), [], [])) #(state, action, path)
    while not open.isEmpty():
        p = open.pop()
        state = p[0]
        actions = p[1]
        path = p[2]
        if problem.isGoalState(state):
            return actions
        for succ in problem.getSuccessors(state):
            if not succ[0] in path:
                open.push((succ[0], actions + [succ[1]], path + [succ[0]]))
    return []
            

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    step_cost = 1
    open = util.Queue()
    start = problem.getStartState()
    open.push((start, [], 0))  # (state, action, cost)
    seen = {start : 0}
    while not open.isEmpty():
        p = open.pop()
        state = p[0]
        cost = p[2]
        if cost <= seen[state]:
            actions = p[1]
            if problem.isGoalState(state):
                return actions
            for succ in problem.getSuccessors(state):
                succ_state = succ[0]
                new_cost = cost + step_cost
                if (not succ_state in seen) or (new_cost < seen[succ_state]):
                    open.push((succ_state, actions + [succ[1]], new_cost))
                    seen[succ_state] = new_cost
    return []

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    open = util.PriorityQueue()
    start = problem.getStartState()
    open.push((start, [], 0), 0)  # (state, action, cost), initial cost
    seen = {start: 0}
    while not open.isEmpty():
        p = open.pop()
        state = p[0]
        cost = p[2]
        if cost <= seen[state]:
            actions = p[1]
            if problem.isGoalState(state):
                return actions
            for succ in problem.getSuccessors(state):
                succ_state = succ[0]
                new_cost = cost + succ[2]
                if (not succ_state in seen) or (new_cost < seen[succ_state]):
                    open.push((succ_state, actions + [succ[1]], new_cost), new_cost)
                    seen[succ_state] = new_cost
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

from util import PriorityQueue
import heapq

class BreakTiePriorityQueue(PriorityQueue):

    def push(self, item, priority, second_priority):
        entry = (priority, second_priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, _, item) = heapq.heappop(self.heap)
        return item

    def update(self, item, priority, second_priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal priority, compare second priority, if item with higher second
        # priority update its priority and rebuild the heap.
        # If item already in priority queue with lower priority, or lower priority and equal second priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, s, c, i) in enumerate(self.heap):
            if i == item:
                if p < priority:
                    break
                elif p == priority:
                    if s <= second_priority:
                        break
                del self.heap[index]
                self.heap.append((priority, second_priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority, second_priority)


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    '''open = util.PriorityQueue()
    start = problem.getStartState()
    biasfactor = 1.0001 #so that longer paths will be prioritized when breaking ties
    open.push((start, [], 0), 0 + heuristic(start, problem)*biasfactor)
    seen = {start: 0}
    while not open.isEmpty():
        p = open.pop()
        state = p[0]
        cost = p[2]
        if cost <= seen[state]:
            actions = p[1]
            if problem.isGoalState(state):
                return actions
            for succ in problem.getSuccessors(state):
                succ_state = succ[0]
                new_cost = cost + succ[2]
                if (not succ_state in seen) or (new_cost < seen[succ_state]):
                    open.push((succ_state, actions + [succ[1]], new_cost), new_cost + heuristic(succ_state, problem)*biasfactor)
                    seen[succ_state] = new_cost
    return []
    The following implementation use priority queue that break ties depending on smaller heuristic (larger real cost)'''
    open = BreakTiePriorityQueue()
    start = problem.getStartState()
    open.push((start, [], 0), 0 + heuristic(start, problem), heuristic(start, problem))
    seen = {start: 0}
    while not open.isEmpty():
        p = open.pop()
        state = p[0]
        cost = p[2]
        if cost <= seen[state]:
            actions = p[1]
            if problem.isGoalState(state):
                return actions
            for succ in problem.getSuccessors(state):
                succ_state = succ[0]
                new_cost = cost + succ[2]
                if (not succ_state in seen) or (new_cost < seen[succ_state]):
                    open.push((succ_state, actions + [succ[1]], new_cost),
                              new_cost + heuristic(succ_state, problem), heuristic(succ_state, problem))
                    seen[succ_state] = new_cost
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch