# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
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
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first
    [2nd Edition: p 75, 3rd Edition: p 87]

    Your search algorithm needs to return a list of actions that reaches
    the goal. Make sure to implement a graph search algorithm
    [2nd Edition: Fig. 3.18, 3rd Edition: Fig 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """

    startState = problem.getStartState()

    # list of all the state on the grid pacman visited
    visitedPositions = [startState]
    path = _dfsHelper(problem, startState, util.Stack(), visitedPositions)
    return path.list


def _dfsHelper(problem, nodeToFringe, currentPath, visitedPositions):
    successors = problem.getSuccessors(nodeToFringe)
    if successors is not None:
        successors = successors[::-1]  # reversing the successors input for requested output
    for neighbor in successors:

        #  if one of the successors is a goal, return a path to it (without fringing it)
        if (problem.isGoalState(neighbor[0])) is True:
            currentPath.push(neighbor[1])
            return currentPath
    for neighbor in successors:

        # if a path is valid, return it; else return None
        if neighbor[0] not in visitedPositions:
            visitedPositions.append(neighbor[0])
            currentPath.push(neighbor[1])
            res = _dfsHelper(problem, neighbor[0], currentPath, visitedPositions)
            if res is not None:
                return res

    # this is not a path to the goal; backtrack
    currentPath.pop()
    return None


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    [2nd Edition: p 73, 3rd Edition: p 82]
    """

    startState = problem.getStartState()

    # list of all the state on the grid pacman visited
    visitedPositions = [startState]

    # starting at goal point
    if problem.isGoalState(startState):
        return []

    # a list of list that holds all the positions and directions of all the paths the bfs explores
    queue = [ [(startState, 'dummy', 0)] ]   # a queue of paths (partial/complete paths)
    # final optimal path to the destination
    goalPath = []
    # a flag to enable braking from the loop when destination is reached
    flag = False

    # number of corners out of four the BFS touched
    while queue:
        # pop the first item from the queue
        nodeToFringe = queue.pop(0)

        # update visitedPositions from the nodeToFringe (it is the last element in the list
        # and first element in the tuple
        # if isCornersProblem:
        #     lastPosition = nodeToFringe[-1][0][0]
        # else:
        #     lastPosition = nodeToFringe[-1][0]

        visitedPositions.append(nodeToFringe[-1][0])

        # iterate on all the possible successor nodes
        successors = problem.getSuccessors(nodeToFringe[-1][0])

        for child in successors:
            childPosition = child[0]
            # if the successor node is in the visitedPositions or already explored by other bfs search continue
            # else enter this if statement
            if childPosition not in visitedPositions:
                # copy newPath so that its address in memory is different then nodeToFringe's address
                newPath = nodeToFringe[:]
                newPath.append(child)

                # if goal path is reached break
                if problem.isGoalState(child[0]):
                    goalPath = newPath
                    flag = True
                    break

                # else add this newPath as the last element in the list
                queue.append(newPath)
                visitedPositions.append(childPosition)
        if flag:
            break

    # return an list of all the directions
    result = [direction[1] for direction in goalPath][1:]
    return result

def uniformCostSearch(problem):
    "Search the node of least total cost first."
    startState = problem.getStartState()
    # list of all the state on the grid pacman visited
    visitedPositions = [startState]
    # check if we start at goal state
    if (problem.isGoalState(startState)) is True:
        return []

    # new cost function that works with on the list we enter the queue
    newCostFunction = lambda path: problem.getCostOfActions(map(lambda (a, b, c): b,  path[1:]))

    # init a priority queue with our new cost function
    queue = util.PriorityQueueWithFunction(newCostFunction)
    queue.push([(startState, 'dummy', 0)])
    # final optimal path to the destination
    goalPath = []

    while not queue.isEmpty():
        # pop the first item from the queue
        nodeToFringe = queue.pop()

        # if node to fringe is the goal state, break
        if problem.isGoalState(nodeToFringe[-1][0]):
            goalPath = nodeToFringe
            break

        # update visitedPositions from the nodeToFringe (it is the last element in the list
        # and first element in the tuple
        visitedPositions.append(nodeToFringe[-1][0])

        # iterate on all the possible successor nodes
        successors = problem.getSuccessors(nodeToFringe[-1][0])

        for child in successors:
            # if the successor node is in the visitedPositions or already explored by other better path
            # else enter this if statement
            if child[0] not in visitedPositions:
                # copy newPath so that its address in memory is different then nodeToFringe's address
                newPath = nodeToFringe[:]
                # add child as and node that has not been expended, only generated
                newPath.append(child)
                queue.push(newPath)

    # return an array of all the directions
    return [direction[1] for direction in goalPath][1:]

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."

    # heuristic setup heuristic(state, problem)


    startState = problem.getStartState()
    # list of all the state on the grid pacman visited
    visitedPositions = [startState]
    # check if we start at goal state
    if (problem.isGoalState(startState)) is True:
        return []

    # new cost function that works with on the list we enter the queue
    newCostFunction = lambda path: problem.getCostOfActions(map(lambda (a, b, c): b, path[1:])) +\
                                   heuristic(path[-1][0], problem)

    # init a priority queue with our new cost function
    queue = util.PriorityQueueWithFunction(newCostFunction)
    queue.push([(startState, 'dummy', 0)])
    # final optimal path to the destination
    goalPath = []

    while not queue.isEmpty():
        # pop the first item from the queue
        nodeToFringe = queue.pop()

        # if node to fringe is the goal state, break
        if problem.isGoalState(nodeToFringe[-1][0]):
            goalPath = nodeToFringe
            break

        # update visitedPositions from the nodeToFringe (it is the last element in the list
        # and first element in the tuple
        visitedPositions.append(nodeToFringe[-1][0])

        # iterate on all the possible successor nodes
        successors = problem.getSuccessors(nodeToFringe[-1][0])

        for child in successors:
            # if the successor node is in the visitedPositions or already explored by other better path
            # else enter this if statement
            if child[0] not in visitedPositions:
                # copy newPath so that its address in memory is different then nodeToFringe's address
                newPath = nodeToFringe[:]
                # add child as and node that has not been expended, only generated
                newPath.append(child)
                queue.push(newPath)

    # return an array of all the directions
    return [direction[1] for direction in goalPath][1:]

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
