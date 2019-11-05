# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

# my imports are the followings imports:
import sys
import random


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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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

        # closest food distance
        newFoodList = newFood.asList()
        closestFoodDistance = -1
        for food in newFoodList:
            distance = manhattanDistance(newPos, food)
            if closestFoodDistance == -1 or distance <= closestFoodDistance:
                closestFoodDistance = distance

        # closest ghost position and average ghost position
        closestGhostDistance = -1
        averageGhostPosition = (0, 0)
        numOfGhosts = 0
        for ghost in successorGameState.getGhostPositions():
            # count number of ghosts
            numOfGhosts = numOfGhosts + 1
            # find minimum distance from pacMan to ghost
            distance = manhattanDistance(ghost, newPos)
            if closestGhostDistance == -1 or closestGhostDistance < distance:
                closestGhostDistance = distance

            # find average ghost position
            averageGhostPosition = averageGhostPosition + ghost

        # divide sum of all ghost positions by numOfGhost to get the AverageGhostPosition
        averageGhostPosition = ((averageGhostPosition[0] / numOfGhosts), (averageGhostPosition[1] / numOfGhosts))
        distanceToAverageGhost = manhattanDistance(averageGhostPosition, newPos)

        # if scared times is true then make closest ghost increase the points
        if newScaredTimes != [0]:
            closestGhostDistance = closestFoodDistance * -1
            distanceToAverageGhost = distanceToAverageGhost * -1

        # make sure not to divide by zero
        if closestGhostDistance == 0:
            closestGhostDistance = 0.01
        if distanceToAverageGhost == 0:
            distanceToAverageGhost = 1
        if closestFoodDistance == 0:
            closestFoodDistance = 0.5

        # higher number the better the position
        return successorGameState.getScore() - (1 / closestGhostDistance) - (
                (0.5 / numOfGhosts) / distanceToAverageGhost) + (0.5 / closestFoodDistance)


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """

        # a minmax search algorithm without pruning,
        # this algorithm can work with multiple min players like needed in the pacman game for multiply ghosts
        #
        def minMaxSearch(depth, gameState, agentIndex):
            # braking condition for this recursive function
            # if depth ir reached or this is the wining or losing state, return the evaluation of the gameState
            # that was reached
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            # list of all the minMaxSearch results
            searchResults = []

            # determine if max's plays or min's move
            pacManMove = False
            if agentIndex == 0:
                pacManMove = True

            # max players move, tries to maximize
            if pacManMove:
                for action in gameState.getLegalActions(agentIndex):
                    searchResults.append(minMaxSearch(depth, gameState.generateSuccessor(agentIndex, action), 1))
                return max(searchResults)

            else:

                # ghosts move, min player
                newAgent = agentIndex + 1
                if agentIndex + 1 == gameState.getNumAgents():
                    # next turn is back to pacMan, increment depth
                    newAgent = 0
                    depth = depth + 1

                # play all the next moves for the ghost
                for action in gameState.getLegalActions(agentIndex):
                    searchResults.append(minMaxSearch(depth, gameState.generateSuccessor(agentIndex, action), newAgent))
                return min(searchResults)

        maxResult = None
        finalAction = None
        # run on all of PacMan's legal actions
        for action in gameState.getLegalActions(0):
            res = minMaxSearch(0, gameState.generateSuccessor(0, action), 1)
            if maxResult is None or res > maxResult:
                maxResult = res
                finalAction = action

        return finalAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
  """

    def getAction(self, gameState):
        """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
        """ 
        this is the alpha beta search algorithm that allows pruning
        we get as parameters: 
            depth - max depth the search runs 
            gameState - current game state we are searching
            agentIndex - the index of the agent to determine if pacMan or ghost is playing
            a, b - the best value from this gameState position to the root, allows pruning
        and returns the value v which is the best value we can reach if both play optimally 
        """

        def alphaBetaSearch(depth, gameState, agentIndex, a, b):
            # braking condition for this recursive function
            # if depth ir reached or this is the wining or losing state, return the evaluation of the gameState
            # that was reached
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            # determine if max's plays or min's move
            pacManMove = False
            if agentIndex == 0:
                pacManMove = True

            if pacManMove:
                # max's turn, pacMan
                v = -sys.maxint - 1
                # run on all action for this state
                for action in gameState.getLegalActions(agentIndex):
                    # following pseudo code from textbook
                    v = max(v, alphaBetaSearch(depth, gameState.generateSuccessor(agentIndex, action), 1, a, b))
                    # alphaBeta optimization
                    if v > b:
                        return v
                    # update a
                    a = max(a, v)
                # if there was no pruning return max v from all states
                return v

            else:
                # min's turn, ghost
                v = sys.maxint

                # ghosts move, min player
                newAgent = agentIndex + 1
                if agentIndex + 1 == gameState.getNumAgents():
                    # next turn is back to pacMan, increment depth
                    newAgent = 0
                    depth = depth + 1

                # play all the next moves for the ghost
                for action in gameState.getLegalActions(agentIndex):
                    # following psudo code from textbook
                    v = min(v, alphaBetaSearch(depth, gameState.generateSuccessor(agentIndex, action), newAgent, a, b))
                    # alpaBeta optimization
                    if v < a:
                        return v
                    # update a
                    b = min(b, v)
                    # if there was no pruning return max v from all states
                return v

        ## run the pacmans initialization of the search
        maxScore = None
        finalAction = None
        a = None
        for action in gameState.getLegalActions(0):
            res = alphaBetaSearch(0, gameState.generateSuccessor(0, action), 1, -sys.maxint - 1, sys.maxint)
            a = max(res, a)
            if maxScore is None or res > maxScore:
                maxScore = res
                finalAction = action

        return finalAction
        # return Directions.STOP


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
        """ this function preforms expectation max search, this means that we are not considering anymore that
            the ghosts play optimaly, becuase of cure they dont, they play randomly
            we get as parameters: 
                depth - max depth the search runs 
                gameState - current game state we are searching
                agentIndex - the index of the agent to determine if pacMan or ghost is playing
            and returns the value v which is the expected value we can reach when ghosts dont play optimally 
         """

        def expectiMaxSearch(currentDepth, gameState, agentIndex):
            # braking condition for this recursive function
            # if depth ir reached or this is the wining or losing state, return the evaluation of the gameState
            # that was reached
            if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            # list of all the minMaxSearch results
            searchResults = []

            # determine if max's plays or min's move
            pacManMove = False
            if agentIndex == 0:
                pacManMove = True

            # max players move, tries to maximize
            if pacManMove:

                for action in gameState.getLegalActions(agentIndex):
                    searchResults.append(
                        expectiMaxSearch(currentDepth, gameState.generateSuccessor(agentIndex, action), 1))
                # return max result from all the searchResults
                return max(searchResults)

            else:

                # ghosts move, min player
                newAgent = agentIndex + 1
                if agentIndex + 1 == gameState.getNumAgents():
                    # next turn is back to PacMan, increment depth
                    newAgent = 0
                    currentDepth = currentDepth + 1

                # play all the next moves for the ghost
                for action in gameState.getLegalActions(agentIndex):
                    temp = expectiMaxSearch(currentDepth, gameState.generateSuccessor(agentIndex, action), newAgent)
                    searchResults.append(temp)
                avg = sum(searchResults) / float(len(searchResults))
                return avg

        ## run the pacmans initialization of this expectimax search search, with current depth 0
        maxResult = None
        finalAction = None
        # run on all of Pacman's legal actions
        for action in gameState.getLegalActions(0):
            res = expectiMaxSearch(0, gameState.generateSuccessor(0, action), 1)
            if maxResult is None or res > maxResult:
                maxResult = res
                finalAction = action

        return finalAction


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
    DESCRIPTION: <write something here so we know what you did>
    this is the improved evaluation function
    basicly all we did was copy the original evaluation function we wrote at the top of this file
    and we change successorGameState to currentGameState, our initial method was good enough to pass the tests here
    we took into consideration the following ideas
        1. calculate the Mannhattan distance to the closes food and make the half of its inverse added to the points
        because the higher score is the better places where food is close will get a better score
        2. closest ghost position - we calculated the closest Mannhattan distance to that ghost, and by our opinion
        this is the most important factor, so it matters the most, of curse because this is a bad quality being close
        to a ghost we subtracted that from the score
        3. same thing as 3, but for the average position of all ghosts
        4. if newScaredTimes is on, we ate a power pallet we flipped the sighs of the ghost positions from negative
        to positive
    """
    # Useful information you can extract from a GameState (pacman.py)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # closest food distance
    newFoodList = newFood.asList()
    closestFoodDistance = -1
    for food in newFoodList:
        distance = manhattanDistance(newPos, food)
        if closestFoodDistance == -1 or distance <= closestFoodDistance:
            closestFoodDistance = distance

    # closest ghost position and average ghost position
    closestGhostDistance = -1
    averageGhostPosition = (0, 0)
    numOfGhosts = 0
    for ghost in currentGameState.getGhostPositions():
        # count number of ghosts
        numOfGhosts = numOfGhosts + 1
        # find minimum distance from pacMan to ghost
        distance = manhattanDistance(ghost, newPos)
        if closestGhostDistance == -1 or closestGhostDistance < distance:
            closestGhostDistance = distance

        # find average ghost position
        averageGhostPosition = averageGhostPosition + ghost

    # divide sum of all ghost positions by numOfGhost to get the AverageGhostPosition
    averageGhostPosition = ((averageGhostPosition[0] / numOfGhosts), (averageGhostPosition[1] / numOfGhosts))
    distanceToAverageGhost = manhattanDistance(averageGhostPosition, newPos)

    # if scared times is true then make closest ghost increase the points
    if newScaredTimes != [0]:
        closestGhostDistance = closestFoodDistance * -1
        distanceToAverageGhost = distanceToAverageGhost * -1

    # make sure not to divide by zero
    if closestGhostDistance == 0:
        closestGhostDistance = 0.01
    if distanceToAverageGhost == 0:
        distanceToAverageGhost = 1
    if closestFoodDistance == 0:
        closestFoodDistance = 0.5

    # higher number the better the position
    return currentGameState.getScore() - (1 / closestGhostDistance) - (
            (0.5 / numOfGhosts) / distanceToAverageGhost) + (0.5 / closestFoodDistance)


# Abbreviation
better = betterEvaluationFunction


class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest
  """

    def getAction(self, gameState):
        """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)

      I copied the alphaBetaSearch function
      and used the better Evaluation function
      Good Luck at the contest!!!!
    """
        def alphaBetaSearch(depth, gameState, agentIndex, a, b):
            # braking condition for this recursive function
            # if depth ir reached or this is the wining or losing state, return the evaluation of the gameState
            # that was reached
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return betterEvaluationFunction(gameState)

            # determine if max's plays or min's move
            pacManMove = False
            if agentIndex == 0:
                pacManMove = True

            if pacManMove:
                # max's turn, pacMan
                v = -sys.maxint - 1
                # run on all action for this state
                for action in gameState.getLegalActions(agentIndex):
                    # following pseudo code from textbook
                    v = max(v, alphaBetaSearch(depth, gameState.generateSuccessor(agentIndex, action), 1, a, b))
                    # alphaBeta optimization
                    if v > b:
                        return v
                    # update a
                    a = max(a, v)
                # if there was no pruning return max v from all states
                return v

            else:
                # min's turn, ghost
                v = sys.maxint

                # ghosts move, min player
                newAgent = agentIndex + 1
                if agentIndex + 1 == gameState.getNumAgents():
                    # next turn is back to pacMan, increment depth
                    newAgent = 0
                    depth = depth + 1

                # play all the next moves for the ghost
                for action in gameState.getLegalActions(agentIndex):
                    # following psudo code from textbook
                    v = min(v, alphaBetaSearch(depth, gameState.generateSuccessor(agentIndex, action), newAgent, a, b))
                    # alphaBeta optimization
                    if v < a:
                        return v
                    # update a
                    b = min(b, v)
                    # if there was no pruning return max v from all states
                return v
        ## run the pacmans initialization of the search
        maxScore = None
        finalAction = None
        a = None
        for action in gameState.getLegalActions(0):
            res = alphaBetaSearch(0, gameState.generateSuccessor(0, action), 1, -sys.maxint - 1, sys.maxint)
            a = max(res, a)
            if maxScore is None or res > maxScore:
                maxScore = res
                finalAction = action

        return finalAction
