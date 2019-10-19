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
    '''
      How do we evaluate a position's potential
        - Number of food elements left
        - Position of ghosts to pacman
        - Direction of action

    '''
    rawScore = 0
    if currentGameState.isWin():
      return float('inf')
    elif successorGameState.isLose():
      return float('-inf')

    foodList = newFood.asList()
    # iterate through the length of ghoststates and scaredtimes
    ghost_dist = float('inf')
    for i in range(0, len(newGhostStates)):
      ghost_dist = min(ghost_dist, manhattanDistance(newGhostStates[i].getPosition(), newPos))

      if successorGameState.getNumFood() < currentGameState.getNumFood():
        rawScore += 2000
      else:
        food_dist = float('inf')
        for food in foodList:
          food_dist = min(food_dist, manhattanDistance(food, newPos))

        if newScaredTimes[i] > 5:
          rawScore += 8**(2-ghost_dist)
          rawScore += 8**(2-food_dist)
        else:
          rawScore -= 8**(2-ghost_dist)
          rawScore += 8**(2-food_dist)

    return rawScore

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

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    if gameState.isWin() or gameState.isLose():
      return Directions.STOP

    pacmanIndex = 0
    ghostIndex = 1
    currDepth = 1
    maxDepth = self.depth

    currMove = Directions.STOP
    possibleMoves = gameState.getLegalActions(pacmanIndex)

    v = float('-inf')
    for move in possibleMoves:
      if gameState.generateSuccessor(pacmanIndex, move).isWin():
        return move

      score = self.moveMin(gameState.generateSuccessor(pacmanIndex, move), currDepth, maxDepth, ghostIndex)
      if score > v:
        v = score
        currMove = move

    return currMove
        
  def moveMax(self, gameState, currDepth, maxDepth, agentIndex):
    if gameState.isWin() or gameState.isLose():
      return self.evaluationFunction(gameState)

    v = float('-inf')

    possibleMoves = gameState.getLegalActions(agentIndex)
    nextStates = [gameState.generateSuccessor(agentIndex, move) for move in possibleMoves]
    scores = [self.moveMin(state, currDepth, maxDepth, 1) for state in nextStates]

    v = max(scores)
    return v
  
  def moveMin(self, gameState, currDepth, maxDepth, agentIndex):
    if gameState.isWin() or gameState.isLose() or currDepth == maxDepth:
      return self.evaluationFunction(gameState)
        
    v = float('inf')
    possibleMoves = gameState.getLegalActions(agentIndex)
    nextStates = [gameState.generateSuccessor(agentIndex, move) for move in possibleMoves]
    if currDepth == maxDepth:
      if agentIndex == gameState.getNumAgents():
        scores = [self.evaluationFunction(state) for state in nextStates]
      else:
        scores = [self.moveMin(state, currDepth, maxDepth, agentIndex+1) for state in nextStates]
    else:
      if agentIndex == gameState.getNumAgents()-1:
        scores = [self.moveMax(state, currDepth+1, maxDepth, 0) for state in nextStates]
      else:
        scores = [self.moveMin(state, currDepth, maxDepth, agentIndex+1) for state in nextStates]

    v = min(scores)
    return v
      

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    if gameState.isWin() or gameState.isLose():
      return Directions.STOP

    pacmanIndex = 0
    ghostIndex = 1
    currDepth = 1
    maxDepth = self.depth

    v = float('-inf')
    # MAX's best option on path to root
    alpha = float('-inf')
    # MIN's best option on path to root
    beta = float('inf')
    
    currMove = Directions.STOP
    possibleMoves = gameState.getLegalActions(pacmanIndex)

    for move in possibleMoves:
      state = gameState.generateSuccessor(pacmanIndex, move)
      if state.isWin():
        return move

      score = self.moveMin(state, currDepth, maxDepth, ghostIndex, alpha, beta)
      if score > v:
        v = score
        currMove = move

    return currMove

  def moveMax(self, gameState, currDepth, maxDepth, agentIndex, alpha, beta):
    if gameState.isWin() or gameState.isLose():
      return self.evaluationFunction(gameState)

    v = float('-inf')
    possibleMoves = gameState.getLegalActions(agentIndex)
    for move in possibleMoves:
      state = gameState.generateSuccessor(agentIndex, move)
      v = max(v, self.moveMin(state, currDepth, maxDepth, agentIndex+1, alpha, beta))
      if v >= beta:
        return v
      alpha = max(v, alpha)
    
    return v

  def moveMin(self, gameState, currDepth, maxDepth, agentIndex, alpha, beta):
    if gameState.isWin() or gameState.isLose():
      return self.evaluationFunction(gameState)

    v = float('inf')
    possibleMoves = gameState.getLegalActions(agentIndex)
    for move in possibleMoves:
      state = gameState.generateSuccessor(agentIndex, move)
      if currDepth == maxDepth:
        # last min move
        if agentIndex == gameState.getNumAgents()-1:
          v = min(v, self.evaluationFunction(state))
        else:
          v = min(v, self.moveMin(state, currDepth, maxDepth, agentIndex+1, alpha, beta))
      else:
        if agentIndex == gameState.getNumAgents()-1:
          v = min(v, self.moveMax(state, currDepth+1, maxDepth, 0, alpha, beta))
        else:
          v = min(v, self.moveMin(state, currDepth, maxDepth, agentIndex+1, alpha, beta))
      
      if v <= alpha:
        return v
      beta = min(beta, v)

    return v


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
    # util.raiseNotDefined()
    if gameState.isWin() or gameState.isLose():
      return Directions.STOP
    
    pacmanIndex = 0
    ghostIndex = 1
    currDepth = 1
    maxDepth = self.depth
    currMove = Directions.STOP

    v = float('-inf')
    possibleMoves = gameState.getLegalActions(pacmanIndex)
    for move in possibleMoves:
      state = gameState.generateSuccessor(pacmanIndex, move)
      if state.isWin():
        return move
      
      score = self.moveMin(state, currDepth, maxDepth, ghostIndex)
      if score > v:
        v = score
        currMove = move
      
    return currMove

  def moveMax(self, gameState, currDepth, maxDepth, agentIndex):
    if gameState.isWin() or gameState.isLose():
      return self.evaluationFunction(gameState)

    possibleMoves = gameState.getLegalActions(agentIndex)
    nextStates = [gameState.generateSuccessor(agentIndex, move) for move in possibleMoves]

    scores = [self.moveMin(state, currDepth, maxDepth, agentIndex+1) for state in nextStates]
    return max(scores)

  def moveMin(self, gameState, currDepth, maxDepth, agentIndex):
    if gameState.isWin() or gameState.isLose():
      return self.evaluationFunction(gameState)

    v = 0
    possibleMoves = gameState.getLegalActions(agentIndex)
    nextStates = [gameState.generateSuccessor(agentIndex, move) for move in possibleMoves]
    # for state in nextStates:
    if currDepth == maxDepth:
      if agentIndex == gameState.getNumAgents()-1:
        scores = [self.evaluationFunction(state) for state in nextStates]
      else:
        scores = [self.moveMin(state, currDepth, maxDepth, agentIndex+1) for state in nextStates]
    else:
      if agentIndex == gameState.getNumAgents()-1:
        scores = [self.moveMax(state, currDepth+1, maxDepth, 0) for state in nextStates]
      else:
        scores = [self.moveMin(state, currDepth, maxDepth, agentIndex+1) for state in nextStates]

    numStates = len(scores)
    v = sum(scores)/numStates
    return v


def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    Values returned are most likely negative and are scaled depending
     on a variety of criteria.

    Distance to ghosts:
      If ghost is not scared:
        - grows exponentially larger depending on how close each ghost
        is to PacMan
        - these values are then subtracted from the returned value
        
    Distance to food:
      - grows exponentially larger depending on how close each food pellet
       is to PacMan
      - these values are then added to the returned value

    Number of food elements:
      - decreases linearly as the number of remaining food elements
       decreases
      - this value is subtracted from the returned value
    
    Number of food capsules:
      - decreases linearly as the number of remaining food capsules 
       decreases
      - this value is then subtracted from the returned value
    
    Number of scared ghosts:
      - decreases linearly the number of remaining scared ghosts decreases
      - this value is added to the returned value

    
  """
  "*** YOUR CODE HERE ***"
  # util.raiseNotDefined()
  if currentGameState.isWin():
    return float('inf')
  
  if currentGameState.isLose():
    return float('-inf')

  ret = 0
  currentPos = currentGameState.getPacmanPosition()
  ghostStates = currentGameState.getGhostStates()
  ghostPositions = currentGameState.getGhostPositions()
  foodCount = currentGameState.getNumFood()
  foodList = currentGameState.getFood().asList()
  scaredTimes = [ghost.scaredTimer for ghost in ghostStates]
  food_dist = float("inf")
  numPellets = len(currentGameState.getCapsules())

  ghost_scared_dist = float('inf')
  active_ghost_dist = float('inf')

  for i in range(0, len(ghostStates)):
    ghost_dist = manhattanDistance(currentPos, ghostStates[i].getPosition())
    if scaredTimes[i] > 10:
      ret += 25
    else:
      active_ghost_dist = min(active_ghost_dist, ghost_dist)

  for food in foodList:
    food_dist = min(food_dist, manhattanDistance(currentPos, food))
  
  ret += 16**(2-food_dist)
  ret -= 5*numPellets
  ret -= 32**(2-active_ghost_dist)
  ret -= 30*foodCount

  return ret

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
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

