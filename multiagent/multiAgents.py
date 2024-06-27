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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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

        newGhostPositions = [ghostState.getPosition() for ghostState in newGhostStates]
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        tmp=successorGameState.getLegalPacmanActions()
        midis=1000000
        for pos in newGhostPositions:
            if newPos == pos:
                return -1000000000-(5-len(tmp))*1000
            if(pos[0] == newPos[0] and abs(pos[1] - newPos[1]) == 1):
                return -100000000-(5-len(tmp))*1000
            if(pos[1] == newPos[1] and abs(pos[0] - newPos[0]) == 1):
                return -100000000-(5-len(tmp))*1000
            midis=min(midis,abs(pos[0]-newPos[0])+abs(pos[1]-newPos[1]))
        "*** YOUR CODE HERE ***"
        
        
        
        if(newFood.count()==0):
            return 100000000
        mi=100000000
        # print(newFood.asList(),newPos)
        for food in newFood.asList():
            mi=min(mi,abs(food[0]-newPos[0])+abs(food[1]-newPos[1]))
        # print(mi)
        if(currentGameState.getFood().count()>newFood.count()):
            mi=0
        mi*=60
        Penalty=20
        if(midis<=4):
            Penalty+=(4-midis)*20
        if(len(tmp)<=3):
            
            mi+=(4-len(tmp))*Penalty
        if(action=="Stop"):
            mi+=100000
        return 10000000-mi-newFood.count()*300
def scoreEvaluationFunction(currentGameState: GameState):
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
    def dfs(self,gameState: GameState,depth,agentIndex):
        if(depth==0 or gameState.isWin() or gameState.isLose()):
            return (self.evaluationFunction(gameState),Directions.STOP)
        num=gameState.getNumAgents()
        if(agentIndex==0):
            mx=(-1000000000,Directions.STOP)
            for action in gameState.getLegalActions(agentIndex):
                mx=max(mx,self.dfs(gameState.generateSuccessor(agentIndex,action),depth-int((agentIndex+1)==num),(agentIndex+1)%num)[:1]+(action,))
            return mx
        else:
            mn=(1000000000,Directions.STOP)
            for action in gameState.getLegalActions(agentIndex):
                mn=min(mn,self.dfs(gameState.generateSuccessor(agentIndex,action),depth-int((agentIndex+1)==num),(agentIndex+1)%num)[:1]+(action,))
            return mn
    def getAction(self, gameState: GameState):
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
        
        return self.dfs(gameState,self.depth,0)[1]
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def dfs(self,gameState: GameState,alpha, beta,depth,agentIndex,):
        if(depth==0 or gameState.isWin() or gameState.isLose()):
            return (self.evaluationFunction(gameState),Directions.STOP)
        num=gameState.getNumAgents()
        if(agentIndex==0):
            mx=(-1000000000,Directions.STOP)
            
            for action in gameState.getLegalActions(agentIndex):
                mx=max(mx,self.dfs(gameState.generateSuccessor(agentIndex,action),alpha, beta,depth-int((agentIndex+1)==num),(agentIndex+1)%num)[:1]+(action,))
                alpha=max(alpha,mx[0])
                if beta<alpha:
                    break
            return mx
        else:
            mn=(1000000000,Directions.STOP)
            for action in gameState.getLegalActions(agentIndex):
                mn=min(mn,self.dfs(gameState.generateSuccessor(agentIndex,action),alpha, beta,depth-int((agentIndex+1)==num),(agentIndex+1)%num)[:1]+(action,))
                beta=min(beta,mn[0])
                if beta<alpha:
                    break
            return mn
    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.dfs(gameState,-1000000000,1000000000,self.depth,0)[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def dfs(self,gameState: GameState,depth,agentIndex):
        if(depth==0 or gameState.isWin() or gameState.isLose()):
            return (self.evaluationFunction(gameState),Directions.STOP)
        num=gameState.getNumAgents()
        if(agentIndex==0):
            mx=(-1000000000,Directions.STOP)
            for action in gameState.getLegalActions(agentIndex):
                tmp=self.dfs(gameState.generateSuccessor(agentIndex,action),depth-int((agentIndex+1)==num),(agentIndex+1)%num)
                score=tmp[0]
                if(action==Directions.STOP and depth==self.depth):
                    score-=20
                mx=max(mx,(score,action))
            return mx
        else:
            case_num=len(gameState.getLegalActions(agentIndex))
            sm=0
            if(case_num==0):
                return (100000000,Directions.STOP)  
            for action in gameState.getLegalActions(agentIndex):
                sm+=self.dfs(gameState.generateSuccessor(agentIndex,action),depth-int((agentIndex+1)==num),(agentIndex+1)%num)[0]
            sm/=case_num
            mn=(sm,Directions.STOP)
            return mn 
    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        
        return self.dfs(gameState,self.depth,0)[1]
def getnxt(mp,cur):
    dx=[0,0,1,-1]
    dy=[1,-1,0,0]
    res=[]
    for i in range(4):
        x=cur[0]+dx[i]
        y=cur[1]+dy[i]
        # print(mp[x][y])
        if mp[x][y]==False:
            res.append((x,y))
    # print(len(res))
    return res
def bfs(currentGameState: GameState):
    newFood = currentGameState.getFood()
    mp=currentGameState.getWalls()
    st = util.Queue()
    
    dis={}
    dis[currentGameState.getPacmanPosition()]=0
    st.push(currentGameState.getPacmanPosition())
    cur=-1
    cnt=0
    while not st.isEmpty():
        cur = st.pop()
        cnt+=1
        if cur in newFood.asList():
            return dis[cur]
        for next in getnxt(mp,cur):
            if next not in dis:
                dis[next]=dis[cur]+1
                st.push(next)
    # print(mp)
    util.raiseNotDefined()
def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    score=currentGameState.getScore()
    if(currentGameState.isWin()or currentGameState.isLose()):
        if(score>=1000):
            score+=200
        elif(score>=500):
            score+=100
        return score*100
    
    newpos=currentGameState.getPacmanPosition()    
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newGhostPositions = [ghostState.getPosition() for ghostState in newGhostStates]
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    choice=currentGameState.getLegalPacmanActions()
    midis=1000000
    for states in newGhostStates:
        pos=states.getPosition()
        if(states.scaredTimer>2):
            continue
        midis=min(midis,abs(pos[0]-newpos[0])+abs(pos[1]-newpos[1]))
    "*** YOUR CODE HERE ***"
    
    k=1.0
    if(midis<=3):
        k*=0.8
    if(midis<=2):
        k*=midis/5
    if(midis<=1 and len(choice)<=2):
        score-=50*(2-midis)*(2-len(choice))
        k*=0.5
    newfood_num=newFood.count()
    mp=currentGameState.getWalls()
    print(mp)
    # mi=100000000
    # # print(newFood.asList(),newPos)
    # for food in newFood.asList():
    #     mi=min(mi,abs(food[0]-newpos[0])+abs(food[1]-newpos[1]))
    # # print(mi)
    mi=bfs(currentGameState)
    score*=100
      
    score+=20-mi*4
    if(k*newfood_num<0):
        util.unimplemented()
    return score+k*newfood_num*40

    return 0

# Abbreviation
better = betterEvaluationFunction
