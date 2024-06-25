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


def dfs_depth(problem: SearchProblem, dep,st,res,vis,flag=[]):
    cur=st.pop()
    st.push(cur)
    vis[cur]=1
    if problem.isGoalState(cur):
        flag.append(1)
        return
    for next in problem.getSuccessors(cur):
        if next[0] not in vis:
            st.push(next[0])
            res.append(next[1])
            temp = dfs_depth(problem, dep+1, st,res,vis,flag) 
            if(flag!=[]):
                return
            st.pop()
            res.pop()
    del vis[cur]
    return 

    
def depthFirstSearch(problem: SearchProblem):
    st =util.Stack()
    st.push(problem.getStartState())
    res=[]
    vis={}
    dfs_depth(problem,0,st,res,vis)
    # st
    # util.raiseNotDefined()
    return  res
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    """
    "*** YOUR CODE HERE ***"
    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))
# problem.getSuccessors(problem.getStartState()))

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    st = util.Queue()
    com={}
    com[problem.getStartState()]="None"
    di={}
    di[problem.getStartState()]="None"
    st.push(problem.getStartState())
    cur=-1
    while not st.isEmpty():
        cur = st.pop()
        if problem.isGoalState(cur):
            break
        for next in problem.getSuccessors(cur):
            if next[0] not in com:
                com[next[0]]=cur
                di[next[0]]=next[1]
                st.push(next[0])
    res=[]
    
    while com[cur]!="None":
        res.append(di[cur])
        cur=com[cur]
    res.reverse()
    return res

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    st = util.PriorityQueue()
    st.push(problem.getStartState(),0)
    dist={} 
    dist[problem.getStartState()]=0
    di={}
    di[problem.getStartState()]=[]
    vis={}
    cur=-1
    while not st.isEmpty():
        cur = st.pop()
        if(cur in vis):
            continue

        if problem.isGoalState(cur):
            return di[cur]
        for next in problem.getSuccessors(cur):
            tmp=di[cur].copy()
            tmp.append(next[1])
            cost=problem.getCostOfActions(tmp)
            if next[0] not in dist or cost<dist[next[0]]:
                dist[next[0]]=cost
                di[next[0]]=tmp
                st.update(next[0],cost)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    st = util.PriorityQueue()
    st.push(problem.getStartState(),0+heuristic(problem.getStartState(),problem))
    dist={} 
    dist[problem.getStartState()]=0
    di={}
    di[problem.getStartState()]=[]
    vis={}
    cur=-1
    while not st.isEmpty():
        cur = st.pop()
        if(cur in vis):
            continue

        if problem.isGoalState(cur):
            return di[cur]
        for next in problem.getSuccessors(cur):
            tmp=di[cur].copy()
            tmp.append(next[1])
            cost=problem.getCostOfActions(tmp)+heuristic(next[0],problem)
            if next[0] not in dist or cost<dist[next[0]]:
                dist[next[0]]=cost
                di[next[0]]=tmp
                st.update(next[0],cost)
    



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
