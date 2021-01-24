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

        "* YOUR CODE HERE *"

        score = successorGameState.getScore()
        if(action == 'Stop'):
            score -= 50


        
        newFoodPositions = newFood.asList()
        food_list = currentGameState.getFood()
        food_list1 = food_list.asList()

        foodDistances = [manhattanDistance(newPos,foodPosition) for foodPosition in food_list1]
        capsuleDistances = [manhattanDistance(newPos,cP) for cP in currentGameState.getCapsules()]


        if len(foodDistances) == 0:
            return 1e9

        score += 1.5/(1 + min(foodDistances))

        ghost_states_present = currentGameState.getGhostStates()
        l = len(ghost_states_present)

        closestGhost = 1e18
        i = 0
        while(i<l):

        	ghostState = ghost_states_present[i]

        	c1 = ghostState.configuration

        	c = c1.pos

        	d = manhattanDistance(newPos,c)
        	if(closestGhost > d):
        		closestGhost = d
        		nearestGhost = ghostState


        	i = i + 1
        	


        if(nearestGhost.scaredTimer > 0):
            if(closestGhost == 0):
                score += 200
        else:
            score -= 1.0/(1 + closestGhost)
            # if len(capsuleDistances) > 0:
                # score += 1.5/(1 + min(capsuleDistances))

        

        return(score)

        

    




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

    def smallest_num(self):
    	t = float("-inf")
    	return(t)

    def minimax_max(self,gameState,index,depth):

    	a = ""

    	sd = depth 
    	legal_acts = gameState.getLegalActions(index)

    	m = self.smallest_num()
    	

    	l = len(legal_acts)
    	num_of_agents = gameState.getNumAgents()

    	i = 0

    	while(i<l):

    		temp_act = legal_acts[i]
    		#succ_i = index+1

    		#sd = depth

    		succ = gameState.generateSuccessor(index,temp_act)
    		

    		if(num_of_agents == (index+1)):
    			current_val,act = self.eval(succ,depth+1,0)

    		else:
    			current_val,act = self.eval(succ,depth,index+1)

    		
    		## current_val,act = self.eval(succ,depth,index+1)

    		if(m < current_val):
    			
    			a = temp_act
    			m = current_val

    		i+=1

    	return m,a
        
        

    def largest_num(self):
    	t = float("inf")
    	return(t)


    def minimax_min(self,gameState,index,depth):

    	m = self.largest_num()
    	a = ""


    	sd = depth 
    	legal_acts = gameState.getLegalActions(index)

    	

    	l = len(legal_acts)
    	num_of_agents = gameState.getNumAgents()
    	i = 0

    	while(i<l):
    		temp_act = legal_acts[i]
    		succ_i = index+1

    		sd = depth

    		succ = gameState.generateSuccessor(index,temp_act)
    		

    		if(num_of_agents == (index+1)):
    			
    			current_val,act = self.eval(succ,depth+1,0)

    			## current_val,act = self.eval(succ,depth,index+1)
    		else:
    			current_val,act = self.eval(succ,depth,index+1)


    		

    		if(m > current_val):
    			m = current_val
    			a = temp_act

    		i+=1

    	return m,a


    def eval(self,gameState,depth,index):

    	legal_acts = gameState.getLegalActions(index)

    	l = -1
    	f_score = 0

    	l = len(legal_acts)

    	sd = self.depth

    	if(depth==sd):

    		f_score = gameState.getScore()
    		return f_score,""

    	elif(l==0):

    		f_score = gameState.getScore()
    		return f_score,""

    	elif(index==0):

    		m_val = self.minimax_max(gameState,index,depth)
    		return(m_val)

    	else:
    		mi_val = self.minimax_min(gameState,index,depth)
    		return(mi_val)



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

        initial_i = 0
        start_depth = 0
        score,act = self.eval(gameState,start_depth,initial_i)
        ##print(score)
        return(act)

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def largest_num(self):
    	t = float("inf")
    	return(t)

    def smallest_num(self):
    	t = float("-inf")
    	retunr(t)

    


    def alphabeta_min(self,gameState,alpha,beta,index,depth):

    	legal_acts = gameState.getLegalActions(index)

    	a = ""
    	m = self.largest_num()
    	sd = depth

    	l = len(legal_acts)
    	i = 0

    	num_of_agents = gameState.getNumAgents()

    	while(i<l):


    		act = legal_acts[i]
    		succ_i = index+1

    		
    		succ = gameState.generateSuccessor(succ_i-1,act)
    		temp_sc = ()

    		if((index+1)==num_of_agents):
    		
    			temp_sc = self.eval1(succ,alpha,beta,0,depth+1)

    		else:
    			temp_sc = self.eval1(succ,alpha,beta,index+1,depth)

    		

    		if(m > temp_sc[1]):
    			m = temp_sc[1]
    			a = act 

    		if(m<beta):
    			beta = m 

    		if(alpha > m):
    			return a,m


    	#return a,m

    		i+=1

    	return a,m


    def alphabeta_max(self,gameState,alpha,beta,index,depth):

    	legal_acts = gameState.getLegalActions(index)

    	a = ""
    	m = self.smallest_num()
    	sd = depth

    	l = len(legal_acts)
    	i = 0

    	num_of_agents = gameState.getNumAgents()

    	while(i<l):


    		act = legal_acts[i]

    		succ_i = index+1
    		temp_sc = ()
    		succ = gameState.generateSuccessor(succ_i-1,act)

    		if((index+1)==num_of_agents):
    			
    			temp_sc = self.eval1(succ,alpha,beta,0,depth+1)

    		else:
    			temp_sc = self.eval1(succ,alpha,beta,index+1,depth)

    		

    		if(m < temp_sc[1]):
    			m = temp_sc[1]
    			a = act 

    		if(alpha<m):
    			alpha = m 

    		if(beta < m):
    			return a,m


    	#return a,m

    		i+=1

    	return a,m
        
        

    def eval1(self, gameState, alpha, beta, index, depth):

    	sd = self.depth

    	f_score = 0

    	legalMoves = gameState.getLegalActions(index)
    	l = len(legalMoves)

    	if(sd == depth):
    		f_score = gameState.getScore()
    		return "",f_score

    	elif(l==0):
    		f_score = gameState.getScore()
    		return "",f_score

    	elif(index==0):
    		a,m = self.alphabeta_max(gameState,alpha,beta,index,depth)
    		return a,m 
    	else:
    		a,m = self.alphabeta_min(gameState,alpha,beta,index,depth)
    		return a,m 



    def formulate(self,gs,alpha,beta):
    	set_depth = 0
    	set_index = 0
    	res = self.eval1(gs,alpha,beta,set_index,set_depth)
    	return(res)


    def smallest_num(self):
    	t = float("-inf")
    	return(t)

    def largest_num(self):
    	t = float("inf")
    	return(t)


    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        alpha = self.smallest_num()
        beta = self.largest_num()

        act,score = self.formulate(gameState,alpha,beta)


        return(act)

        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """


    def largest_num(self):
    	t = float("inf")
    	return(t)

    def smallest_num(self):
    	t = float("-inf")
    	return(t)

    def expectimax_max(self, gameState, index, depth):

    	a = ""
    	m = self.smallest_num()

    	legal_acts = gameState.getLegalActions(index)
    	l = len(legal_acts)
    	sd = depth

    	num_of_agents = gameState.getNumAgents()

    	i = 0
    	while(i<l):

    		succ_i = index + 1
    		sd = depth

    		act = legal_acts[i]

    		succ = gameState.generateSuccessor(index,act)
    		temp_act = ""
    		temp_sc = 0

    		if(num_of_agents == (index+1)):
    			succ_i = 0
    			sd += 1
    			temp_act,temp_sc = self.eval2(succ,succ_i,sd)

    		else:
    			temp_act,temp_sc = self.eval2(succ,succ_i,sd)

    		

    		if(m<temp_sc):
    			m = temp_sc
    			a = act


    		
    		i = i + 1

    	return a,m

    def expectimax_expect(self, gameState, index, depth):

    	num_of_agents = gameState.getNumAgents()

    	legal_acts = gameState.getLegalActions(index)
    	l = len(legal_acts)

    	prob = float(1.0/l)

    	a = ""
    	e = 0

    	i = 0
    	while(i<l):
    		act = legal_acts[i]

    		succ_i = index+1
    		sd = depth

    		succ = gameState.generateSuccessor(index,act)
    		temp_act = ""
    		temp_sc = 0

    		if((index+1)==num_of_agents):
    			succ_i = 0
    			sd = sd + 1
    			temp_act, temp_sc = self.eval2(succ,succ_i,sd)

    		else:
    			temp_act, temp_sc = self.eval2(succ,succ_i,sd)

    		

    		e = e + temp_sc



    		i+=1

    	return a,e

    	
        
        

    def eval2(self,gameState,index,depth):
        
    	sd = self.depth
    	a = ""

    	legal_acts = gameState.getLegalActions(index)
    	l =len(legal_acts)

    	if(sd == depth):
    		f_score = self.evaluationFunction(gameState)
    		
    		return a,f_score


    	elif(l==0):
    		f_score = self.evaluationFunction(gameState)

    		return a,f_score

    	elif(index==0):
    		m = self.expectimax_max(gameState, index, depth)
    		return(m)

    	else:
    		e = self.expectimax_expect(gameState, index, depth)
    		return(e)


        




    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        st = 0
        sd = 0

        start_index = 0
        start_depth = 0
        action,score = self.eval2(gameState,start_index,start_depth)

        return(action)
        util.raiseNotDefined()



def betterEvaluationFunction(currentGameState):
        
    successorGameState = currentGameState
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "* YOUR CODE HERE *"

    

    score = currentGameState.getScore()

    
    food_list = newFood.asList()
    l = len(food_list)


    nearest_food_loc = float('inf')

    '''food_distances = [util.manhattanDistance(newPos, food_position) for food_position in food_list]

    nearest_food_loc = min(food_distances)'''

    i = 0
    while(i<l):

        d = util.manhattanDistance(newPos,food_list[i])
        if(d < nearest_food_loc):
            nearest_food_loc = d
        i+=1


    ghost_positions = currentGameState.getGhostPositions()
    l1 = len(ghost_positions)
    
    temp = 0
    n = 0

    j = 0

    tlist = []

    while(j<l1):

    	g = ghost_positions[j]
    	g1 = newGhostStates[j]

    	d = util.manhattanDistance(g,newPos)

    	temp = temp + d 
    	if(g1.scaredTimer > 0):
    		temp -= d


    	if(d<=1):
    		n += 1
    		'''if(g1.scaredTimer > 0):
    			n -= 1'''

    	j+=1
    	tlist.append(g1)


    sum_ghost_dis = temp + 1



    
    newCapsule = currentGameState.getCapsules()
    nearest_cp_distance = float('inf')

    numberOfCapsules = len(newCapsule)

    i = 0
    while(i<numberOfCapsules):

        d = util.manhattanDistance(newPos,newCapsule[i])
        if(d < nearest_cp_distance):
            nearest_cp_distance = d
        i+=1

    ## 1,1 951
    ## 2,1 890
    ## 1,2 984
    ## 1,2,0.5,1 
    ## 1,2,0.5,0.5 829.67
    t2 = (1.0/float(nearest_food_loc))-(1.0/float(sum_ghost_dis))-(1)*(n)-(1)*(numberOfCapsules) ## with scared = fail, without scared = 850 ish
    ## final :
    t3 = (5.0/float(nearest_food_loc))-(2.0/float(sum_ghost_dis))+(10.0/(nearest_cp_distance+1)) - (5)*l -(20)*(numberOfCapsules) ## with scared 1292.21 without = 1312.59



    score += t3

    return(score)

    util.raiseNotDefined()



# Abbreviation
better = betterEvaluationFunction
