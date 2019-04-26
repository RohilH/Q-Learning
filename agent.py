import numpy as np
import utils
import random


class Agent:

    def __init__(self, actions, Ne, C, gamma):
        # self.actions = []
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma
        self.actionslist = []
        self.reset()
        # Create the Q and N Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()

    def train(self):
        self._train = True

    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path, self.Q)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # self.actions = []
        self.actionslist = []
        self.points = 0
        self.s = None
        self.a = None

    def act(self, state, points, dead):
        '''
        :param state: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] from environment.
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: the index of action. 0,1,2,3 indicates up,down,left,right separately

        TODO: write your function here.
        Return the index of action the snake needs to take, according to the state and points known from environment.
        Tips: you need to discretize the state to the state space defined on the webpage first.
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the 480x480 board)

        Q(s,a)←Q(s,a)+α(R(s)+γmax(Q(s′,a′))−Q(s,a))
        C/(C+N(s,a))
        '''
        if dead:
            # stateTuple = self.generateState(self.s)
            nStateTuple = self.generateState(state)

            # print(state[0], state[1])
            # print("Num moves: ", len(self.actionslist), "Score: ", points)
            self.setQVal2(self.s, nStateTuple, self.a, dead, points)
            self.reset()
            return

        if len(self.actionslist) != 0:
            nStateTuple = self.generateState(state)

            self.setQVal2(self.s, nStateTuple, self.a, dead, points)

        stateTuple = self.generateState(state)

        fval = [0,0,0,0]
        for i in range(4):
            fval[i] = self.getFVal(self.getQVal(stateTuple, i), self.getNVal(stateTuple, i))
        # print(fval)
        fval = fval[::-1]
        actionToTake = np.argmax(fval)
        actionToTake = 3 - actionToTake
        # print("index: ", actionToTake)
        self.actionslist.append(actionToTake)

        self.N[stateTuple[0]][stateTuple[1]][stateTuple[2]][stateTuple[3]][stateTuple[4]][stateTuple[5]][stateTuple[6]][stateTuple[7]][actionToTake] += 1
        self.s = stateTuple
        self.a = actionToTake
        self.points = points
        return self.actionslist[-1]

        # stateTuple = self.generateState(state)
        # fval = [0,0,0,0]
        # for i in range(4):
        #     fval[i] = self.getFVal(self.getQVal(stateTuple, i), self.getNVal(stateTuple, i))
        # fval = fval[::-1]
        # actionToTake = np.argmax(fval)
        # actionToTake = 3 - actionToTake
        # self.actionslist.append(actionToTake)
        # if dead:
        #     # print(state[0], state[1])
        #     # print("Num moves: ", len(self.actionslist), "Score: ", points)
        #     self.setQVal(state, self.actionslist[-1])
        #     self.reset()
        #     return
        #
        # self.N[stateTuple[0]][stateTuple[1]][stateTuple[2]][stateTuple[3]][stateTuple[4]][stateTuple[5]][stateTuple[6]][stateTuple[7]][actionToTake] += 1
        #
        # if len(self.actionslist) != 0:
        #     self.setQVal(state, self.actionslist[-1])
        # return self.actionslist[-1]



    def setQVal2(self, stateTuple, nStateTuple, action, dead, points):

        reward = -0.1
        if dead:
            reward = -1
        if points - self.points > 0:
            reward = 1

        qsOfA = self.getQVal(stateTuple, action)
        alpha = self.C/(self.C + self.getNVal(stateTuple, action))
        qprime = max(self.getQVal(nStateTuple, 0), self.getQVal(nStateTuple, 1), self.getQVal(nStateTuple, 2),  self.getQVal(nStateTuple, 3))
        setQ = qsOfA + alpha * (reward + self.gamma * qprime - qsOfA)

        self.Q[stateTuple[0]][stateTuple[1]][stateTuple[2]][stateTuple[3]][stateTuple[4]][stateTuple[5]][stateTuple[6]][stateTuple[7]][action] = setQ

    def reward(self, state, action):
        isDead = self.isDead(state, action)
        if isDead:
            return -1
        if (state[0] == state[3] and state[1] == state[4]):
            return 1
        return -0.1
    def generateState(self, state):
        stateTuple = [0,0,0,0,0,0,0,0]
        if (state[0] == utils.GRID_SIZE): # snake_head_x == 40
            stateTuple[0] = 1 # adjoining_wall_x on left side
        elif (state[0] == utils.GRID_SIZE*12): # snake_head_x == 480
            stateTuple[0] = 2  # adjoining_wall_x on right side
        if (state[1] == utils.GRID_SIZE): # snake_head_y == 40
            stateTuple[1] = 1  # adjoining_wall_y on top side
        elif (state[1] == utils.GRID_SIZE*12): # snake_head_y == 480
            stateTuple[1] = 2  # adjoining_wall_y on bottom side
        if (state[0] > state[3]): # snake_head_x > food_x
            stateTuple[2] = 1 # food on snake left
        elif (state[0] < state[3]): # snake_head_x < food_x
            stateTuple[2] = 2 # food on snake right
        if (state[1] > state[4]): # snake_head_y > food_y
            stateTuple[3] = 1 # food on snake top
        elif (state[1] < state[4]): # snake_head_y < food_y
            stateTuple[3] = 2 # food on snake bottom
        for bodyX, bodyY in state[2]:
            if (state[0] == (bodyX + 40) and state[1] == bodyY):
                stateTuple[7] = 1 # body on right of current head
            if (state[0] == bodyX - 40 and state[1] == bodyY):
                stateTuple[6] = 1 # body on left of current head
            if (state[0] == bodyX and state[1] == bodyY + 40):
                stateTuple[5] = 1 # body on bottom of current head
            if (state[0] == bodyX and state[1] == bodyY - 40):
                stateTuple[4] = 1 # body on top of current head
        return stateTuple

    def setQVal(self, state, action):
        stateL = state
        stateR = state
        stateU = state
        stateD = state

        stateU[1] -= 40
        stateD[1] += 40
        stateL[0] -= 40
        stateR[0] += 40
        stateTuple = self.generateState(state)
        stateTupleU = self.generateState(stateU)
        stateTupleD = self.generateState(stateD)
        stateTupleL = self.generateState(stateL)
        stateTupleR = self.generateState(stateR)

        qsOfA = self.getQVal(stateTuple, action)
        alpha = self.C/(self.C + self.getNVal(stateTuple, action))
        qprime = max(self.getQVal(stateTupleU, 0), self.getQVal(stateTupleD, 1), self.getQVal(stateTupleL, 2),  self.getQVal(stateTupleR, 3))
        setQ = qsOfA + alpha * (self.reward(state, action) + self.gamma * qprime - qsOfA)
        self.Q[stateTuple[0]][stateTuple[1]][stateTuple[2]][stateTuple[3]][stateTuple[4]][stateTuple[5]][stateTuple[6]][stateTuple[7]][action] = setQ

    def getQVal(self, stateTuple, action):
        return self.Q[stateTuple[0]][stateTuple[1]][stateTuple[2]][stateTuple[3]][stateTuple[4]][stateTuple[5]][stateTuple[6]][stateTuple[7]][action]
    def getNVal(self, stateTuple, action):
        return self.N[stateTuple[0]][stateTuple[1]][stateTuple[2]][stateTuple[3]][stateTuple[4]][stateTuple[5]][stateTuple[6]][stateTuple[7]][action]
    def getFVal(self, qval, nval):
        if nval < self.Ne:
            return 1
        return qval
    def isDead(self, state, action):
        delta_x = delta_y = 0
        if action == 0:
            delta_y = -1 * utils.GRID_SIZE
        elif action == 1:
            delta_y = utils.GRID_SIZE
        elif action == 2:
            delta_x = -1 * utils.GRID_SIZE
        elif action == 3:
            delta_x = utils.GRID_SIZE

        old_body_head = None
        if len(state[2]) == 1:
            old_body_head = state[2][0]
        state[2].append((state[0], state[1]))
        state[0] += delta_x
        state[1] += delta_y

        # if len(state[2]) > 0:
        del(state[2][0])

        # colliding with the snake body or going backwards while its body length
        # greater than 1
        if len(state[2]) >= 1:
            for seg in state[2]:
                if state[0] == seg[0] and state[1] == seg[1]:
                    return True

        # moving towards body direction, not allowing snake to go backwards while
        # its body length is 1
        if len(state[2]) == 1:
            if old_body_head == (state[0], state[1]):
                return True

        # collide with the wall
        if (state[0] < utils.GRID_SIZE or state[1] < utils.GRID_SIZE or
            state[0] + utils.GRID_SIZE > utils.DISPLAY_SIZE-utils.GRID_SIZE or state[1] + utils.GRID_SIZE > utils.DISPLAY_SIZE-utils.GRID_SIZE):
            return True

        return False
