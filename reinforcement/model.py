import nn

class DeepQNetwork():
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """
    def __init__(self, state_dim, action_dim):
        self.num_actions = action_dim
        self.state_size = state_dim

        # Remember to set self.learning_rate, self.numTrainingGames,
        # self.parameters, and self.batch_size!
        "*** YOUR CODE HERE ***"
        self.learning_rate = 1
        self.numTrainingGames = 10000
        self.batch_size = 128
        self.parameters = []
        self.cnt=0
        # w1
        self.parameters.append(nn.Parameter(self.state_size, 500))
        #b1
        self.parameters.append(nn.Parameter(1, 500))
        # w2
        self.parameters.append(nn.Parameter(500, 300))
        #b2
        self.parameters.append(nn.Parameter(1, 300))
        # w3
        self.parameters.append(nn.Parameter(300, self.num_actions))
        #b3
        self.parameters.append(nn.Parameter(1, self.num_actions))
    def set_weights(self, layers):
        self.parameters = []
        for i in range(len(layers)):
            self.parameters.append(layers[i])

    def get_loss(self, states, Q_target):
        """
        Returns the Squared Loss between Q values currently predicted 
        by the network, and Q_target.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            loss node between Q predictions and Q_target
        """
        "*** YOUR CODE HERE ***"
        Q_pred = self.run(states)
        loss = nn.SquareLoss(Q_pred, Q_target)
        return loss

    def run(self, states):
        """
        Runs the DQN for a batch of states.
        The DQN takes the state and returns the Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            result: (batch_size x num_actions) numpy array of Q-value
                scores, for each of the actions
        """
        "*** YOUR CODE HERE ***"
        actions = []
        for i in range(self.num_actions):
            actions.append(i)
        # import copy
        # Q_pred = copy.deepcopy(states)
        Q_pred=states
        assert len(self.parameters) % 2 == 0
        for i in range(0, len(self.parameters) - 2, 2):
            Q_pred = nn.Linear(Q_pred, self.parameters[i])
            Q_pred = nn.AddBias(Q_pred, self.parameters[i + 1])
            Q_pred = nn.ReLU(Q_pred)
        Q_pred = nn.AddBias(nn.Linear(Q_pred, self.parameters[-2]), self.parameters[-1])
        # Q_pred = nn.ReLU(Q_pred)
        return Q_pred

    def gradient_update(self, states, Q_target):
        """
        Update your parameters by one gradient step with the .update(...) function.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            None
        """
        "*** YOUR CODE HERE ***"
        loss = self.get_loss(states, Q_target)
        # print(self.parameters)
        gradients = nn.gradients(loss, self.parameters)
        # print(gradients)
        # assert 0
        for i in range(len(self.parameters)):
            self.parameters[i].update(gradients[i], -self.learning_rate)
        # self.cnt+=1
        # if(self.cnt>7000):
        #     self.learning_rate=0.5
        # elif(self.cnt>4000):
        #     self.learning_rate=2

        return None
