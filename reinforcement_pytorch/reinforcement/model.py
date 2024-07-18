import pickle  # 确保在文件顶部导入pickle
    

"""
Functions you should use.
Please avoid importing any other torch functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""

from torch.nn import Module
from torch.nn import  Linear
from torch import tensor, double, optim
from torch.nn.functional import relu, mse_loss



class DeepQNetwork(Module):
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """
    def __init__(self, state_dim, action_dim):
        self.num_actions = action_dim
        self.state_size = state_dim
        super(DeepQNetwork, self).__init__()
        # Remember to set self.learning_rate, self.numTrainingGames,
        # and self.batch_size!
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.15
        self.numTrainingGames = 3300
        self.batch_size = 1024
        self.cnt=0
        import torch
        self.module_=torch.nn.Sequential(
            torch.nn.Linear(state_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            # torch.nn.Linear(128, 64),
            # torch.nn.ReLU(),
            
            torch.nn.Linear(128, action_dim)
        )
        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        "**END CODE"""
        self.double()


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
        Q_pred = self.forward(states)
        loss = mse_loss(Q_pred, Q_target)
        return loss

    def forward(self, states):
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
        return self.module_(tensor(states, dtype=double))
    def run(self, states):
        return self.forward(states)

    # import pickle

    def Save_model(self, suffix, replay_memory):
        import torch
        # 使用torch.save保存模型
        model_file_name = f"model_{suffix}.pth"
        torch.save(self.module_.state_dict(), model_file_name)
        
        # 保存记忆库到文件
        memory_file = f"replay_memory_{suffix}.pth"
        torch.save(replay_memory, memory_file)

        print(f"模型已保存为 {model_file_name}，记忆库已保存为 {memory_file}")
    def Load_model(self,str:str):
        import torch
        self.module_=torch.load('model'+str+'.pth')
    def gradient_update(self, states, Q_target):
        """
        Update your parameters by one gradient step with the .update(...) function.
        You can look at the ML project for an idea of how to do this, but note that rather
        than iterating through a dataset, you should only be applying a single gradient step
        to the given datapoints.

        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            None
        """
        "*** YOUR CODE HERE ***"
        tt=self.learning_rate
        # if(self.cnt>35000):
        #     tt=0.01
        # print(tt,self.cnt)
        
        self.optimizer.zero_grad()
        loss = self.get_loss(states, Q_target)  
        loss.backward()
        self.optimizer.step()
        return None