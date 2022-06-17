import numpy as np
import random
import copy
import pickle


class NASbenchEnv:
    def __init__(self,data_set, val_at_epoch, config_to_state, state_to_config, seed=0):
        assert data_set in ['cifar10', 'cifar100', 'ImageNet16-120']
        np.random.seed(seed)
        random.seed(seed)
        self.kspace = [(0,4),(0,4),(0,4),(0,4),(0,4),(0,4)]
        with open("./data/"+data_set+"_data", "rb") as file:
            self.nasbench = pickle.load(file)
        #node id in [0,5] to edit its value. if its 6, we have a terminal state
        self.state_space = self.kspace+[(0,6)] 
        #each node has 5 choices (0,4)
        self.action_space = (0,4) 
        self.state_dim = 1
        for k in self.state_space:
            self.state_dim *= (k[1]+1)
        self.action_dim = self.action_space[1]+1
        self.val_at_epoch = val_at_epoch
        self.config_to_state = config_to_state
        self.state_to_config = state_to_config

    def reset(self,init_state=None):
        if init_state is None:
            random_state = [2,4,2,4,2,4,np.random.randint(0,6)]
        else:
            random_state = copy.deepcopy(list(self.state_to_config(init_state)))
        self.state = random_state
        return self.config_to_state(self.state)

    def step(self, action):

        reward = self.evaluate(state=self.state,action=action)
        #update state 
        edit_node_id = self.state[-1]
        self.state[edit_node_id] = action
        new_edit_node_id = np.random.randint(self.state_space[-1][0],self.state_space[-1][1]+1)
        self.state[-1] = new_edit_node_id
        if new_edit_node_id == 6:
            done = True
        else:
            done = False
        return self.config_to_state(self.state), reward, done

    def evaluate(self, state, action):
        #executing the action results in configuring the node located at state[-1]  
        if type(state) == int:
            state_ = list(self.state_to_config(state))
        else:
            state_ = copy.deepcopy(state)      
        edit_node_id = state_[-1]
        state_[edit_node_id] = action
        arch_config = state_[0:-1]
        validation_acc = self.nasbench[tuple(arch_config)]
        return validation_acc[self.val_at_epoch]

    def random_action(self):
        edit_node_id = self.state[-1]
        bounds = self.state_space[edit_node_id]
        return np.random.randint(bounds[0],bounds[1]+1)

class ReducedNASbenchEnv:
    def __init__(self,data_set, val_at_epoch, config_to_state, state_to_config, seed=0):
        assert data_set in ['cifar10', 'cifar100', 'ImageNet16-120']
        np.random.seed(seed)
        random.seed(seed)
        self.kspace = [(0,4),(0,4),(0,4),(0,4),(0,4)]
        with open("./data/"+data_set+"_data", "rb") as file:
            self.nasbench = pickle.load(file)
        #node id in [0,3] to edit its value. if its 5, we have a terminal state
        self.state_space = self.kspace+[(0,5)] 
        #each node has 5 choices (0,4)
        self.action_space = (0,4) 
        self.state_dim = 1
        for k in self.state_space:
            self.state_dim *= (k[1]+1)
        self.action_dim = self.action_space[1]+1
        self.val_at_epoch = val_at_epoch
        self.config_to_state = config_to_state
        self.state_to_config = state_to_config

    def reset(self,init_state=None):
        if init_state is None:
            random_state = [4,2,4,2,4,np.random.randint(0,5)]
        else:
            random_state = copy.deepcopy(list(self.state_to_config(init_state)))
        self.state = random_state
        return self.config_to_state(self.state)

    def step(self, action):

        reward = self.evaluate(state=self.state,action=action)
        #update state 
        edit_node_id = self.state[-1]
        self.state[edit_node_id] = action
        new_edit_node_id = np.random.randint(self.state_space[-1][0],self.state_space[-1][1]+1)
        self.state[-1] = new_edit_node_id
        if new_edit_node_id == 5:
            done = True
        else:
            done = False
        return self.config_to_state(self.state), reward, done

    def evaluate(self, state, action):
        #executing the action results in configuring the node located at state[-1]  
        if type(state) == int:
            state_ = list(self.state_to_config(state))
        else:
            state_ = copy.deepcopy(state)      
        edit_node_id = state_[-1]
        state_[edit_node_id] = action
        arch_config = [1]+state_[0:-1]
        validation_acc = self.nasbench[tuple(arch_config)]
        return validation_acc[self.val_at_epoch]

    def random_action(self):
        edit_node_id = self.state[-1]
        bounds = self.state_space[edit_node_id]
        return np.random.randint(bounds[0],bounds[1]+1)