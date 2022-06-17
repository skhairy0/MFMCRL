import numpy as np
from itertools import product

class NASBenchStateTransformer:
    def __init__(self, reduced_search_space = False):
        self.config_to_state_hi,self.state_to_config_hi, self.state_pointer_hi = dict(),dict(),0
        self.config_to_state_low,self.state_to_config_low, self.state_pointer_low = dict(),dict(),0
        self.reduced_search_space = reduced_search_space
        possible_arch_config = list(product([0,1,2,3,4], repeat=6)) 
        possible_arch_config = [list(x) for x in possible_arch_config]
        possible_arch_config = [tuple(x[0]+[x[1]]) for x in list(product(possible_arch_config,[0,1,2,3,4,5,6]))]
        for config in possible_arch_config:
            self.map_config_to_state_hi(config)
        self.low_to_hi_dict = dict()
        for s_hi in np.arange(len(possible_arch_config)):
            s_low = self.transform_to_s_low(s_hi)
            if s_low not in self.low_to_hi_dict:
                self.low_to_hi_dict[s_low] = [s_hi]
            else:
                self.low_to_hi_dict[s_low].append(s_hi)

    def low_to_hi_map(self,s_low):
        return self.low_to_hi_dict[s_low]

    def map_config_to_state_hi(self, config):
        if tuple(config) in self.config_to_state_hi:
            return self.config_to_state_hi[tuple(config)]
        else:
            self.config_to_state_hi[tuple(config)] = self.state_pointer_hi
            self.state_to_config_hi[self.state_pointer_hi] = tuple(config)
            self.state_pointer_hi += 1
            return self.state_pointer_hi - 1

    def map_state_to_config_hi(self, state):
        if state in self.state_to_config_hi:
            return self.state_to_config_hi[state]
        else:
            raise Exception("Inverse Map is Not Available")

    def map_config_to_state_low(self, config):
        if tuple(config) in self.config_to_state_low:
            return self.config_to_state_low[tuple(config)]
        else:
            self.config_to_state_low[tuple(config)] = self.state_pointer_low
            self.state_to_config_low[self.state_pointer_low] = tuple(config)
            self.state_pointer_low += 1
            return self.state_pointer_low - 1

    def map_state_to_config_low(self, state):
        if state in self.state_to_config_low:
            return self.state_to_config_low[state]
        else:
            raise Exception("Inverse Map is Not Available")

    def transform_to_s_low(self, s_hi):
        if self.reduced_search_space:
            config_hi = self.map_state_to_config_hi(s_hi)
            node_hi = config_hi[-1]
            if node_hi == 6:
                node_low = 5 
            elif node_hi >= 1 and node_hi<= 5: 
                node_low = node_hi - 1
            else:
                node_low = -1   
            config_low = list(config_hi)[1:-1]+[node_low] 
            s_low = self.map_config_to_state_low(config=config_low)
        else:
            config_hi = self.map_state_to_config_hi(s_hi)
            config_low = config_hi
            s_low = self.map_config_to_state_low(config=config_low)
        return s_low
