import numpy as np
import random
import copy

class MDPGen:
    def __init__(self,seed=0):
        np.random.seed(seed)
        random.seed(seed)

    def rand(self,states, actions, p_terminal=0.1):
        """
        Generate random dense ``P`` and ``R``.
        Parameters
        ----------
        states : int
            Number of states (> 1)
        actions : int
            Number of actions (> 1)
        """
        # definition of transition matrix : square stochastic matrix
        assert states > 1, "The number of states S must be greater than 1."
        assert actions > 1, "The number of actions A must be greater than 1."

        P = np.zeros((actions, states, states))
        # definition of reward matrix (values between -1 and +1)
        R = np.zeros((actions, states))

        for action in range(actions):
            for state in range(states):
                # create a random mask
                m = np.random.random(states)
                r = np.random.random()
                m[m <= r] = 0
                m[m > r] = 1

                # Make sure that there is atleast one transition in each state
                if m[0:-1].sum() == 0:
                    m[np.random.randint(0, states-1)] = 1
                
                if state == states -1:
                    #this is the terminal state
                    P[action][state] = np.zeros(states)
                    P[action][state][-1] = 1.0
                    R[action][state] = 0.0
                else:
                    P[action][state][0:-1] = m[0:-1] * np.random.random(states-1)
                    P[action][state][0:-1] = (P[action][state][0:-1] / P[action][state][0:-1].sum())*(1-p_terminal)
                    P[action][state][-1] = p_terminal
                    assert np.isclose(np.sum(P[action][state]), 1.0), "State transition row does sums to {}".format(np.sum(P[action][state]))
                    R[action][state] = m[np.random.randint(0, states)] * np.random.random()
        return (P, R)

class ENVSim:
    def __init__(self,P,Ri,no_init_states=10, seed=0):
        self.Pija = P
        np.random.seed(seed)
        random.seed(seed)
        self.Ria = Ri.T
        self.timestep = 0
        self.state_space = np.arange(0,self.Ria.shape[0])
        self.state = random.choice(self.state_space)
        self.state_dim = self.Ria.shape[0]
        self.action_dim = self.Ria.shape[1]
        self.no_init_states = no_init_states

    def reset(self,init_state=None):
        if init_state is None:
            self.state = random.choice(self.state_space[0:self.no_init_states])
        else:
            self.state = init_state
        return self.state

    def step(self,action):
        assert action <= self.Ria.shape[1] and action >= 0, "invalid action"
        imm_reward = self.Ria[self.state,action]
        dist_next_states = self.Pija[action,self.state,:].ravel()
        next_state = random.choices(population=self.state_space,weights=dist_next_states)[0]
        self.state = next_state
        done = True
        for a in range(self.action_dim):
            if self.Pija[a,self.state,self.state]!=1.0:
                done = False
                break
        return self.state, imm_reward,done

    def evaluate(self,state,action):
        return self.Ria[state,action]

    def random_action(self):
        return random.choice(np.arange(0,self.action_dim))

class MultiFidEnvs:
    def __init__(self,states, actions, p_terminal=0.1,targetSNR=2,seed=0):
        self.states = states
        self.actions = actions
        self.p_terminal = p_terminal
        self.targetSNR = targetSNR
        self.seed = seed

    def get_envs(self):
        mdp_gen = MDPGen(seed=self.seed)
        P_hi,R_hi = mdp_gen.rand(states=self.states, actions=self.actions,p_terminal=self.p_terminal)
        R_noise, R_SNR = self.get_noise_scale(arr=R_hi,targetSNR=self.targetSNR)
        R_low = copy.deepcopy(R_hi) + np.random.normal(0,R_noise,size=R_hi.shape)
        P_noise, P_SNR = self.get_noise_scale(arr=P_hi,targetSNR=self.targetSNR)
        P_low = copy.deepcopy(P_hi) + np.random.normal(0,P_noise,size=P_hi.shape)
        
        for action in range(self.actions):
            for state in range(self.states):
                if state == self.states -1:
                    #this is the terminal state
                    P_low[action][state] = np.zeros(self.states)
                    P_low[action][state][-1] = 1.0
                    R_low[action][state] = 0.0
                else:
                    P_low[action][state] = P_low[action][state] / P_low[action][state].sum()
                assert np.isclose(np.sum(P_low[action][state]), 1.0), "State transition row does sums to {}".format(np.sum(P_low[action][state]))
        
        env_high = ENVSim(P=P_hi,Ri=R_hi,seed=self.seed)
        env_low = ENVSim(P=P_low,Ri=R_low,seed=self.seed)
        mean_delta_R = np.abs(R_hi-R_low).mean()
        mean_delta_P = np.abs(P_hi-P_low).mean()
        return env_high, env_low, P_SNR,R_SNR, mean_delta_R, mean_delta_P

    def get_noise_scale(self,arr,targetSNR):
        SNRS = []
        NOISE_SCALES = np.arange(0.0001,2,0.0001)
        for noise in NOISE_SCALES:
            SNR = (arr**2/noise**2).mean()
            SNRS.append(SNR)
        SNRS = np.asarray(SNRS)
        SNR_ = SNRS[np.argmin(np.abs(SNRS-targetSNR))]
        noise_ = NOISE_SCALES[np.argmin(np.abs(SNRS-targetSNR))]
        return noise_, SNR_