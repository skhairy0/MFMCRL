import numpy as np
import random
from tqdm import tqdm
from itertools import product

class MonteCarloAgent:
    def __init__(self, low_to_hi_map,transform_to_s_low, seed=0, discount=0.99, epsilon=0.1,n_low_sa=10,train_episodes=200,test_freq=50,test_episodes=200):
        np.random.seed(seed)
        random.seed(seed)
        self.discount = discount
        self.epsilon = epsilon
        self.train_episodes = train_episodes
        self.test_freq = test_freq
        self.test_episodes = test_episodes
        self.n_low_sa = n_low_sa
        self.low_to_hi_map = low_to_hi_map #one to many map
        self.transform_to_s_low = transform_to_s_low #many to one map
                              
              
    def generate_episode(self,env, q_table, initial_state_action=None, epsilon_greedy=True, low_fid_episode=False):
        s_list = []
        r_list = []
        a_list = []
        if initial_state_action is not None:
            state = env.reset(init_state=initial_state_action[0])
        else:
            state = env.reset()
        s_list.append(state)
        done, first_action = False, True
        while not done:
            if epsilon_greedy: #epsilon greedy policy
                if np.random.uniform()<= 1-self.epsilon+self.epsilon/env.action_dim:
                    if not low_fid_episode:
                        q_row = q_table[state]
                    else:
                        s_hi_s = self.low_to_hi_map(state)
                        q_row = np.sum(q_table[s_hi_s],axis=0)
                    a = np.random.choice(np.flatnonzero(q_row == q_row.max()))
                else:
                    a = env.random_action()
            else:
                if not low_fid_episode:
                    q_row = q_table[state]
                else:
                    s_hi_s = self.low_to_hi_map(state)
                    q_row = np.sum(q_table[s_hi_s],axis=0)          
                a = np.random.choice(np.flatnonzero(q_row == q_row.max()))
            #mask first action only if we have a specified action
            if initial_state_action is not None and first_action:
                first_action = False
                a = initial_state_action[1]
            a_list.append(a)
            next_state,imm_reward,done = env.step(action=a)
            r_list.append(imm_reward)
            state = next_state
            s_list.append(state)
        return np.asarray(s_list[:-1]),np.asarray(a_list),np.asarray(r_list)
      
    def test_policy(self,env, q_table, num_episode=5):
        mean_ep_rewards = []
        for _ in range(num_episode):
            _,_,r_episode = self.generate_episode(env=env, q_table=q_table, epsilon_greedy=False)
            mean_ep_rewards.append(np.mean(r_episode))
        return np.mean(mean_ep_rewards)

    def MonteCarloRL(self, env):
        #initialization
        return_list = dict()
        q_vals = np.zeros(shape=(env.state_dim,env.action_dim))
        mean_test_ep_rewards = []
        mean_test_ep_rewards.append(self.test_policy(env=env, q_table=q_vals, num_episode=self.test_episodes))
        no_interactions = 0
        for ep in tqdm(range(self.train_episodes)):
            s_episode,a_episode,r_episode = self.generate_episode(env= env, q_table=q_vals, epsilon_greedy=True)
            no_interactions += len(s_episode)+1
            s_a_pairs = list(zip(s_episode,a_episode))
            G = 0
            for t in np.arange(len(s_episode)-1,-1,-1):
                G = self.discount*G + r_episode[t]
                if (s_episode[t],a_episode[t]) not in s_a_pairs[0:t]:
                    if (s_episode[t],a_episode[t]) not in return_list:
                        return_list[(s_episode[t],a_episode[t])] = [G]
                    else:
                        return_list[(s_episode[t],a_episode[t])].append(G)
                    q_vals[s_episode[t],a_episode[t]] = np.mean(return_list[(s_episode[t],a_episode[t])] )
            if (ep+1)%self.test_freq==0:
                mean_test_ep_rewards.append(self.test_policy(env=env, q_table=q_vals, num_episode=self.test_episodes))
        return dict(test_ep_rewards=np.asarray(mean_test_ep_rewards))

    def MultiFidelityMonteCarloRL(self, env_hi, env_low):
        return_list_population_low = dict()
        return_list_low = dict()
        return_list_high =  dict()
        q_vals_mfrl = np.zeros(shape=(env_hi.state_dim,env_hi.action_dim))
        mean_test_ep_rewards = []
        mean_test_ep_rewards.append(self.test_policy(env=env_hi, q_table=q_vals_mfrl, num_episode=self.test_episodes))
        no_highfid_interactions = 0
        no_lowfid_interactions = 0
        variance_red_factor = []
        for ep in tqdm(range(self.train_episodes)):
            #generate high fidelity episode
            s_episode,a_episode,r_episode = self.generate_episode(env = env_hi, q_table=q_vals_mfrl, epsilon_greedy=True)
            no_highfid_interactions += len(s_episode)+1
            no_lowfid_interactions += len(s_episode)+1
            s_a_pairs = list(zip(s_episode,a_episode))
            #convert episode to a low fid trajectory using generatiev environment
            s_episode_low,r_episode_low = [],[]
            for pair in s_a_pairs:
                s_low = self.transform_to_s_low(pair[0])
                s_episode_low.append(s_low)
                r_episode_low.append(env_low.evaluate(state=s_low, action=pair[1]))
           
            s_a_pairs_low = list(zip(s_episode_low,a_episode))
            #Obtain n_low_sa samples of low fid returns for each (s_low, a) in that appeared in the low fid trajectory
            return_list_population_low, no_iteractions = self.update_low(env_low, return_list_population_low,s_a_pairs_low,q_vals_mfrl)
            no_lowfid_interactions += no_iteractions

            r_low = np.asarray(r_episode_low)
            r_high = np.asarray(r_episode)
            assert len(r_low) == len(r_high)
            #Estimate high & low fidelity returns based on current episode
            G_high = 0
            G_low = 0
            ep_var_red = []
            for t in np.arange(len(s_episode)-1,-1,-1):
                G_high = self.discount*G_high + r_high[t]
                G_low = self.discount*G_low + r_low[t]
                 #Include low fidelity return estimates based on current trajectory to return_list_population_low
                if (s_episode_low[t],a_episode[t]) not in s_a_pairs_low[0:t]:
                    if (s_episode_low[t],a_episode[t]) not in return_list_population_low:
                        return_list_population_low[(s_episode_low[t],a_episode[t])] = [G_low]
                    else:
                        return_list_population_low[(s_episode_low[t],a_episode[t])].append(G_low)

                if (s_episode[t],a_episode[t]) not in s_a_pairs[0:t]:
                    if (s_episode[t],a_episode[t]) not in return_list_high:
                        return_list_high[(s_episode[t],a_episode[t])] = [G_high]
                        return_list_low[(s_episode[t],a_episode[t])] = [G_low]
                    else:
                        return_list_high[(s_episode[t],a_episode[t])].append(G_high)
                        return_list_low[(s_episode[t],a_episode[t])].append(G_low)

                    #Update q_val estimate for this state-action pair
                    state = s_episode[t]
                    action = a_episode[t]
                    pair = (state,action)
                    t = np.mean(return_list_low[pair])
                    m = np.mean(return_list_high[pair])
            
                    if len(return_list_high[pair]) <= 1 or len(return_list_low[pair]) <=1:
                        #we can't estimate variance if we dont have at least two data points
                        q_vals_mfrl[state,action] = m
                        ep_var_red.append(1)
                        continue

                    tau = np.mean(return_list_population_low[(self.transform_to_s_low(state),action)])

                    var_high = np.std(return_list_high[pair],ddof=1)**2
                    var_low = np.std(return_list_low[pair],ddof=1)**2
                    r_high_pair = np.asarray(return_list_high[pair])
                    r_low_pair = np.asarray(return_list_low[pair])
                    assert len(r_high_pair)==len(r_low_pair)
                    if np.sqrt(var_low)>0 and np.sqrt(var_high)>0:
                        cov_low_high = np.multiply(r_high_pair-r_high_pair.mean(), r_low_pair-r_low_pair.mean()).sum()/(len(r_low_pair)-1)
                        rho = cov_low_high/(np.sqrt(var_low)*np.sqrt(var_high))
                        correction = (tau-t)*rho*np.sqrt(var_high)/np.sqrt(var_low)
                        ep_var_red.append(1-rho**2)
                    else:
                        correction = 0.0
                        ep_var_red.append(1)
                    q_vals_mfrl[state,action] = m + correction

            if (ep+1)%self.test_freq==0:
                mean_test_ep_rewards.append(self.test_policy(env=env_hi, q_table=q_vals_mfrl, num_episode=self.test_episodes))

            variance_red_factor.append(np.mean(ep_var_red))
       
        return dict(test_ep_rewards=np.asarray(mean_test_ep_rewards),low_fid_interactions=no_lowfid_interactions,high_fid_interactions= no_highfid_interactions,variance_red_factor=np.asarray(variance_red_factor))
 
    def update_low(self,env_low, return_list_population_low,s_a_pairs_low,q_vals_mfrl):
        no_iteractions = 0
        unique_sa = set(s_a_pairs_low)
        sa_sample_count = dict()
        for sa in unique_sa:
            sa_sample_count[sa] = 0

        for sa in unique_sa:
            while sa_sample_count[sa]<self.n_low_sa:
                s_episode,a_episode,r_episode = self.generate_episode(env=env_low, q_table=q_vals_mfrl,initial_state_action=sa, epsilon_greedy=True, low_fid_episode = True)
                no_iteractions += len(s_episode)+1
                s_a_pairs = list(zip(s_episode,a_episode))
                G = 0
                for t in np.arange(len(s_episode)-1,-1,-1):
                    G = self.discount*G + r_episode[t]
                    if (s_episode[t],a_episode[t]) not in s_a_pairs[0:t]:
                        if (s_episode[t],a_episode[t]) not in return_list_population_low:
                            return_list_population_low[(s_episode[t],a_episode[t])] = [G]
                        else:
                            return_list_population_low[(s_episode[t],a_episode[t])].append(G)
    
                        if (s_episode[t],a_episode[t]) in sa_sample_count:
                            sa_sample_count[(s_episode[t],a_episode[t])] += 1
        return return_list_population_low, no_iteractions


