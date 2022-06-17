import pickle
import numpy as np
import os
import argparse
from synthetic_mdp_envs import MultiFidEnvs
from agent import MonteCarloAgent
from mpi4py import MPI

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-states', type=int, default=200, help="number of states")
    parser.add_argument('-actions', type=int, default=8, help="number of actions")
    parser.add_argument('-snr', type=float, default=2, help="target snr")
    parser.add_argument('-discount', type=float, default=0.99, help="discount factor")
    parser.add_argument('-epsilon', type=float, default=0.1, help="exploration noise for epsilon soft policy")
    parser.add_argument('-n_low_sa', type=int, default=1, help="number of trajectories per state action pair")
    parser.add_argument('-train_ep', type=int, default=10000, help="number of training episodes")
    parser.add_argument('-test_ep', type=int, default=200, help="number of test episodes")
    parser.add_argument('-test_freq', type=int, default=50, help="testing frequency, every how many training episodes policy is tested")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    states = args.states
    actions = args.actions
    snr = args.snr
    snr_db = float("{0:.2f}".format(np.log10(snr)*10))
    seed = rank


    folder = './data/Exp_s_'+str(states)+'_a_'+str(actions)+'_snr_'+str(snr)+'_n_low_sa_'+str(args.n_low_sa)
    isdir = os.path.isdir(folder+'/')
    if not isdir and rank==0:
        os.makedirs(folder)


    multifid_obj = MultiFidEnvs(states, actions, targetSNR=args.snr,seed=seed)
    env_hi, env_low, P_SNR,R_SNR, mean_delta_R, mean_delta_P = multifid_obj.get_envs()
    prob_descr = dict(P_SNR=P_SNR,R_SNR=R_SNR, mean_delta_R=mean_delta_R, mean_delta_P=mean_delta_P)

    #for synthetic MDPs, the state-action space is the same across the two environments
    #but the dynamics and reward functions are different
    def low_to_hi_map(state):
        return state

    def transform_to_s_low(state):
        return state

    print("Training Monte Carlo RL (MCRL)")
    agent = MonteCarloAgent(low_to_hi_map,transform_to_s_low, seed=seed, discount=args.discount, epsilon=args.epsilon,n_low_sa=args.n_low_sa,train_episodes=args.train_ep,test_freq=args.test_freq,test_episodes=args.test_ep)
    high_fidility_result = agent.MonteCarloRL(env=env_hi)

    print("Training Multi-Fidelity Monte Carlo RL (MFMCRL)")
    agent = MonteCarloAgent(low_to_hi_map,transform_to_s_low, seed=seed, discount=args.discount, epsilon=args.epsilon,n_low_sa=args.n_low_sa,train_episodes=args.train_ep,test_freq=args.test_freq,test_episodes=args.test_ep)
    multi_fidility_result = agent.MultiFidelityMonteCarloRL(env_hi, env_low)

    all_results = dict()
    all_results['HF'] = high_fidility_result
    all_results['MF'] = multi_fidility_result
    all_results['prob_descr'] = prob_descr


    ##write to disk
    save_file = folder+'/seed_'+str(seed)
    with open(save_file, "wb") as file:
        pickle.dump(all_results, file)
    print("Problem Specs: States {}, Actions {}, SNR dB {}, Seed {}".format(states, actions,snr_db,seed))
    print(prob_descr)
