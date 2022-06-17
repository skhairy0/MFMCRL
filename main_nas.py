import pickle
import os
import argparse
from nasbench201_envs import NASbenchEnv,ReducedNASbenchEnv
from utils import NASBenchStateTransformer
from agent import MonteCarloAgent
from mpi4py import MPI

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-hi_dataset', type=int, default=0, help="dataset index of high fidelity environment")
    parser.add_argument('-low_dataset', type=int, default=0, help="dataset index of low fidelity environment")
    parser.add_argument('-hi_valAtEpoch', type=int, default=200, help="epoch at which validation accuracy is used as a reward in high fidelity environment")
    parser.add_argument('-low_valAtEpoch', type=int, default=10, help="epoch at which validation accuracy is used as a reward in low fidelity environment")
    parser.add_argument('-low_reduced_search_space', type=int, default=1, help="0 means both low and high fidelity have the same dynamics/state space. Otherwise, low fidility will use the env with reduced search space")
    parser.add_argument('-n_low_sa', type=int, default=10, help="number of low fid traj per state-action pair")
    parser.add_argument('-train_ep', type=int, default=10000, help="number of training episodes")
    parser.add_argument('-discount', type=float, default=0.99, help="discount factor")
    parser.add_argument('-epsilon', type=float, default=0.1, help="exploration noise for epsilon soft policy")
    parser.add_argument('-test_ep', type=int, default=200, help="number of test episodes")
    parser.add_argument('-test_freq', type=int, default=50, help="testing frequency, every how many training episodes policy is tested")


    args = parser.parse_args()
    hi_dataset = ['cifar10', 'cifar100', 'ImageNet16-120'][args.hi_dataset]
    low_dataset = ['cifar10', 'cifar100', 'ImageNet16-120'][args.low_dataset]
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    seed = rank

    folder = './data/hi_'+hi_dataset+'_'+str(args.hi_valAtEpoch)+'_low_'+low_dataset+'_'+str(args.low_valAtEpoch)+'_train_'+str(args.train_ep)+'_nlow_'+str(args.n_low_sa)
    isdir = os.path.isdir(folder+'/')
    if not isdir and rank==0:
        os.makedirs(folder)

    if args.low_reduced_search_space == 0:
        state_transformer = NASBenchStateTransformer(reduced_search_space=False)
    else:
        state_transformer = NASBenchStateTransformer(reduced_search_space=True)

    env_hi = NASbenchEnv(data_set=hi_dataset, val_at_epoch=args.hi_valAtEpoch, config_to_state=state_transformer.map_config_to_state_hi,state_to_config= state_transformer.map_state_to_config_hi, seed=seed)

    if args.low_reduced_search_space == 0:
        env_low = NASbenchEnv(data_set=low_dataset, val_at_epoch=args.low_valAtEpoch, config_to_state=state_transformer.map_config_to_state_low,state_to_config= state_transformer.map_state_to_config_low, seed=seed)
    else:
        env_low = ReducedNASbenchEnv(data_set=low_dataset, val_at_epoch=args.low_valAtEpoch, config_to_state=state_transformer.map_config_to_state_low,state_to_config= state_transformer.map_state_to_config_low, seed=seed)

    low_to_hi_map = state_transformer.low_to_hi_map
    transform_to_s_low =state_transformer.transform_to_s_low


    print("Training Monte Carlo RL (MCRL)")
    agent = MonteCarloAgent(low_to_hi_map,transform_to_s_low, seed=seed, discount=args.discount, epsilon=args.epsilon,n_low_sa=args.n_low_sa,train_episodes=args.train_ep,test_freq=args.test_freq,test_episodes=args.test_ep)
    high_fidility_result = agent.MonteCarloRL(env=env_hi)

    print("Training Multi-Fidelity Monte Carlo RL (MFMCRL)")
    agent = MonteCarloAgent(low_to_hi_map,transform_to_s_low, seed=seed, discount=args.discount, epsilon=args.epsilon,n_low_sa=args.n_low_sa,train_episodes=args.train_ep,test_freq=args.test_freq,test_episodes=args.test_ep)
    multi_fidility_result = agent.MultiFidelityMonteCarloRL(env_hi, env_low)

    all_results = dict()
    all_results['HF'] = high_fidility_result
    all_results['MF'] = multi_fidility_result


    ##write to disk
    save_file = folder+'/seed_'+str(seed)
    with open(save_file, "wb") as file:
        pickle.dump(all_results, file)
    print("Training is complete.")
