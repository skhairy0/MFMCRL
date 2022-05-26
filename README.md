# Multifidelity Reinforcement Learning with Control Variates

This repository is the official implementation of the paper "Multifidelity Reinforcement Learning with Control Variates."

## Requirements


- Python 3.8.11
- NumPy 1.19.5
- mpi4py 3.1.2

## Training and Evaluation 

To train the agent using both MCRL and MFMCRL and test the agent's performance on the synthetic MDP problem in the paper, run this command:

```train
mpirun -np 36 python main_synthetic.py -states 200 -actions 8 -snr 2.0 -n_low_sa 1 -train_ep 10000
```
To train the agent using both MCRL and MFMCRL and test the agent's performance on the NAS problem with the Imagenet dataset such that the high and low fidelity environments have the same search space (case (i) in the paper), run this command:

```train
mpirun -np 36 python main_nas.py -hi_dataset 2 -low_dataset 2 -low_reduced_search_space 0 -n_low_sa 1 -train_ep 10000
```

## Results

Running either of the previous commands will yield a pickled dictionary of results in the data folder. By loading this dictionary into "results", 

- results["HF"]["test_ep_rewards"]: test episode rewards of the MCRL agent trained on the single high-fidelity environment.
- results["MF"]["test_ep_rewards"]: test episode rewards of the MFMCRL agent trained on the multifidelity environment.
