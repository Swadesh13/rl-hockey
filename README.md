# Reinforcement Learning in a Hockey Environment (PhiZero) 
This work was done as part of the Reinforcement Learning (RL) course at the University of Tübingen in the winter semester 2024/2025. The goal was to develop an RL agent for a 2D two-player hockey game that can beat two basic opponent players as well as all other student's agents in a final tournament.

More on the hockey environment can be found [here](https://github.com/martius-lab/hockey-env/tree/master)

The working behind the tournament server can be found [here](https://github.com/martius-lab/comprl-hockey-agent)

Certificate for being in the top 7% of participants can be found [here](assets/8th_place_certificate.pdf)

We presented both on- and off-policy solutions for this problem. In particular, these are the algorithms that each of the authors have implemented:

- Proximal Policy Optimization (PPO) [Vojtěch Sýkora]
- Soft Actor-Critic (SAC) [Swadesh Jana]
- Twin Delayed DDPG (TD3+4=TD7) [Rojan Abolhassani]

An extensive report containing detailed algorithm, modifications, and experiments descriptions can be found [here]().

## Getting Started

First install the requirements for python >=3.10 
```
pip install -r requirements.txt
```

## Proximal Policy Optimization (PPO)
make sure you are in the top directory rl-hockey 

```bash
python -m ppo.ppo --config configs/ppo_hockey.yaml -q --train
python -m ppo.ppo --config configs/ppo_hockey.yaml -q --eval
```
Our best models with each modification can be found under `models/ppo` where you can find `.yaml` config files corresponding to each model. If you want to evaluate any of those, add the path to the yaml file of your desired model from `models/ppo` to the command line arguments. The default `configs/ppo_hockey.yaml` uses the saved `models/ppo/ppo_offensive_pressure.yaml` model since this one performed the best.
In the `models/ppo` folder can be seen the evaluation against many opponents of each of the PPO models.

To evaluate PPO against all opponents go to `ppo/ppo.py` and set `EVAL_AGAINST_ALL = True` and then run again 
```
python -m ppo.ppo --config configs/ppo_hockey.yaml -q --eval
```

To train PPO against all models check out `ppo/train_against_all.py` or run
```
python -m ppo.train_against_many --config configs/ppo_hockey.yaml -q
```

## Soft Actor Critic (SAC)

make sure you are in the top directory rl-hockey 

```bash
python -m sac.sac --config configs/sac_hockey.yaml -q --train
```

For league:

```bash
python -m league.py --config configs/sac_hockey.yaml -q
```

## TD7
make sure you are in the top directory rl-hockey, for training a model run:

```bash
python -m td7.hockey_main --config configs/td7_hockey.yaml --train
```
for evaluation run:

```bash
python -m td7.hockey_main --config configs/td7_hockey.yaml --eval
```
You can choose the target to evaluate against in the configs/td7_hockey.yaml.

## Authors
Vojtěch Sýkora, Rojan Abolhassani, Swadesh Jana

