# rl-hockey
Code for RL agents in a air hockey environment - University of Tuebingen RL course competition 

[hockey environment](https://github.com/martius-lab/hockey-env/tree/master)

[competition server github](https://github.com/martius-lab/comprl-hockey-agent)

# Proximal Policy Optimization (PPO)
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

# Additional info
[2023 competition leaderboard](http://al-hockey.is.tuebingen.mpg.de/)