# rl-hockey
Code for RL agents in a air hockey environment - University of Tuebingen RL course competition 

[competition leaderboard](http://al-hockey.is.tuebingen.mpg.de/)

[hockey environment](https://github.com/martius-lab/hockey-env/tree/master)

## Training and testing
make sure you are in the top directory rl-hockey 
for ppo
```bash
python -m ppo.ppo --config configs/ppo_hockey.yaml --train
python -m ppo.ppo --config configs/ppo_hockey.yaml --eval
```

for td3
```bash
cd rl-hockey 
python -m td3.td3 --config configs/td3_hockey.yaml --train
python -m td3.td3 --config configs/td3_hockey.yaml --eval
```

`-q` stops printing the config and args. config is the file config (overriden) and args are just the command line arguments that are not None.

for figuring out how the parser works just do 

```bash
python -m ppo.ppo --help
```