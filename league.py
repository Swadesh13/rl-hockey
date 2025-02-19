from henv.env import BasicOpponent, HockeyEnv_SB3
from ppo.ppo import PPO_HockeyAgent
from sac.sac import SAC_HockeyAgent
from utils.league import League, load_saved_models
from utils.parsing import get_config_from_args, parse_args

args = parse_args()
cfg = get_config_from_args(args, cfgnode=True)

env = HockeyEnv_SB3(
    False, cfg.environment.additional_rewards, cfg.environment.reward_multiplier
)

# agent = SAC_HockeyAgent(env, cfg)
agent = PPO_HockeyAgent(env, cfg)

models = load_saved_models(sac=False, td3=False, ppo=True, env=env)
print("\nModels loaded:")
models.append(BasicOpponent(weak=False))
print(models, "\n")

league = League(
    agent,
    models,
    max_score=20,
    additional_rewards=cfg.environment.additional_rewards,
    reward_multiplier=cfg.environment.reward_multiplier,
)


league.train_agent_league(
    rounds=20,
    eval_rounds=5,
    update_opp_every_rounds=4,
    show_score_rounds=1,
    total_timesteps=100_000,
    log_interval=cfg.training.log_interval,
)
