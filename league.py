from henv.env import BasicOpponent, HockeyEnv_SB3
from sac.sac import SAC_HockeyAgent
from utils.league import League, load_saved_models
from utils.parsing import get_config_from_args, parse_args

args = parse_args()
cfg = get_config_from_args(args, cfgnode=True)

env = HockeyEnv_SB3(
    False, cfg.environment.additional_rewards, cfg.environment.reward_multiplier
)

agent = SAC_HockeyAgent(env, cfg)

models = [{"name": "basic_opponent_strong", "model": BasicOpponent(weak=False)}]
# models.extend(load_saved_models(sac=True, td7=True, ppo=False, env=env))
print("\nModels loaded:")
print(*[m["name"] for m in models], sep="\n")

league = League(
    agent,
    models,
    sample_opp_randomly=False,
    # max_score=10,
    additional_rewards=cfg.environment.additional_rewards,
    reward_multiplier=cfg.environment.reward_multiplier,
)

league.train_agent_league(
    rounds=20,
    eval_rounds=5,
    eval_all=True,
    update_opp_every_rounds=[5],
    show_score_rounds=1,
    total_timesteps=100_000,
    log_interval=cfg.training.log_interval,
    discard_randomly=True,
    train_against_basic=True,
    # timestep_scheduler_factor=0.1,
)
