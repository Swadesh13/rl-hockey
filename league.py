from utils.parsing import parse_args, get_config_from_args
from henv.env import HockeyEnv_SB3, BasicOpponent
from sac.sac import SAC_HockeyAgent
from utils.league import League, load_saved_models

args = parse_args()
cfg = get_config_from_args(args, cfgnode=True)

env = HockeyEnv_SB3(False, cfg.environment.additional_rewards, cfg.environment.reward_multiplier)

agent = SAC_HockeyAgent(env, cfg)

models = load_saved_models(True, True, True, env)
models.append(BasicOpponent(False))

league = League(
    agent,
    models,
    additional_rewards=cfg.environment.additional_rewards,
    reward_multiplier=cfg.environment.reward_multiplier,
)

league.train_agent_league(10, 5, 2, total_timesteps=100000, log_interval=cfg.training.log_interval)
