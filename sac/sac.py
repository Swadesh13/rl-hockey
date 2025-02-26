import os
from sac.sac_agent import get_SAC_agent, SAC, SAC_PM
from henv.hockey_agent import HockeyAgent


class SAC_HockeyAgent(HockeyAgent):
    def __init__(self, env, config):
        """
        Initializes the SAC agent for the Hockey environment.

        Parameters:
        - env: The environment instance.
        - config: Configuration node (CfgNode).
        """
        super().__init__(env, config)
        self.model = get_SAC_agent(self.env, self.config)

    def load(self, load_path=None):
        """
        Loads the SAC model.
        """
        path = load_path or self.model_path
        if os.path.exists(f"{path}.zip"):
            self.model = (SAC_PM if self.config.model.prioritized_memory else SAC).load(
                path, env=self.env, **self.config
            )
            print(f"Model loaded from {path}")
        else:
            print(f"No model found at {path}. Starting with a new model.")


if __name__ == "__main__":
    from utils.parsing import parse_args, get_config_from_args, print_config, print_args

    args = parse_args()
    cfg = get_config_from_args(args, cfgnode=True)

    # if not args.quiet:
    #     print_config(cfg)
    #     print_args(args)

    from henv.env import HockeyEnv_SB3

    env = HockeyEnv_SB3(
        False,
        cfg.environment.additional_rewards,
        cfg.environment.reward_multiplier,
    )

    for noise in ["brown", "pink", None]:
        cfg.model.noise = noise
        print(cfg.model.noise)
        agent = SAC_HockeyAgent(env, cfg)

        if args.train:
            agent.train(total_timesteps=cfg.training.total_timesteps)
            agent.save()

        if args.eval:
            agent.load()
            agent.evaluate(num_episodes=args.eval_episodes)
