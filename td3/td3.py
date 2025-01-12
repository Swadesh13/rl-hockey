import os
from stable_baselines3 import TD3
from henv.hockey_agent import HockeyAgent


class TD3_HockeyAgent(HockeyAgent):
    def __init__(self, env, config):
        """
        Initializes the TD3 agent for the Hockey environment.

        Parameters:
        - env: The environment instance.
        - config: Configuration node (CfgNode).
        """
        super().__init__(env, config)
        hyperparameters = self.config.model.hyperparameters
        self.model = TD3(
            "MlpPolicy",
            self.env,
            **hyperparameters,
            verbose=self.config.logging.verbose,
        )

    def load(self, load_path=None):
        """
        Loads the TD3 model.
        """
        path = load_path or self.model_path
        if os.path.exists(f"{path}.zip"):
            self.model = TD3.load(path, env=self.env)
            print(f"Model loaded from {path}")
        else:
            print(f"No model found at {path}. Starting with a new model.")


if __name__ == "__main__":
    from utils.parsing import parse_args, get_config_from_args, print_config, print_args

    args = parse_args()
    cfg = get_config_from_args(args, cfgnode=True)

    if not args.quiet:
        print_config(cfg)
        print_args(args)

    from henv.env import HockeyEnv_SB3

    env = HockeyEnv_SB3.make_vec_env(n_envs=cfg.environment.n_envs)

    agent = TD3_HockeyAgent(env, config=cfg)

    if args.train:
        agent.train(total_timesteps=cfg.training.total_timesteps)
        agent.save()

    if args.eval:
        agent.load()
        agent.evaluate(num_episodes=args.eval_episodes)
