from abc import abstractmethod
from henv.env import HockeyEnv_SB3
from fvcore.common.config import CfgNode


class HockeyAgent:
    def __init__(self, env: HockeyEnv_SB3, config: CfgNode):
        """
        Initializes the agent for the Hockey environment.

        Parameters:
        - env: The environment instance.
        - config: Configuration node (CfgNode).
        """
        self.env = env
        self.config = config
        if not self.config.training.model_path:
            raise ValueError("Model path for training is not specified in the configuration.")
        self.model_path = self.config.training.model_path

        self.model = None

    def train(self, total_timesteps: int = None, log_interval: int = None, progress_bar: bool = False):
        """
        Trains the PPO model.

        Parameters:
        - total_timesteps: Total timesteps for training.
        """
        if not self.model:
            raise ValueError("Model not loaded!")
        tt = total_timesteps or self.config.training.total_timesteps
        li = log_interval or self.config.training.log_interval
        if self.config.logging.verbose:
            print("Starting training...")
        self.model.learn(total_timesteps=tt, log_interval=li, progress_bar=progress_bar)
        if self.config.logging.verbose:
            print("Training complete.")

    def evaluate(self, num_episodes: int = 10, render_mode: str = "human", opponent_right=None):
        """
        Evaluates the trained model in the environment.

        Parameters:
        - num_episodes: Number of episodes to evaluate (overrides config if provided).
        - render_mode: Mode for rendering the environment (overrides config if provided).
        - opponent_right: Optional opponent for the evaluation.

        Returns:
        - mean_reward: Mean reward over all episodes.
        - std_reward: Standard deviation of rewards over all episodes.
        """
        from utils.evaluate import eval_agent

        return eval_agent(
            self.model,
            opponent_right=opponent_right,
            num_episodes=num_episodes,
            render_mode=render_mode,
        )

    def save(self, save_path: str = None):
        """
        Saves the trained SB3 model.
        """
        path = save_path or self.model_path
        self.model.save(path)
        print(f"Model saved at {path}")

    @abstractmethod
    def load(self, load_path: str = None):
        """
        Loads the trained SB3 model.
        """
        pass

    def predict(self, obs, deterministic=True):
        """
        Predict action for a given function
        """
        return self.model.predict(obs, deterministic=deterministic)
