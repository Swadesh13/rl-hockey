from abc import abstractmethod

from fvcore.common.config import CfgNode
from stable_baselines3.common.callbacks import CallbackList

from henv.env import HockeyEnv_SB3
from comprl.client import Agent
import uuid
import numpy as np


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
            raise ValueError(
                "Model path for training is not specified in the configuration."
            )
        self.model_path = self.config.training.model_path

        self.model = None

    def train(
        self,
        total_timesteps: int = None,
        log_interval: int = None,
        progress_bar: bool = False,
        callbacks: list = None,
        tb_log_name: str = None,
    ):
        """
        Trains the PPO model.

        Parameters:
        - total_timesteps: Total timesteps for training.
        - log_interval: Log interval for training progress.
        - progress_bar: Whether to display a progress bar during training.
        - callbacks: List of callbacks to use during training.
        - tb_log_name: str, tensorboard logger name
        """
        if not self.model:
            raise ValueError("Model not loaded!")
        tt = total_timesteps or self.config.training.total_timesteps
        li = log_interval or self.config.training.log_interval
        callback_list = CallbackList(callbacks) if callbacks else None

        if self.config.logging.verbose:
            print("Starting training...")

        self.model.learn(
            total_timesteps=tt,
            log_interval=li,
            progress_bar=progress_bar,
            callback=callback_list,
            tb_log_name=tb_log_name,
        )

        if self.config.logging.verbose:
            print("Training complete.")

    def evaluate(
        self,
        verbose,
        num_episodes: int = 10,
        render_mode: str = "human",
        opponent_right=None,
        modes=["NORMAL"],
        env=None,
    ):
        """
        Evaluates the trained model in the environment.

        Parameters:
        - num_episodes: Number of episodes to evaluate (overrides config if provided).
        - render_mode: Mode for rendering the environment (overrides config if provided).
        - opponent_right: Optional opponent for the evaluation.
        - modes: List of modes to cycle through during evaluation (e.g., ["NORMAL", "TRAIN_SHOOTING", "TRAIN_DEFENSE"]).

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
            modes=modes,
            env=env,
            verbose=verbose,
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

class HockeyCompetetionAgent(Agent):
    def __init__(self, agent : HockeyAgent) -> None:
        super().__init__()

        self.hockey_agent = agent

    def get_step(self, observation: list[float]) -> list[float]:
        # NOTE: If your agent is using discrete actions (0-7), you can use
        # HockeyEnv.discrete_to_continous_action to convert the action:
        #
        # from hockey.hockey_env import HockeyEnv
        # env = HockeyEnv()
        # continuous_action = env.discrete_to_continous_action(discrete_action)
        obs = np.array(observation, dtype=np.float32)  # Ensure correct dtype
        action, _ = self.hockey_agent.predict(obs, deterministic=True)
        return [float(a) for a in action] 

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id))
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"Game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )