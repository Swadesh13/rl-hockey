import os
from glob import glob
from typing import List
import numpy as np
from henv.env import HockeyEnv_SB3, BasicOpponent
from utils.parsing import save_config, get_default_ppo_config, get_default_sac_config, get_default_td3_config


class LeagueHockeyEnv(HockeyEnv_SB3):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.opponent = None

    def set_opponent(self, opponent):
        self.opponent = opponent

    def step(self, action):
        if self.opponent:
            return super().step(action)
        else:
            raise ValueError("Opponent not set")


class League:
    """
    Creates a league where a main agent trains against multiple opponents.
    """

    def __init__(self, agent, opponents: List = [], save_dir=None, **env_args):
        self.save_dir = save_dir
        if not self.save_dir:
            self.save_dir = f"./logs/league_{len(glob('./logs/league_*'))}"
        agent.model.tensorboard_log = os.path.join(self.save_dir, "tensorboard")
        print("Saving files at:", self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        save_path = os.path.join(self.save_dir, "model_0")

        save_config(os.path.join(self.save_dir, "config.yaml"), agent.config)

        agent.save(save_path)
        self.env = LeagueHockeyEnv(**env_args)
        agent = type(agent)(self.env, agent.config)
        agent.load(save_path)

        self.current_agent = agent
        self.opponents = opponents
        if not self.opponents:
            self.opponents.append(BasicOpponent(weak=False))
        # 0 - loss, 1 - draw, 2 - win
        with open(f"{self.save_dir}/init_opp.txt", "w") as f:
            for opp in self.opponents:
                f.write(f"{opp}\n")

        self.scores = [0] * len(self.opponents)

        self.curr_idx = None

    def sample_opponent(self):
        probs = np.array(self.scores)
        if all(probs == np.max(probs)):
            self.curr_idx = np.random.choice(len(self.opponents))
        else:
            probs = np.max(probs) - probs
            probs = probs / np.sum(probs)
            self.curr_idx = np.random.choice(len(self.opponents), p=probs)
        return self.opponents[self.curr_idx]

    def update_scores(self, score):
        self.scores[self.curr_idx] += score

    def eval_agent(self, opponent):
        obs, info = self.env.reset()
        obs_agent2 = self.env.obs_agent_two()

        while True:
            a1, _ = self.current_agent.predict(obs, deterministic=True)
            a2, _ = opponent.predict(obs_agent2, deterministic=True)
            obs, reward, done, _, info = self.env.step(np.hstack([a1, a2]))
            obs_agent2 = self.env.obs_agent_two()

            if done:
                break

        if info["winner"] == 0:
            self.update_scores(0.5)
        elif info["winner"] == 1:
            self.update_scores(2.0)

    def train_agent_league(self, rounds, eval_rounds=1, update_opp_every_rounds=5, show_score_rounds=1, **train_kwargs):
        print("Starting League...")

        for i in range(1, rounds + 1):
            opponent = self.sample_opponent()
            print("Opponent sampled:", opponent)
            self.env.set_opponent(opponent)
            self.current_agent.train(**train_kwargs)
            for _ in range(eval_rounds):
                self.eval_agent(opponent)
            if update_opp_every_rounds and i % update_opp_every_rounds == 0:
                save_path = os.path.join(self.save_dir, f"model_{i}")
                self.current_agent.save(save_path)
                print(f"Updated model saved at {save_path}")
                env = self.current_agent.env
                cfg = self.current_agent.config
                new_opp = type(self.current_agent)(env, cfg)
                new_opp.load(save_path)
                self.opponents.append(new_opp)
                self.scores.append(0)
            if show_score_rounds and i % show_score_rounds == 0:
                print(f"Score after {i} rounds")
                print(*list(zip(self.opponents, self.scores)), sep="\n")
        print("Ending League...")

        with open(f"{self.save_dir}/final_opp.txt", "w") as f:
            for opp, sc in zip(self.opponents, self.scores):
                f.write(f"{opp} - {sc}\n")


def load_saved_models(sac=False, td3=False, ppo=False, env=None, cfg=[]):
    """
    Load saved models for inference / training other models
    sac/td3/ppo : bool, whether to load models
    env: Hockey env
    cfg: List of config files. len(cfg) is at max 3 - 1 each for sac, td3, ppo
    """
    models = []
    if not env:
        from henv.env import HockeyEnv_SB3

        env = HockeyEnv_SB3(False)

    def _load(env, cfg, root, model_class):
        paths = glob(root)
        for p in paths:
            m = model_class(env, cfg)
            m.load(p[:-4])
            yield m

    if sac:
        from sac.sac import SAC_HockeyAgent

        models.extend(list(_load(env, cfg[0] if cfg else get_default_sac_config(), "models/sac/*.zip", SAC_HockeyAgent)))
    if td3:
        from td3.td3 import TD3_HockeyAgent

        models.extend(list(_load(env, cfg[1] if len(cfg) == 2 else get_default_td3_config(), "models/td3/*.zip", TD3_HockeyAgent)))
    if ppo:
        from ppo.ppo import PPO_HockeyAgent

        models.extend(list(_load(env, cfg[2] if len(cfg) == 3 else get_default_ppo_config(), "models/ppo/*.zip", PPO_HockeyAgent)))
    return models


if __name__ == "__main__":
    from utils.parsing import parse_args, get_config_from_args, print_config, print_args, save_config

    args = parse_args()
    cfg = get_config_from_args(args, cfgnode=True)

    if not args.quiet:
        print_config(cfg)
        print_args(args)

    from henv.env import HockeyEnv_SB3

    env = HockeyEnv_SB3(False, cfg.environment.additional_rewards, cfg.environment.reward_multiplier)

    from sac.sac import HockeySACAgent

    agent = HockeySACAgent(env, cfg)
    # Load the agent as well
    agent_copy = HockeySACAgent(env, cfg)

    league = League(
        agent,
        [agent_copy, BasicOpponent(False)],
        additional_rewards=cfg.environment.additional_rewards,
        reward_multiplier=cfg.environment.reward_multiplier,
    )

    league.train_agent_league(10, 2, 5, total_timesteps=1000, log_interval=cfg.training.log_interval)
