from typing import List
import os
from glob import glob
import numpy as np
from henv.env import BasicOpponent, HockeyEnv_SB3, hockey_env
from utils.parsing import save_config


class LeagueHockeyEnv(HockeyEnv_SB3):
    """
    Same class as HockeyEnv_SB3. Just to make sure an opponent is set before playing.
    """

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
    Creates a league where a main agent trains against multiple opponents. Can be self-play as well.

    Args:
        agent: HockeyAgent, the main agent to train.
        opponents: List[HockeyAgent|BasicOpponent], list of opponents to play against.
        save_dir: str, directory to store models.
        sample_opp_randomly: bool, whether to sample randomly weighted by score or the lowest score.
        max_score: float, used only with sample_opp_randomly. Train against all agents upto a max_score.
        env_args: additional environment args for initialization. See HockeyEnv_SB3.
    """

    def __init__(
        self,
        agent,
        opponents: List = [],
        save_dir=None,
        sample_opp_randomly=True,
        max_score=None,
        **env_args,
    ):
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
            self.opponents.append(
                {"name": "basic_opponent_strong", "model": BasicOpponent(weak=False)}
            )
        with open(f"{self.save_dir}/init_opp.txt", "w") as f:
            for opp in self.opponents:
                f.write(f"{opp['name']}\n")

        self.scores = [0] * len(self.opponents)

        self.curr_idx = None
        self.max_score = max_score
        self.sample_opp_randomly = sample_opp_randomly

    def sample_opponent(self, no_basic=False):
        """
        Sample an opponent based on current scores.
        Two choices:
            1. Using np.random weighted by current scores.
            2. Lowest current score.

        Args:
            no_basic: Whether to sample the basic opponent or not.
                Sometimes, too much training on the basic opponent can overfit to a deterministic opponent.
        """
        probs = np.array(self.scores)
        if self.sample_opp_randomly:
            if all(probs == np.max(probs)):
                self.curr_idx = np.random.choice(len(self.opponents))
            else:
                if 0 in probs:
                    self.curr_idx = np.random.choice(np.argwhere(probs == 0).flatten())
                else:
                    probs = np.max(probs) - probs
                    if self.max_score:
                        mask = np.where(np.array(self.scores) >= self.max_score, 0, 1)
                        if sum(mask):
                            probs *= mask
                    probs = probs / np.sum(probs)
                    self.curr_idx = np.random.choice(len(self.opponents), p=probs)
        else:
            inds = np.where(probs == probs.min())[0]
            self.curr_idx = np.random.choice(inds)
        if (
            "basic" in self.opponents[self.curr_idx]["name"]
            and no_basic
            and len(self.opponents) == 1
        ):
            env = self.current_agent.env
            cfg = self.current_agent.config
            new_opp = type(self.current_agent)(env, cfg)
            new_opp.load(os.path.join(self.save_dir, "model_0"))
            self.opponents.append({"name": "model_0", "model": new_opp})
            self.scores.append(0)
            self.curr_idx = 1
        elif (
            "basic" in self.opponents[self.curr_idx]["name"]
            and no_basic
            and not self.sample_opp_randomly
        ):
            # if sampling with lowest score and basic opponent has lowest score
            _temp = self.scores[self.curr_idx]
            _idx = self.curr_idx
            self.scores[_idx] = max(self.scores) + 1
            self.sample_opponent(no_basic)
            self.scores[_idx] = _temp
        elif "basic" in self.opponents[self.curr_idx]["name"] and no_basic:
            self.sample_opponent(no_basic)
        return self.opponents[self.curr_idx]

    def update_scores(self, score):
        """
        Updates the score for the current opponent.
        """
        self.scores[self.curr_idx] += score

    def eval_agent(self, opponent):
        """
        Evaluate the agent with an opponent and return the score.
        """
        env = hockey_env.HockeyEnv()
        obs, info = env.reset()
        obs_agent2 = env.obs_agent_two()

        while True:
            a1, _ = self.current_agent.predict(obs, deterministic=True)
            a2, _ = opponent.predict(obs_agent2, deterministic=True)
            obs, reward, done, _, info = env.step(np.hstack([a1, a2]))
            obs_agent2 = env.obs_agent_two()

            if done:
                break

        if info["winner"] == 0:
            return 0.5
        elif info["winner"] == 1:
            return 2.0
        return 0

    def _eval(self, eval_all, eval_rounds, opp=None):
        """
        Evaluate function.

        Args:
            eval_all: bool, if evaluate all or just the given opponent.
            eval_rounds: int, number of rounds to evaluate.
            opp: HockeyAgent|BasicOpponent|None, opponent to eval against. Only needed if not eval_all.
        """
        if eval_all:
            self.scores = [0] * len(self.opponents)
            for idx, opp in enumerate(self.opponents):
                self.curr_idx = idx
                for _ in range(eval_rounds):
                    self.update_scores(self.eval_agent(opp["model"]))
        else:
            for _ in range(eval_rounds):
                self.update_scores(self.eval_agent(opp["model"]))

    def train_agent_league(
        self,
        rounds,
        eval_rounds=1,
        eval_all=True,
        update_opp_every_rounds=5,
        show_score_rounds=1,
        discard_randomly=False,
        max_opponents=7,
        train_against_basic=True,
        timestep_scheduler_factor=None,
        min_timestamps=25000,
        **train_kwargs,
    ):
        """
        Main League training function.

        Args:
        rounds: int, number of rounds to perform one round of training.
        eval_rounds: int, number of times to play against opponent for evaluation/updating scores.
        eval_all: bool, whether to evaluate against all or current opponent. if False, past evaluation information is stored.
        update_opp_every_rounds: int, introduces self-play by appending the list of opponents with its own version.
        show_score_rounds: int, logging scores to display.
        discard_randomly: bool, whether to discard opponents randomly. This is important for keeping a shorter list of opponents.
        max_opponents: int, max opponents to keep rest discard.
        train_against_basic: bool, whether to train against the basic opponent.
        timestep_scheduler_factor: float, schedule timesteps. Initially can be high and reduce by factor.
        min_timestamps: int, minimum number of timesteps. Only important if using timestep_scheduler_factor
        train_kwargs: training kwargs, see HockeyAgent.
        """

        print("Starting League...")
        if isinstance(update_opp_every_rounds, list):
            update_opp_idx = 0
        self._eval(True, eval_rounds, None)
        print(
            "Initial scores:",
            *[f"{o['name']}: {s}" for o, s in zip(self.opponents, self.scores)],
            sep="\n",
        )
        if timestep_scheduler_factor:
            timestep_scheduler = 1
        for i in range(1, rounds + 1):
            opponent = self.sample_opponent(not train_against_basic)
            print(f"\nROUND {i}: Opponent sampled:", opponent["name"])
            self.env.set_opponent(opponent["model"])
            if i > 1 and timestep_scheduler_factor and "total_timesteps" in train_kwargs:
                timestep_scheduler *= 1 - timestep_scheduler_factor
                train_kwargs["total_timesteps"] = max(
                    min_timestamps,
                    int(train_kwargs["total_timesteps"] * self.timestep_scheduler),
                )
            self.current_agent.train(
                tb_log_name=type(self.current_agent).__name__
                + f"_{i}_"
                + opponent["name"],
                **train_kwargs,
            )
            self._eval(eval_all, eval_rounds, opponent)
            if update_opp_every_rounds and (
                (
                    isinstance(update_opp_every_rounds, list)
                    and (i - sum(update_opp_every_rounds[:update_opp_idx]))
                    % update_opp_every_rounds[update_opp_idx]
                    == 0
                )
                or (
                    isinstance(update_opp_every_rounds, int)
                    and i % update_opp_every_rounds == 0
                )
            ):
                if isinstance(update_opp_every_rounds, list) and update_opp_idx != -1:
                    update_opp_idx += 1
                    if update_opp_idx >= len(update_opp_every_rounds):
                        update_opp_idx = -1
                save_path = os.path.join(self.save_dir, f"model_{i}")
                self.current_agent.save(save_path)
                print(f"Updated model saved at {save_path}")
                env = self.current_agent.env
                cfg = self.current_agent.config
                new_opp = type(self.current_agent)(env, cfg)
                new_opp.load(save_path)
                self.opponents.append({"name": f"model_epoch_{i}", "model": new_opp})
                self.scores.append(0)
            if show_score_rounds and i % show_score_rounds == 0:
                print(f"Score after {i} rounds")
                print(
                    *[f"{o['name']}: {s}" for o, s in zip(self.opponents, self.scores)],
                    sep="\n",
                )
            if (
                discard_randomly
                and len(self.opponents) > max_opponents
                and np.sum(np.array(self.scores) == 0) < 0.3 * len(self.opponents)
            ):
                probs = np.array(self.scores[:-2])
                probs /= probs.sum()
                _idx = np.random.choice(len(self.opponents) - 2, p=probs)
                if "basic" not in self.opponents[_idx]["name"]:
                    opp_ = []
                    scores = []
                    print("Discarding", self.opponents[_idx]["name"])
                    for i, (opp, score) in enumerate(zip(self.opponents, self.scores)):
                        if _idx != i:
                            opp_.append(opp)
                            scores.append(score)
                    self.opponents = opp_
                    self.scores = scores

        print("Ending League...")

        with open(f"{self.save_dir}/final_opp.txt", "w") as f:
            for opp, sc in zip(self.opponents, self.scores):
                f.write(f"{opp['name']} - {sc}\n")


def load_saved_models(sac=False, td7=False, ppo=False, env=None, cfg=[]):
    """
    Load saved models for inference / training other models

    Args:
        sac/td3/ppo : bool, whether to load models
        env: Hockey env
        cfg: List of config files. len(cfg) is at max 3 - 1 each for sac, td3, ppo

    Returns:
        List of models for training against.
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
            yield m, os.path.basename(p)[:-4]

    if sac:
        from utils.parsing import get_default_sac_config
        from sac.sac import SAC_HockeyAgent

        for m, n in list(
            _load(
                env,
                cfg[0] if cfg else get_default_sac_config(),
                "models/sac/*.zip",
                SAC_HockeyAgent,
            )
        ):
            models.append({"name": n, "model": m})

    if td7:
        from utils.load import LoadTD7Agents

        for n, m in LoadTD7Agents(env).items():
            models.append({"name": n, "model": m})

    if ppo:
        from ppo.load_ppo_models import load_all_ppo_agents

        for n, m in load_all_ppo_agents().items():
            models.append({"name": n, "model": m})

    return models


if __name__ == "__main__":
    from utils.parsing import (
        get_config_from_args,
        parse_args,
        print_args,
        print_config,
        save_config,
    )

    args = parse_args()
    cfg = get_config_from_args(args, cfgnode=True)

    if not args.quiet:
        print_config(cfg)
        print_args(args)

    from henv.env import HockeyEnv_SB3

    env = HockeyEnv_SB3(
        False, cfg.environment.additional_rewards, cfg.environment.reward_multiplier
    )

    CURRENT_MAIN = "ppo"
    if CURRENT_MAIN == "sac":
        from sac.sac import SAC_HockeyAgent as Agent
    elif CURRENT_MAIN == "ppo":
        from ppo.ppo import PPO_HockeyAgent as Agent
    # elif CURRENT_MAIN == "td3":
    #     from td3.td3_hockey import TD3_HockeyAgent as Agent

    agent = Agent(env, cfg)
    # Load the agent as well
    agent_copy = Agent(env, cfg)

    league = League(
        agent,
        [agent_copy, BasicOpponent(False)],
        additional_rewards=cfg.environment.additional_rewards,
        reward_multiplier=cfg.environment.reward_multiplier,
    )

    league.train_agent_league(
        10, 2, 5, total_timesteps=1000, log_interval=cfg.training.log_interval
    )
