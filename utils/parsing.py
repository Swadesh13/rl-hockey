import argparse

import yaml
from fvcore.common.config import CfgNode


def parse_args():
    parser = argparse.ArgumentParser(description="Run RL training script")

    parser.add_argument("--quiet", "-q", action="store_true", help="Run the program in quiet mode")

    # Config file argument
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config file",
    )

    # Train and eval options
    parser.add_argument("--train", action="store_true", help="Run training script")
    parser.add_argument("--eval", action="store_true", help="Run evaluation script")
    parser.add_argument(
        "--eval_episodes",
        "-ee",
        type=int,
        default=10,
        help="Number of episodes to run during evaluation",
    )

    # Override arguments
    parser.add_argument("--env_name", type=str, default=None, help="Override the environment name")
    parser.add_argument("--n_envs", type=int, default=None, help="Override number of environments")
    parser.add_argument("--seed", type=int, default=None, help="Override the seed")
    parser.add_argument(
        "--total_timesteps",
        "-tt",
        type=int,
        default=None,
        help="Override total timesteps",
    )
    parser.add_argument(
        "--save_model_every",
        type=int,
        default=None,
        help="Override save model frequency",
    )
    parser.add_argument("--model_path", type=str, default=None, help="Override model save path")
    parser.add_argument("--log_dir", type=str, default=None, help="Override logging directory")
    parser.add_argument("--verbose", type=int, default=None, help="Override verbosity level")

    # Hyperparameters overrides
    parser.add_argument("--learning_rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--n_steps", type=int, default=None, help="Override number of steps")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--n_epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--gamma", type=float, default=None, help="Override gamma")
    parser.add_argument("--gae_lambda", type=float, default=None, help="Override GAE lambda")
    parser.add_argument("--clip_range", type=float, default=None, help="Override clip range")
    parser.add_argument(
        "--vf_coef",
        type=float,
        default=None,
        help="Override value function coefficient",
    )
    parser.add_argument("--ent_coef", type=float, default=None, help="Override entropy coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=None, help="Override max gradient norm")

    return parser.parse_args()


def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config_path, config_data):
    "Save YAML configuration file"
    with open(config_path, "w", encoding="utf-8") as yaml_file:
        yaml.dump(config_data, yaml_file, sort_keys=False, allow_unicode=True)


def override_args(config, args):
    """Override configuration values with command-line arguments."""
    # Environment overrides
    if args.env_name is not None:
        config["environment"]["env_name"] = args.env_name
    if args.seed is not None:
        config["environment"]["seed"] = args.seed
    if args.n_envs is not None:
        config["environment"]["n_envs"] = args.n_envs

    # Training overrides
    if args.total_timesteps is not None:
        config["training"]["total_timesteps"] = args.total_timesteps
    if args.save_model_every is not None:
        config["training"]["save_model_every"] = args.save_model_every
    if args.model_path is not None:
        config["training"]["model_path"] = args.model_path

    # Logging overrides
    if args.log_dir is not None:
        config["logging"]["log_dir"] = args.log_dir
    if args.verbose is not None:
        config["logging"]["verbose"] = args.verbose

    # Hyperparameters overrides
    hyperparams = config["model"]["hyperparameters"]
    if args.learning_rate is not None:
        hyperparams["learning_rate"] = args.learning_rate
    if args.n_steps is not None:
        hyperparams["n_steps"] = args.n_steps
    if args.batch_size is not None:
        hyperparams["batch_size"] = args.batch_size
    if args.n_epochs is not None:
        hyperparams["n_epochs"] = args.n_epochs
    if args.gamma is not None:
        hyperparams["gamma"] = args.gamma
    if args.gae_lambda is not None:
        hyperparams["gae_lambda"] = args.gae_lambda
    if args.clip_range is not None:
        hyperparams["clip_range"] = args.clip_range
    if args.vf_coef is not None:
        hyperparams["vf_coef"] = args.vf_coef
    if args.ent_coef is not None:
        hyperparams["ent_coef"] = args.ent_coef
    if args.max_grad_norm is not None:
        hyperparams["max_grad_norm"] = args.max_grad_norm

    return config


def convert_to_cfgnode(config):
    """Convert a dictionary-based config to a CfgNode for dot notation access."""
    return CfgNode(config)


def get_config_from_args(args, cfgnode=True):
    """
    Parses command-line arguments, loads a configuration file, and optionally converts it to a configuration node.
    """
    config = load_config(args.config)
    config = override_args(config, args)
    if cfgnode:
        config = convert_to_cfgnode(config)
    return config


def get_default_ppo_config(cfgnode=True):
    print("Loading default PPO config 'configs/ppo_hockey.yaml'")
    config = load_config("configs/ppo_hockey.yaml")
    if cfgnode:
        config = convert_to_cfgnode(config)
    return config


def get_default_td3_config(cfgnode=True):
    print("Loading default TD3 config 'configs/td3_hockey.yaml'")
    config = load_config("configs/td3_hockey.yaml")
    if cfgnode:
        config = convert_to_cfgnode(config)
    return config


def get_default_sac_config(cfgnode=True):
    print("Loading default SAC config 'configs/td3_hockey.yaml'")
    config = load_config("configs/sac_hockey.yaml")
    if cfgnode:
        config = convert_to_cfgnode(config)
    return config


def print_config(cfg):
    """Print the configuration with a line of dashes above and below."""
    print("-" * 11 + " CONFIG " + "-" * 11)
    print(cfg)
    print("-" * 30)


def print_args(args):
    print("-" * 12 + " ARGS " + "-" * 12)
    for arg, value in vars(args).items():
        if value is not None:
            print(f"{arg}: {value}")
    print("-" * 30)
