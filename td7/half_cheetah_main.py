from utils.parsing import parse_args, get_config_from_args, print_config, print_args
import gymnasium as gym
from .initilizer import CreateGymEnv,CreatePendulumV1EnvVector
from .model import TD7
from .agent import TD7GymAgent


if __name__ == "__main__":
    args = parse_args()
    config = get_config_from_args(args, cfgnode=True)

    if not args.quiet:
        print_config(config)
        print_args(args)

    evalEnv = CreateGymEnv(envName=config.train_env.env_name)
    trainEnv = CreatePendulumV1EnvVector(envName=config.train_env.env_name ,envNum=config.train_env.n_envs)

    model = TD7(
        config=config.model,
        actionSpace=evalEnv.action_space,
        obsSpace= evalEnv.observation_space
    )
    agent = TD7GymAgent(
        config=config,
        model=model,
        evalEnv=evalEnv,
        trainEnv=trainEnv
    )

    if args.train:
        agent.train()
        agent.evaluate()
        agent.save()

    if args.eval:
        agent.load()
        agent.evaluate()