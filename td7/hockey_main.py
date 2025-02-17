from utils.parsing import parse_args, get_config_from_args, print_config, print_args
from .initilizer import CreateHoeckyEnv,CreateHoeckyEnvs
from .model import TD7
from .agent import TD7Agent

if __name__ == "__main__":
    args = parse_args()
    config = get_config_from_args(args, cfgnode=True)

    if not args.quiet:
        print_config(config)
        print_args(args)

    evalEnv = CreateHoeckyEnv(config.eval_env)
    trainEnv = CreateHoeckyEnvs(config.train_env)

    model = TD7(
        config = config.model,
        actionSpace = trainEnv.single_action_space,
        obsSpace = trainEnv.single_observation_space
    )

    agent = TD7Agent(
        config = config,
        model = model,
        trainEnv = trainEnv,
        evalEnv = evalEnv
    )

    if args.train:
        agent.train()
        agent.evaluate()
        agent.save()

    if args.eval:
        agent.load(config.evaluation.model_date)
        agent.evaluate()