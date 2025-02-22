from utils.parsing import parse_args, get_config_from_args, print_config, print_args
from .initilizer import CreateHockyEnv,CreateHockyEnvVector,CreateHockyEnvSelfVector,CreateHockyEnvAllVector,CreateHockyAgentFromTeam,CreateHockyEnvFromOpponent
from .model import TD7
from .agent import TD7HockyAgent
import multiprocessing
from henv.env import BasicOpponent


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    args = parse_args()
    config = get_config_from_args(args, cfgnode=True)

    if not args.quiet:
        print_config(config)
        print_args(args)

    trainEnv = None
    if args.train:
        if config.train_env.mode == "normal":
            trainEnv = CreateHockyEnvVector(config.train_env)
        elif config.train_env.mode == "self":
            trainEnv = CreateHockyEnvSelfVector()
        elif config.train_env.mode == "all":
            trainEnv = CreateHockyEnvAllVector()
        else:
            raise ValueError("Invalid mode for train_env")
    
    evalEnv = CreateHockyEnv(seed=config.eval_env.seed, addtionalRewards=config.eval_env.additional_rewards,
                             weakOpponent=config.eval_env.weak_opponent)

    opponent = None
    if config.evaluation.mode == "normal":
        opponent = BasicOpponent(weak=config.evaluation.weak)
    elif config.evaluation.mode == "team":
        opponent = CreateHockyAgentFromTeam(algo=config.evaluation.algo, name=config.evaluation.type)
    else:
        raise ValueError("Invalid mode for evaluation")

    if config.model.load:
        agent = TD7HockyAgent(
            config = config,
            model = None,
            trainEnv = trainEnv,
            evalEnv = evalEnv,
            loadModel = True,
            modelsDir = config.model.models_dir,
            modelName = config.model.model_load_name
        )
    else:
        model = TD7(
                config = config.model,
                actionSpace = evalEnv.action_space,
                obsSpace = evalEnv.observation_space
        )
        agent = TD7HockyAgent(
            config = config,
            model = model,
            evalEnv = evalEnv,
            trainEnv= trainEnv
        )

    if args.train:
        agent.train()
        agent.evaluate(opponent_right=opponent)
        agent.save()

    if args.eval:
        agent.load()
        agent.evaluate(opponent_right=opponent)