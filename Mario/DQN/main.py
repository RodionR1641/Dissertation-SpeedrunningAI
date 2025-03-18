import torch
import os
import keyboard
#from Rainbow.agent import Agent
from agent_no_prioritised import Agent
from Rainbow.agent import Agent_Rainbow
from Rainbow_RND.agent import Agent_Rainbow_RND
from mario import DQN_Mario
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from distutils.util import strtobool
import argparse
import datetime
import logging
import time


def print_info():
    print("starting logging")
    logging.info(f"Process {rank} started training on GPUs")

    if torch.cuda.is_available():
        try:
            logging.info(torch.cuda.current_device())
            logging.info("GPU Name: " + torch.cuda.get_device_name(0))
            logging.info("PyTorch Version: " + torch.__version__)
            logging.info("CUDA Available: " + str(torch.cuda.is_available()))
            logging.info("CUDA Version: " + str(torch.version.cuda))
            logging.info("Number of GPUs: " + str(torch.cuda.device_count()))
        except RuntimeError as e:
            logging.info(f"{e}")
    else:
        logging.info("cuda not available")


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="SuperMarioBros-1-1-v0",
        help="the id of the gym environment")
    parser.add_argument("--seed", type=int, default=777,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, #reproduce experiments
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="DQN-experiments",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--testing", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="tells whether training or testing agent")
    
    parser.add_argument("--agent-type", type=int, default=2,
        help="tells which DQN agent to use: 0=dueling double, 1=rainbow, 2=rainbow with RND")
    
    parser.add_argument("--num-epochs", type=int, default=200_000,
        help="number of episodes the training goes for")
    
    # Algorithm specific arguments
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
        help="the learning rate of the optimizer")
    parser.add_argument("--n-epochs", type=int, default=100_000,
        help="total timesteps of the experiments")
    parser.add_argument("--memory-capacity", type=int, default=100_000,
        help="number of experiences stored in replay buffer"),
    parser.add_argument("--memory-capacity-rainbow", type=int, default=50_000,
        help="number of experiences stored in replay buffer"),
    parser.add_argument("--batch-size", type=int, default=32,
        help="number of experiences sampled each time for training"),
    parser.add_argument("--epsilon", type=float, default=1.0,
        help="the starting epsilon value"),
    parser.add_argument("--min-epsilon", type=float, default=0.1,
        help="the minimum epsilon value"),
    parser.add_argument("--nb-warmup", type=int, default=250_000,
        help="number of timesteps where epsilon decreases from starting to end value"),
    parser.add_argument("--sync-network-rate", type=int, default=1000,
        help="how often(timesteps) the target network copies the online network"),
    parser.add_argument("--use-vit", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="decide if using vit or not for model"),
    #rainbow and rnd specific
    parser.add_argument("--v-min", type=int, default=0,
        help="min value of support for categorical DQN"),
    parser.add_argument("--v-max", type=int, default=200,
        help="max value of support for categorical DQN"),
    parser.add_argument("--atom-size", type=int, default=51,
        help="the unit number of support"),
    parser.add_argument("--alpha", type=float, default=0.2,
        help="controls how important prioritised sampling is"),
    parser.add_argument("--beta", type=float, default=0.6,
        help="controls the weight "),
    parser.add_argument("--prior-eps", type=float, default=1e-6,
        help="small value to make sure every experience has chance of being selected"),
    parser.add_argument("--n-step", type=int, default=3,
        help="step number to calculate n-step td error")


    args = parser.parse_args()
    return args

def seed_run():
    torch.manual_seed(args.seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(args.seed)
        #torch.backends.cudnn.benchmark = False # could be useful for reproducibility, but setting to False affects performance
        torch.backends.cudnn.deterministic = args.torch_deterministic
    
    np.random.seed(args.seed)
    random.seed(args.seed)


if __name__ == "__main__":

    args = parse_args()
    print(os.getcwd())


    log_dir = "/cs/home/psyrr4/Code/Code/Mario/logs"
    #os.makedirs(log_dir, exist_ok=True)

    # Define log file name (per process)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    log_file = os.path.join(log_dir, f"experiment_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_rank{rank}.log")

    # Configure logging
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    print_info()
    os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    #parser args setup and seed
    use_vit = args.use_vit
    environment = DQN_Mario(use_vit=use_vit,seed=args.seed)
    num_actions = environment.action_num
    testing = args.testing
    seed_run()
    agent_type = args.agent_type

    #choose which agent type to train

    if agent_type == 0:
        agent = Agent(input_dims=environment.observation_space.shape,
                    env=environment,
                    device=device,
                    nb_actions=num_actions,
                    memory_capacity=args.memory_capacity,
                    batch_size=args.batch_size,
                    epsilon=args.epsilon,
                    min_epsilon=args.min_epsilon,
                    nb_warmup=args.nb_warmup,
                    learning_rate=args.learning_rate,
                    gamma=args.gamma,
                    sync_network_rate=args.sync_network_rate,
                    use_vit=args.use_vit)
        
    elif agent_type == 1:
        agent = Agent_Rainbow(input_dims=environment.observation_space.shape,
                    env=environment,
                    device=device,
                    nb_actions=num_actions,
                    memory_capacity=args.memory_capacity_rainbow,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    gamma=args.gamma,
                    sync_network_rate=args.sync_network_rate,
                    v_min=args.v_min,
                    v_max=args.v_max,
                    atom_size=args.atom_size,
                    alpha=args.alpha,
                    beta=args.beta,
                    prior_eps=args.prior_eps,
                    n_step=args.n_step,
                    use_vit=args.use_vit

        )
    elif agent_type == 2:
        agent = Agent_Rainbow_RND(input_dims=environment.observation_space.shape,
                    env=environment,
                    device=device,
                    nb_actions=num_actions,
                    memory_capacity=args.memory_capacity_rainbow,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    gamma=args.gamma,
                    sync_network_rate=args.sync_network_rate,
                    v_min=args.v_min,
                    v_max=args.v_max,
                    atom_size=args.atom_size,
                    alpha=args.alpha,
                    beta=args.beta,
                    prior_eps=args.prior_eps,
                    n_step=args.n_step,
                    use_vit=args.use_vit
        )
    else:
        print("error - invalid agent_type")
        exit()

    if(testing):
        agent.test()
    else:
        agent.train(args.num_epochs) #pass the Mario environment for agent to train on