import torch
import os
from agent_no_prioritised import Agent
from Rainbow.agent import Agent_Rainbow
from Rainbow_RND.agent import Agent_Rainbow_RND
from mario import DQN_Mario
import numpy as np
import random
from distutils.util import strtobool
import argparse
import datetime
import wandb


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--gym-id", type=str, default="SuperMarioBros-1-1-v0",
        help="the id of the gym environment")
    parser.add_argument("--seed", type=int, default=777,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, #reproduce experiments
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--wandb-project-name", type=str, default="RL_Mario",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    #tracking flag
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    #testing flag
    parser.add_argument("--testing", type=lambda x: bool(strtobool(x)), default= False , nargs="?", const=True,
        help="set to true if just want to test the agent playing the game")
    #resume run - if we have a run that hasnt finished with the same experiment,seed and we can resume it
    parser.add_argument("--resume-run", type=lambda x: bool(strtobool(x)), default= True , nargs="?", const=True,
        help="set to true if just want to test the agent playing the game")
    #agent type
    parser.add_argument("--agent-type", type=int, default=0,
        help="tells which DQN agent to use: 0=dueling double, 1=rainbow, 2=rainbow with RND")
    

    parser.add_argument("--num-epochs", type=int, default=200_000,
        help="number of episodes the training goes for")
    # Algorithm specific arguments
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
        help="the learning rate of the optimizer")
    parser.add_argument("--n-epochs", type=int, default=50_000,
        help="total timesteps of the experiments")
    parser.add_argument("--memory-capacity", type=int, default=50_000,
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
    parser.add_argument("--sync-network-rate", type=int, default=100,
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

    os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    #parser args setup and seed
    use_vit = args.use_vit
    environment = DQN_Mario(use_vit=use_vit,seed=args.seed)
    num_actions = environment.action_num
    testing = args.testing
    resume_run = args.resume_run
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
                    use_vit=args.use_vit
                    )
        exp_name = "DQN_experiment"
        
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
                    use_vit=args.use_vit)
        exp_name = "Rainbow_experiment"

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
                    use_vit=args.use_vit)
        exp_name = "Rainbow_RND_experiment"

    else:
        print("error - invalid agent_type")
        exit()
    
    run_name = f"{exp_name}__{args.seed}__{args.gym_id}__{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_id_name = f"{exp_name}__{args.seed}__{args.gym_id}"
    agent.record_video(run_name)

    if(testing):
        agent.test()
        exit()

    if args.track:
        run_id = None
        run_id_file = f"wandb_ids/{run_id_name}.txt"
        #if we ended a run, we can resume it in wandb just by getting the same run_id of that experiment as before. 
        #the models etc will also be loaded so as to resume the run
        if os.path.exists(run_id_file):
            with open(run_id_file,"r") as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip() #get the last run id for this experiment
                    run_id = last_line
        # Initialize or resume the W&B run
        try:
            if run_id and resume_run:
                # Resume the existing run
                run = wandb.init(
                    id=run_id,
                    resume="must",  # Only resume if the run_id exists
                    project=args.wandb_project_name,
                    entity=args.wandb_entity,
                    config=vars(args),
                    name=run_name,
                    monitor_gym=True, # Monitors videos, but for old gym. Doesn't work now
                    save_code=True
                )
                print(f"Resumed existing run with ID: {run_id}")
            else:
                # Start a new run
                run = wandb.init(
                    project=args.wandb_project_name,
                    entity=args.wandb_entity,
                    config=vars(args),
                    name=run_name,
                    monitor_gym=True, # Monitors videos, but for old gym. Doesn't work now
                    save_code=True
                )
                print(f"Started new run with ID: {run.id}")
            # Save the current run_id to the file
            if not os.path.exists("wandb_ids"):
                os.makedirs("wandb_ids")

            # append the run_id to the file if it's not already there
            # if run_id is None -> there wasnt a previous one in this file, so need to append the current one to become first
            if run_id is None or (run.id+"\n") not in lines:
                with open(run_id_file, "a") as f:
                    f.write(f"{run.id}\n")  # Append the run_id as a new line

            run_id = run.id
            wandb.define_metric("game_steps")
            wandb.define_metric("episodes")

            # Define which metrics use which step
            # Define which metrics use which step
            wandb.define_metric("Charts/*", step_metric="game_steps")
            wandb.define_metric("Charts/*", step_metric="episodes")
            wandb.define_metric("losses/*", step_metric="game_steps")
            wandb.define_metric("losses/*", step_metric="episodes")
            wandb.define_metric("losses_avg/*", step_metric="game_steps")
            wandb.define_metric("losses_avg/*", step_metric="episodes")
            wandb.define_metric("losses_total/*", step_metric="game_steps")
            wandb.define_metric("losses_total/*", step_metric="episodes")

        except wandb.Error as e:
            print(f"Failed to initialize/resume W&B run: {e}")
            exit()

    agent.train(args.num_epochs) #pass the Mario environment for agent to train on

    if args.track:
        wandb.finish()
        #remove the run_id from the file as its done now
        if os.path.exists(run_id_file):
            with open(run_id_file,"r") as f:
                lines = f.readlines()

            #filter out the run_id lines
            lines = [line for line in lines if line.strip() != f"{run_id}\n".strip()]

            # Write the remaining lines back to the file
            with open(run_id_file, "w") as f:
                f.writelines(lines)

            print(f"Removed run_id: {run_id.strip()}")
