from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
import os
import logging
import datetime
import torch
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

print("Got here")
log_dir = "/cs/home/psyrr4/Code/Code/Mario/logs"
if not os.path.exists(log_dir):
	os.makedirs(log_dir)

rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
log_file = os.path.join(log_dir, f"experiment_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_rank{rank}.log")

# Configure logging
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

print("starting logging")
logging.info(f"Process {rank} started training on GPUs")
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    try:
        print(torch.cuda.current_device())
	#print(torch.cuda.is_available())
    except RuntimeError as e:
        print(f"{e}")
else:
	print("cuda not available")

print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("Number of GPUs:", torch.cuda.device_count())

done = True
for step in range(5000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    logging.info(f"Hello World {step}")
    #env.render()
print("done")

env.close()
