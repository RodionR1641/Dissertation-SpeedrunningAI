import logging
import os
import datetime
import torch

# Create log directory
log_dir = "/cs/home/psyrr4/Code/Code/logs"
os.makedirs(log_dir, exist_ok=True)

# Define log file name (per process)
rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
log_file = os.path.join(log_dir, f"experiment_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_rank{rank}.log")

# Configure logging
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Log startup details
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

logging.info(f"device is {device}")


# Example training loop logging
for i in range(1, 10):
    logging.info(f"Hello World {i}")

logging.info("Training completed!")
