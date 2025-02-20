import logging
import os
import datetime
import torch

# Create log directory
log_dir = "/cs/home/psyrr4/Code/Code/Mario/logs"
if not os.path.exists(log_dir):
	os.makedirs(log_dir)

f = open("test_1.txt", "w")
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

#logging.info(f"device is {device}")

print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("Number of GPUs:", torch.cuda.device_count())

# Example training loop logging
for i in range(1, 10):
    logging.info(f"Hello World {i}")

logging.info("Training completed!")
