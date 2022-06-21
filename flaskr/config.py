import torch

MINIBAT_SIZE = 32 # 128 for cartpole
N_WORKERS = 3
REPLAYMEM_WORKER_SIZE = 1000
REPLAYMEM_SIZE = N_WORKERS * REPLAYMEM_WORKER_SIZE # 5000 for cartpole
LEARN_RATE_A = 0.001
LEARN_RATE_C = 0.001
TH_SCORE = 210.

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
print(device)
