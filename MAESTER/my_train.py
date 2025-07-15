import argparse
import os
import random

import dataset as _  # otherwise dataset is not registered as plugin
import numpy
import torch.distributed as dist
import torch.utils.data.distributed
from model import *
from pretrain_engine import set_deterministic
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from utils import get_plugin, read_yaml, save_checkpoint

parser = argparse.ArgumentParser(description="MAESTER Training")
parser.add_argument("--model_config_dir", default="./config", type=str)
parser.add_argument("--model_config_name", default="deafult.yaml", type=str)
# parser.add_argument("--dist_backend", default="nccl", type=str, help="")
# parser.add_argument("--world_size", default=2, type=int, help="")
# parser.add_argument("--init_method", default="tcp://127.0.0.1:56079", type=str, help="")
parser.add_argument("--logdir", default="./checkpoints", type=str, help="log directory")
parser.add_argument("--log_dir", type=str, help="log directory")

print("Starting...", flush=True)
args = parser.parse_args()
set_deterministic()
cfg = read_yaml(os.path.join(args.model_config_dir, args.model_config_name))
print("model_config:", cfg, flush=True)

# ngpus_per_node = torch.cuda.device_count()
# current_device = local_rank = int(os.environ.get("SLURM_LOCALID"))
# rank = int(os.environ.get("SLURM_NODEID")) * ngpus_per_node + local_rank
# torch.cuda.set_device(rank)

# dist.init_process_group(
#     backend=args.dist_backend,
#     init_method=args.init_method,
#     world_size=args.world_size,
#     rank=rank,
# )
# print(f"From Rank: {rank}, ==> Process group ready!", flush=True)
# print(f"From Rank: {rank}, ==> Building model..")
print(f"From Rank: {0}, ==> Building model..")
model = get_plugin("model", cfg["MODEL"]["name"])(cfg["MODEL"]).cuda()
# model = get_plugin("model", cfg["MODEL"]["name"])(cfg["MODEL"])

# model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[current_device])
model.embed_dim = cfg["MODEL"]["embed_dim"]
# print(f"From Rank: {rank}, ==> Model ready!", flush=True)
print(f"From Rank: {0}, ==> Model ready!", flush=True)
# print(f"From Rank: {rank}, ==> Preparing data..")
print(f"From Rank: {0}, ==> Preparing data..")

dataset = get_plugin("dataset", cfg["DATASET"]["name"])(cfg["DATASET"])


# determinstic behaviour
def seed_worker(worker_id):
    set_deterministic()


g = torch.Generator()
g.manual_seed(0)

# sampler = DistributedSampler(dataset)
dataloader = DataLoader(
    dataset,
    batch_size=cfg["DATASET"]["batch_size"],
    # sampler=sampler,
    num_workers=cfg["DATASET"]["num_workers"],
    pin_memory=True,
    worker_init_fn=seed_worker,
    generator=g,
)

optimizer = get_plugin("optim", cfg["OPTIM"]["name"])(model, cfg["OPTIM"])

# print(f"From Rank: {rank}, ==> Data ready!")
print(f"From Rank: {0}, ==> Data ready!")
engine_func = get_plugin("engine", cfg["ENGINE"]["name"])

# logger
logger = SummaryWriter(log_dir=args.log_dir)

for epoch in range(cfg["ENGINE"]["epoch"]):
    # epoch_loss = engine_func(model, dataloader, optimizer, rank, epoch, cfg)
    epoch_loss = engine_func(model, dataloader, optimizer, cfg, epoch, logger)
    # if rank == 0 and (epoch + 1) % 50 == 0:
    if (epoch + 1) % 50 == 0:
        state_dict = model.module.state_dict()
        save_checkpoint(args.logdir, state_dict, name=f"epoch_{epoch:04}.pt")
    print(f"Epoch [{epoch}/{cfg['ENGINE']['epoch']-1}], Loss: {epoch_loss:.4f}")

# print(f"From Rank: {rank}, ==> Training finished!")
print(f"From Rank: {0}, ==> Training finished!")
