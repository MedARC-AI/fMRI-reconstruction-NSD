# Distributed Data Parallel initialization
# Setup multi-gpu PyTorch distributed parallel processing (ddp) if multiple gpus are found
import os
import torch
def ddp_test():
    try:
        # set up multi-gpu distributed parallel processing
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        from socket import gethostname

        rank          = int(os.environ["SLURM_PROCID"])
        world_size    = int(os.environ["WORLD_SIZE"])
        gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
        assert gpus_per_node == torch.cuda.device_count()
        print(f"Hello from rank {rank} of {world_size} on {gethostname()} where there are" \
              f" {gpus_per_node} allocated GPUs per node.", flush=True)

        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)

        local_rank = rank - gpus_per_node * (rank // gpus_per_node)
        print("local rank",local_rank)
        torch.cuda.set_device(local_rank)
        using_ddp = True
    except:
        local_rank = 0
        using_ddp = False
    if using_ddp:
        print("DistributedDataParallel enabled!")
    else:
        print("NOT using distributed parallel processing!")
    return using_ddp, local_rank

def set_ddp():
    import torch.distributed as dist
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK",
                    "LOCAL_RANK", "WORLD_SIZE")
    }
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    n = torch.cuda.device_count()
    device_ids = list(
        range(local_rank * n, (local_rank + 1) * n)
    )

    if local_rank == 0:
        print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend="nccl")
    print(
        f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
    )

    print(
        f"[{os.getpid()}] rank = {dist.get_rank()} ({rank}), "
        + f"world_size = {dist.get_world_size()}, n = {n}, device_ids = {device_ids}"
    )
    device = torch.device("cuda", local_rank)
    return True, local_rank, device

# Slurm reference for DDP:
## !/bin/bash
# SBATCH --job-name=clipvox      #create a short name for your job
# SBATCH --nodes=1               #node count
# SBATCH --ntasks-per-node=2    #with DDP, must equal num of gpus
# SBATCH --cpus-per-task=8     #rule-of-thumb is 4 times number of gpus
# SBATCH --gres=gpu:2  
# SBATCH --mem-per-gpu=40G     
# SBATCH --time=00:10:00       #total run time limit (HH:MM:SS)
# SBATCH --mail-type=begin,end,fail         
# SBATCH --mail-user=your_email@gmail.com
# export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
# export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
# echo "WORLD_SIZE="$WORLD_SIZE
# master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_ADDR=$master_addr
# echo "MASTER_ADDR="$MASTER_ADDR
# srun python Voxel_to_CLIPvoxel.py
