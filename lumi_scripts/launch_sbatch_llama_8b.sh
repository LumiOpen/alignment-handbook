#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --account=project_462000963
#SBATCH --partition=standard-g
#SBATCH --cpus-per-task=56
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=480G
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out # Name of stdout output file
#SBATCH --error=logs/%j.err  # Name of stderr error file

set -euxo pipefail

echo "JOB NAME" $SLURM_JOB_NAME

VENV="/scratch/project_462000353/adamhrin/alignment-handbook/.venv"

module use /appl/local/csc/modulefiles/
module load pytorch/2.7
source $VENV/bin/activate

export HF_HOME="/scratch/project_462000963/users/adamhrin/hf_cache"
export HF_DATASETS_CACHE="/scratch/project_462000963/users/adamhrin/datasets_cache"
export PYTHONPATH="$VENV/lib/python3.11/site-packages"
export SSL_CERT_FILE=$(python -m certifi)

#Distributed variables
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export LOCAL_RANK=$SLURM_LOCALID
#export RANK=$SLURM_PROCID
export WORLD_SIZE=$((SLURM_GPUS_ON_NODE*SLURM_NNODES))


#LOGGING/DEBUGGING
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
#HF_HUB_ENABLE_HF_TRANSFER=1 #Speeds up loading from hf hub, i think
# export TORCH_NCCL_BLOCKING_WAIT=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
#HF_HUB_ENABLE_HF_TRANSFER=1 #Speeds up loading from hf hub, i think
#export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 #This might not work with rccl
#export HSA_FORCE_FINE_GRAIN_PCIE=1 #Supposedly improves performance/prevents hanging
#export HIP_LAUNCH_BLOCKING=1 #Removes async operations
#export TRANSFORMERS_VERBOSITY=error
#export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export NCCL_DEBUG=INFO
export ACCELERATE_LOG_LEVEL=INFO
export OMP_NUM_THREADS=1 #This could be increased
export TOKENIZERS_PARALLELISM=false #Removes error involved with the FastTokenizer and rust/python parallelism.

ACCELERATE_CONFIG_FILE=recipes/accelerate_configs/zero3.yaml
CONFIG_FILE=${1:-"recipes/llama-3/sft/config_8b_poro2.yaml"}

echo "JOBNAME" $SLURM_JOB_NAME
echo "ACCELERATE_CONFIG_FILE" $ACCELERATE_CONFIG_FILE
echo "CONFIG_FILE" $CONFIG_FILE

pwd -P

export CMD=" \
    scripts/sft.py --config $CONFIG_FILE 
    "


#LAUNCHERS
export ACC_LAUNCHER="singularity_wrapper exec accelerate launch \
    --rdzv_conf "rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT" \
    --config_file $ACCELERATE_CONFIG_FILE \
    --num_machines $SLURM_NNODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --role \$(hostname -s|tr -dc '0-9'): \
    --tee 3 \
    "


SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$ACC_LAUNCHER --role \$SLURMD_NODENAME: $CMD"

echo "END TIME: $(date)"

echo "END $SLURM_JOBID: $(date)"
