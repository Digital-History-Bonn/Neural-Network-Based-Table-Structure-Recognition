#!/usr/bin/zsh
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=4
#SBATCH --job-name=train
#SBATCH --output=outputs/tabletransformer-titw-%j.out
#SBATCH --error=outputs/tabletransformer-titw%j.err
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --account=rwth1651

source /home/gh577232/.zshrc
module load CUDA
conda activate kosmos2

epoch=$1
model=$2

echo "$epoch"
echo "$model"

srun python -m src.historicdocumentprocessing.tabletransformer_train --name "titw_call_e250" --dataset "Tablesinthewild" --valid --no-early_stopping --gpus 4 --num_nodes 3 --epochs "$epoch" --load "$model" --identicalname
