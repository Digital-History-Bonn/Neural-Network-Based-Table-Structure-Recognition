#!/usr/bin/zsh
#SBATCH --job-name=train
#SBATCH --output=outputs/tabletransformer-titw-%j.out
#SBATCH --error=outputs/tabletransformer-titw-%j.err
#SBATCH --time=00:45:00
#SBATCH --account=rwth1651

source /home/gh577232/.zshrc
module load CUDA
conda activate kosmos2

nodes=$1
gpus=$2

srun python -m src.historicdocumentprocessing.tabletransformer_train --name "titw_gputest" --dataset "Tablesinthewild" --valid --no-early_stopping --gpus "$gpus" --num_nodes "$nodes"
