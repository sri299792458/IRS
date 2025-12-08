#!/bin/bash
#SBATCH --job-name=vlg_novel_train
#SBATCH --time=14:00:00          # 6 Epochs (~11 hrs)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16       # 16 CPU cores
#SBATCH --mem=64G                # 64GB RAM
#SBATCH --partition=msigpu       # Your partition
#SBATCH --gres=gpu:a100:1        # Requesting 1 A100 GPU
#SBATCH --account=stat8105       # Your account
#SBATCH -o logs/slurm-novel-%j.out
#SBATCH -e logs/slurm-novel-%j.err

# 1. Setup Logging & Environment
mkdir -p logs
module load cuda/12.1.1            
module load conda
source activate IRS              

# 2. Debug Info
echo "Job started on $(hostname) at $(date)"
echo "--------------------"
nvidia-smi
echo "--------------------"

# 3. FAST I/O SETUP
# Copy dataset from local folder to local NVMe ($TMPDIR)
echo "Setting up local data in: $TMPDIR/OCID-VLG"
mkdir -p $TMPDIR/OCID-VLG

echo "Copying OCID-VLG data..."
cp -r OCID-VLG/* $TMPDIR/OCID-VLG/

echo "Data copy complete. Storage size:"
du -sh $TMPDIR/OCID-VLG

# 4. Run Training (Novel Classes Split)
echo "Starting Novel Classes Training..."

# Note: We use experiment_novel_classes.py train
python -u OCID-VLG/experiment_novel_classes.py train \
    --data_dir $TMPDIR/OCID-VLG \
    --checkpoint_dir /scratch.global/kanth042/IRS_project/checkpoints/qwen3-grasp-novel-classes

echo "--------------------"
echo "Job finished at $(date)"
