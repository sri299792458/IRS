#!/bin/bash
#SBATCH --job-name=qwen_grasp_vlg_train
#SBATCH --time=05:00:00          # Resuming for 4 hours
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16       # 16 CPU cores for 8 dataloader workers
#SBATCH --mem=64G                # 64GB RAM
#SBATCH --partition=msigpu       # Your partition
#SBATCH --gres=gpu:a100:1        # Requesting 1 A100 GPU
#SBATCH --account=csci5561       # Your account
#SBATCH -o logs/slurm-vlg-%j.out
#SBATCH -e logs/slurm-vlg-%j.err

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

# 3. FAST I/O SETUP (Critical Speed Boost)
# Copy dataset from local folder (OCID-VLG) in IRS directory to local NVMe ($TMPDIR)
echo "Setting up local data in: $TMPDIR/OCID-VLG"
mkdir -p $TMPDIR/OCID-VLG

echo "Copying OCID-VLG data..."
# Copy from the folder where code also resides
cp -r OCID-VLG/* $TMPDIR/OCID-VLG/

echo "Data copy complete. Storage size:"
du -sh $TMPDIR/OCID-VLG

# 4. Run Training
# Note: Pointing to the fast local copy
# Run the VLG training script
python -u OCID-VLG/train_grasp_vlg.py \
    --data_dir $TMPDIR/OCID-VLG \
    --checkpoint_dir /scratch.global/kanth042/IRS_project/checkpoints/qwen3-grasp-vlg-lora \
    --resume_from_checkpoint /scratch.global/kanth042/IRS_project/checkpoints/qwen3-grasp-vlg-lora/checkpoint-9000 \
    --run_name grasp-vlg-continue

echo "--------------------"
echo "Job finished at $(date)"
