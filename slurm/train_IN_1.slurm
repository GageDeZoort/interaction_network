#!/bin/bash
#SBATCH -o ./job_reports/output.LP_Train_1_400_noPhi.out # STDOUT
#SBATCH --job-name=LP_Train_1_noPhi  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=128G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email on job start, end and fail
#SBATCH --mail-user=jdezoort@princeton.edu

module purge
module load anaconda3
conda activate gnnenv

cd /home/jdezoort/GNN_Repo_Reorganization/gnn_track_challenge/interaction_network
python train_IN.py configs/train_IN_1.yaml
python plot_IN.py configs/train_IN_1.yaml
