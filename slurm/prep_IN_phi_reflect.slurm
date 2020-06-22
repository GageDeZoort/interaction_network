#!/bin/bash
#SBATCH -o ./job_reports/output.prep_LP.out  # STDOUT
#SBATCH --job-name=LP_prep_phi_reflect       # create a short name for your job
#SBATCH --nodes=1                            # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=06:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email on job start, end and fail
#SBATCH --mail-user=jdezoort@princeton.edu

module purge
module load anaconda3
conda activate gnnenv

cd /home/jdezoort/GNN_Repo_Reorganization/gnn_track_challenge/interaction_network
python prep_LP.py configs/prep_LP_phi_reflect.yaml
