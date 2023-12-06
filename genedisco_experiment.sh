#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name="pn"
#SBATCH --output=./slurm_stdout/slurm-pn-%j.out
#SBATCH --error=./slurm_stdout/slurm-pn-%j.err
#SBATCH --array=0-19
#SBATCH --nodelist=oat17

export CONDA_ENVS_PATH=/scratch-ssd/pastin/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/pastin/conda_pkgs
#/scratch-ssd/oatml/run_locked.sh /scratch-ssd/oatml/miniconda3/bin/conda-env update -f discobax.yml
source /scratch-ssd/oatml/miniconda3/bin/activate discobax

export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUBLAS_WORKSPACE_CONFIG=:4096:8

#Data location
export cache_directory=/scratch/pastin/discobax/data
export output_directory=/scratch/pastin/discobax/output
export performance_file_location="./output/GeneDisco_performance.csv"

#Setup - These parameters are fixed across all Genedisco experiments
export model_name="bayesian_mlp"
export acquisition_batch_size=32 #Size of acquisition batch at each cycle
export num_active_learning_cycles=25 #Number of total acquisition cycles
export num_topk_clusters=20 #Number of distinct clusters amongst the overall top interventions
export topk_percent=0.01 #Threshold to define top interventions (default = top percentile)
export seed=(1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000)
export dataset_name="sanchez_2021_tau" #"schmidt_2021_ifng" "schmidt_2021_il2" "zhuang_2019_nk" "sanchez_2021_tau" "zhu_2021_sarscov2"

#Acquisition functions (DiscoBAX and other baselines)
export acquisition_function="discobax" #"random" "topk_bax" "levelset_bax" "discobax" "jepig" "ucb" "thompson_sampling" "topuncertain" "softuncertain" "marginsample" "badge" "coreset" "kmeans_embedding" "kmeans_data" "adversarialBIM"

srun \
    python3 discobax/apps/genedisco_experiment.py  \
            --cache_directory=${cache_directory} \
            --output_directory=${output_directory} \
            --model_name=${model_name} \
            --acquisition_function_name=${acquisition_function} \
            --acquisition_batch_size=${acquisition_batch_size} \
            --num_active_learning_cycles=${num_active_learning_cycles} \
            --dataset_name=${dataset_name} \
            --seed=${seed[$SLURM_ARRAY_TASK_ID]} \
            --performance_file_location=${performance_file_location} \
            --num_topk_clusters=${num_topk_clusters} \
            --topk_percent=${topk_percent}