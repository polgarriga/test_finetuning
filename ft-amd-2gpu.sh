#!/usr/bin/env bash
#SBATCH --job-name=ft_m2m
#SBATCH --output=slurm_logs/m2m_ft_ca_de_1node_amd_%j.out
#SBATCH --error=slurm_logs/m2m_ft_ca_de_1node_amd_%j.err
#SBATCH --nodes=1
#SBATCH --gres gpu:2
#SBATCH --cpus-per-task=128
#SBATCH --time=0-02:00:00
#SBATCH --qos=debug

load_config_amd_0102() {
	module load gcc/10.2.0 rocm/4.0.1 intel/2018.4 mkl/2018.4 python/3.7.4

	export LD_LIBRARY_PATH=/gpfs/projects/bsc88/projects/bne/eval_amd/scripts_to_run/external-lib:$LD_LIBRARY_PATH

	source /gpfs/projects/bsc88/projects/bne/eval_amd/ksenia/venv-fairseq/bin/activate

	echo $SLURM_JOB_NODELIST
}

load_config_amd_0102

python -m torch.distributed.launch --nproc_per_node=2 finetuning_amd.py



