#!/bin/sh

#SBATCH --job-name="test"
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --partition=jakku

ulimit -s unlimited
ulimit -c unlimited

export OMP_NUM_THREADS=8

CONT=${CONT:-"/cstor/dmoise/containers/pytorch-25.02-py3.sif"}
DATADIR=${DATADIR:-"/cstor/dmoise/climate-llm/Prithvi-WxC/data"}
PYTHONDIR=${PYTHONDIR:-"/cstor/dmoise/climate-llm/Prithvi-WxC/python_home"}

CONTVOLS="--bind $(pwd) --bind $DATADIR:/data --bind $PYTHONDIR:/ph_dir"

set -x
srun -n 1 --cpu-bind=none  singularity exec --nv $CONTVOLS $CONT ./start_train.sh 
set +x



