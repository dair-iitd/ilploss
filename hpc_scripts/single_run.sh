#Name of job
#Dep name , project name
#PBS -P cse
#PBS -j oe
#PBS -l select=1:ngpus=1:ncpus=4${selectos}
## SPECIFY JOB NOW

JOBNAME=HPRSR
CURTIME=$(date +%Y%m%d%H%M%S)
cd $PBS_O_WORKDIR 
##module load apps/pythonpackages/3.6.0/pytorch/0.4.1/gpu
##module load apps/anaconda3/4.6.9
## Change to dir from where script was launched
##conda activate tr3
