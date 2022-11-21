#Name of job
#Dep name , project name
#PBS -P cse
##PBS -P darpa.ml.cse
##PBS -P parags.p2.54
##PBS -q high 
#PBS -j oe
#PBS -m bea
### Specify email address to use for notification.
#PBS -M $USER@iitd.ac.in
#PBS -l select=${num_nodes}:ngpus=2:ncpus=4${selectos}
##PBS -l select=${num_nodes}:ngpus=2:ncpus=2${selectos}
## SPECIFY JOB NOW

CURTIME=$(date +%Y%m%d%H%M%S)
##module load apps/pythonpackages/3.6.0/pytorch/0.4.1/gpu
##module load apps/anaconda3/4.6.9
##module load apps/anaconda/3
##module load apps/pytorch/1.5.0/gpu/anaconda3
## Change to dir from where script was launched
##cd $PBS_O_WORKDIR


