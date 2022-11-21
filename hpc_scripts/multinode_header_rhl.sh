#Name of job
#Dep name , project name
#PBS -P cse

#PBS -j oe
#PBS -m bea
### Specify email address to use for notification.
#PBS -M $USER@iitd.ac.in
#PBS -l select=${num_nodes}:ngpus=2:ncpus=6${selectos}
## SPECIFY JOB NOW

CURTIME=$(date +%Y%m%d%H%M%S)
module load apps/pythonpackages/3.6.0/pytorch/0.4.1/gpu
#module load apps/anaconda3/4.6.9
## Change to dir from where script was launched



