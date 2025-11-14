#!/bin/bash
#PBS -N my_cuprate_job         ## job name
#PBS -l nodes=1:ppn=1         ## single-node job, number of cores
#PBS -l walltime=00:30:00      ## max. 2h of wall time
#PBS -l mem=8gb
#PBS -o /kyukon/home/gent/505/vsc50528/electron_doped_cuprates/job_logs/${PBS_JOBNAME}.o${PBS_JOBID}  ## stdout
#PBS -e /kyukon/home/gent/505/vsc50528/electron_doped_cuprates/job_logs/${PBS_JOBNAME}.e${PBS_JOBID}  ## stderr

cd $PBS_O_WORKDIR
module load Python/3.13.5-GCCcore-14.3.0
source venv/bin/activate 
python job_script.py 

