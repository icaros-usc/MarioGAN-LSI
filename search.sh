#!/bin/sh
#
# Usage: gputest.sh
# Change job name and email address as needed 
#        

# -- our name ---
#$ -N MarioGAN-LSI
#$ -S /bin/sh
# Make sure that the .e and .o file arrive in the
#working directory
#$ -cwd
#Merge the standard out and standard error to one file
#$ -j y
# Send mail at submission and completion of script
# Specify CPU queue
#$ -q medium
#$ -t 1-2
#$ -l mem_free=14.0G
/bin/echo Running on host: `hostname`.
/bin/echo In directory: `pwd`
/bin/echo Starting on: `date`

module load anaconda
module list
# Load conda
source activate MarioGAN-LSI
# Run the search
python3 search/run_search.py -w $SGE_TASK_ID -c search/config/experiment/experiment.tml
