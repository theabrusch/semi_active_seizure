#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J Latent_CV_5p
### number of core
#BSUB -n 1 
### specify that all cores should be on the same host
#BSUB -R "span[hosts=1]"
### specify the memory needed
#BSUB -R "rusage[mem=100]"
### Number of hours needed
#BSUB -N 
#BSUB -W 10:00
### added outputs and errors to files
#BSUB -o logs/Output5p_%J.out
#BSUB -e logs/Error5p_%J.err

echo "Runnin script..."

python3 main.py 