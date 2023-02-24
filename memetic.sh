#!/bin/bash

#$ -N memeticTest
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -V
#$ -pe orte 10
#$ -m be
#$ -M 2220220h@umich.mx

conda activate tsp
python memetic.py