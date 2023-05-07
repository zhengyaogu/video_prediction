#!/bin/bash
#
#SBATCH --job-name=generate_blur
#SBATCH --output=generate_blur.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=60GB
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --mail-user=mr6555@nyu.edu

module purge;
cd /scratch/mr6555;
source salmanenv/bin/activate;
cd /home/mr6555/sim_vp
python -u generate_blur_images.py;


