#!/bin/bash
set -e

# CLONE THE GIT REPOSITORY
# This script is executed only once when the notebook instance is created.
cd /home/ec2-user/SageMaker
git clone https://github.com/ArsMedicaTech/decision_trees.git


# Leaving these here just in case...

# OPTIONAL: INSTALL LIBRARIES
# You can also pre-install packages.
# The 'source' commands are important to activate the correct conda environment.
# Uncomment the lines below to install libraries into the pytorch kernel.
#
# source /home/ec2-user/anaconda3/bin/activate pytorch_p310
# pip install --upgrade SomePackage
# source /home/ec2-user/anaconda3/bin/deactivate
