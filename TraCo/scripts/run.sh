#!/bin/bash
set -e

model=${1}
dataset=${2}
K=${3:-50}
index=${4:-1}


python hierarchical_topic_model.py --model ${model} --dataset ${dataset} --num_topic_str ${K}

./scripts/eva.sh ${model} ${dataset} ${K} ${index}
