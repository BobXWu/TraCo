#!/bin/bash
set -e

model=${1}
dataset=${2}
K=${3:-50}
index=${4:-1}
eva_type=${5}


python hierarchical_topic_model.py --model ${model} --dataset ${dataset} --num_topic_str ${K}

./scripts/eva.sh ${model} ${dataset} ${K} ${index} ${eva_type}

