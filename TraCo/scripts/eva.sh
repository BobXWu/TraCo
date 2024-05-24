#!/bin/bash
set -e

model=${1}
dataset=${2}
K=${3:-50}
index=${4:-1}


T=15

echo ------ ${model} ${dataset} K=${K} ${index}th `date` ------

prefix=./output/${dataset}/${model}_K${K}_${index}th
topic_path=${prefix}_T${T}
dataset_dir=../data/${dataset}/
params_path=output/${dataset}/${model}_K${K}_${index}th_params.npz

python utils/eva/hierarchical_topic_quality.py --path ${prefix} --dataset ${dataset}
