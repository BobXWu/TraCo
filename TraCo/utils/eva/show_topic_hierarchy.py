import argparse
import numpy as np
import json

import sys
sys.path.append('./')
from utils.data import file_utils
from utils.model import model_utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    args = parser.parse_args()
    return args


def get_child_topic_idx(phi, topic_idx):
    num_topic, next_num_topic = phi.shape
    num_child_topic = round(next_num_topic / num_topic)

    row = phi[topic_idx]
    child_topic_idx_list = np.argsort(row)[:-(num_child_topic + 1):-1].tolist()

    return child_topic_idx_list


def build_hierarchy(hierarchical_topic_dict, phi_list, topic_idx_list=None, layer_id=0):
    # first layer.
    if topic_idx_list is None:
        topic_idx_list = list(range(phi_list[0].shape[0]))

    # last layer, where layer_id == L-1.
    if layer_id >= len(phi_list):
        # return the topic strings at the last layer.
        hierarchy = np.asarray(hierarchical_topic_dict[layer_id])[topic_idx_list].tolist()
        return hierarchy

    # NOT the last layer.
    hierarchy = dict()
    phi = phi_list[layer_id]

    for topic_idx in topic_idx_list:
        child_topic_idx_list = get_child_topic_idx(phi, topic_idx)
        hierarchy[hierarchical_topic_dict[layer_id][topic_idx]] = build_hierarchy(hierarchical_topic_dict, phi_list, child_topic_idx_list, layer_id + 1)

    return hierarchy


def main():
    args = parse_args()
    data_mat = np.load(f'{args.path}_params.npz', allow_pickle=True)
    phi_list = data_mat['phi_list']

    hierarchical_topic_dict = model_utils.convert_topicStr_to_dict(file_utils.read_text(f'{args.path}_T15'))
    topic_hierarchy = build_hierarchy(hierarchical_topic_dict, phi_list)
    hierarchy_str = json.dumps(topic_hierarchy, indent=4)

    # save hierarchy as json.
    with open(f'{args.path}_hierarchy-T15.json', 'w', encoding='utf-8') as file:
        file.write(hierarchy_str)

    print(hierarchy_str + "\n\n")


if __name__ == '__main__':
    main()
