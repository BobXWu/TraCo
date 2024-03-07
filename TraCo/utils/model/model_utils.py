import numpy as np
from collections import defaultdict


def np_tensor_list(tensor_list):
    return np.asarray([item.detach().cpu().numpy() for item in tensor_list], dtype=object)


def parse_item_info(topic_str):
    item_info = topic_str.split()[0]
    layer_id, topic_idx = (int(item.split('-')[1]) for item in item_info.split('_'))
    return layer_id, topic_idx


def round_list(score_list):
    return ' '.join([f'{item:.5f}' for item in score_list])


def print_topic_words(beta, vocab, num_top_word=15):
    topic_str_list = list()
    for i, topic_dist in enumerate(beta):
        topic_words = np.array(vocab)[np.argsort(
            topic_dist)][:-(num_top_word + 1):-1]
        topic_str = ' '.join(topic_words)
        topic_str_list.append(topic_str)
        # print('Topic {}: {}'.format(i + 1, topic_str))
    return topic_str_list


def convert_topicStr_to_dict(topic_str_list):
    """

    Args:
        topic_str_list: [L-0_K-0 w1 w2 w3 ...]. L indicates the layer, and K indicates the topic.
        keep_info (bool, optional): if keep the item info L-0_K-0. Defaults to False.

    Returns:
        hierarchical_topic_dict: {0: ["w1 w2...", "w1 w2..."], 1: ["w1 w2...", "w1 w2..."]}

    """

    hierarchical_topic_dict = defaultdict(list)

    for topic_str in topic_str_list:
        topic_str_items = topic_str.split()
        layer_id, k = parse_item_info(topic_str)

        dict_item = ' '.join(topic_str_items)

        hierarchical_topic_dict[layer_id].append(dict_item)

    return hierarchical_topic_dict
