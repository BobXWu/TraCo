import os
import numpy as np
import argparse
import torch

from utils.data.TextData import TextData
from utils.data import file_utils
from utils.model import model_utils
from runners.Runner import Runner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset')
    parser.add_argument('-m', '--model_config')
    parser.add_argument('-k', '--num_topic_str', type=str, default='10-50-200')
    parser.add_argument('--num_top_word', type=int, default=15)
    parser.add_argument('--test_index', type=int, default=1)
    args = parser.parse_args()
    return args


def export_beta(beta, vocab, layer_id, num_top_word=15):
    topic_str_list = model_utils.print_topic_words(beta, vocab, num_top_word=num_top_word)
    for k, topic_str in enumerate(topic_str_list):
        topic_str_list[k] = f'L-{layer_id}_K-{k} {topic_str}'
        print(topic_str_list[k])

    return topic_str_list


def main():
    args = parse_args()

    # loading model configuration
    file_utils.update_args(args, path=f'./configs/{args.model_config}.yaml')

    output_prefix = f'output/{args.dataset}/{args.model_config}_K{args.num_topic_str}_{args.test_index}th'
    file_utils.make_dir(os.path.dirname(output_prefix))

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # split the string to get a list of topic numbers
    args.num_topic_list = [int(item) for item in args.num_topic_str.split('-')]
    args.num_topic = args.num_topic_list[-1]
    args.num_topic_layers = len(args.num_topic_list)

    dataset_handler = TextData(args.dataset, args.training.batch_size, args.device)

    args.vocab_size = dataset_handler.train_data.shape[1]

    runner = Runner(args)

    beta_list = runner.train(dataset_handler)

    # print and save topic words.
    topic_str_list = list()
    for layer_id, num_topic in enumerate(range(len(beta_list))):
        topic_str_list.extend(export_beta(beta_list[layer_id], dataset_handler.vocab, layer_id, args.num_top_word))

    file_utils.save_text(topic_str_list, f'{output_prefix}_T{args.num_top_word}')

    # save inferred topic distributions of training set and testing set.
    train_theta_list = runner.test(dataset_handler.train_data)
    test_theta_list = runner.test(dataset_handler.test_data)

    params_dict = {
        'beta_list': beta_list,
        'train_theta_list': train_theta_list,
        'test_theta_list': test_theta_list
    }

    phi_list = runner.model.get_phi_list()
    if isinstance(phi_list[0], torch.Tensor):
        phi_list = model_utils.np_tensor_list(phi_list)
    params_dict['phi_list'] = phi_list

    np.savez_compressed(f'{output_prefix}_params.npz', **params_dict)


if __name__ == '__main__':
    main()
