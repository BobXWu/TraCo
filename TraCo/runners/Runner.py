import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm
from utils.model import model_utils
from models.TraCo import TraCo


class Runner:
    def __init__(self, args):
        self.args = args
        self.model = TraCo(args)
        self.model = self.model.to(args.device)

    def make_optimizer(self,):
        args_dict = {
            'params': self.model.parameters(),
            'lr': self.args.training.learning_rate,
        }

        optimizer = torch.optim.Adam(**args_dict)
        return optimizer

    def train(self, dataset_handler):
        optimizer = self.make_optimizer()

        data_size = len(dataset_handler.train_loader.dataset)

        for epoch in tqdm(range(1, self.args.training.epochs + 1), leave=False):
            self.model.train()
            loss_rst_dict = defaultdict(float)

            for batch_data in dataset_handler.train_loader:

                rst_dict = self.model(batch_data)
                batch_loss = rst_dict['loss']

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                for key in rst_dict:
                    loss_rst_dict[key] += rst_dict[key] * len(batch_data)

            output_log = f'Epoch: {epoch:03d}'
            for key in loss_rst_dict:
                output_log += f' {key}: {loss_rst_dict[key] / data_size :.3f}'

            print(output_log)

        beta_list = self.model.get_beta()
        beta_array = model_utils.np_tensor_list(beta_list)

        return beta_array

    def test(self, input_data):
        data_size = input_data.shape[0]

        hierarchical_theta_list = np.empty(len(self.args.num_topic_list), object)
        for layer_id in range(len(self.args.num_topic_list)):
            hierarchical_theta_list[layer_id] = np.zeros((data_size, self.args.num_topic_list[layer_id]))

        all_idx = torch.split(torch.arange(data_size), self.args.training.batch_size)

        with torch.no_grad():
            self.model.eval()
            for idx in all_idx:
                batch_input = input_data[idx]
                batch_theta_list = self.model.get_theta(batch_input)

                for layer_id in range(len(self.args.num_topic_list)):
                    hierarchical_theta_list[layer_id][idx] = batch_theta_list[layer_id].cpu().numpy()

        return hierarchical_theta_list
