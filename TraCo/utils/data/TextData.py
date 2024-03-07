import torch
from torch.utils.data import DataLoader
import scipy.sparse
import scipy.io

from utils.data import file_utils


class TextData:
    def __init__(self, dataset, batch_size, device):
        dataset_path = f'../data/{dataset}'
        self.train_data, self.test_data, self.vocab = self.load_data(dataset_path)
        self.vocab_size = len(self.vocab)

        self.train_data = torch.from_numpy(self.train_data).to(device)
        self.test_data = torch.from_numpy(self.test_data).to(device)

        self.train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_data, batch_size=batch_size, shuffle=False)

    def load_data(self, path):
        train_data = scipy.sparse.load_npz(f'{path}/train_bow.npz').toarray().astype('float32')
        test_data = scipy.sparse.load_npz(f'{path}/test_bow.npz').toarray().astype('float32')

        vocab = file_utils.read_text(f'{path}/vocab.txt')

        return train_data, test_data, vocab
