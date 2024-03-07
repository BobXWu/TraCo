import torch
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.num_topic = args.num_topic

        self.fc11 = nn.Linear(args.vocab_size, args.model.en1_units)
        self.fc12 = nn.Linear(args.model.en1_units, args.model.en1_units)
        self.fc21 = nn.Linear(args.model.en1_units, args.num_topic)
        self.fc22 = nn.Linear(args.model.en1_units, args.num_topic)
        self.fc1_dropout = nn.Dropout(args.model.dropout)
        self.theta_dropout = nn.Dropout(args.model.dropout)

        self.mean_bn = nn.BatchNorm1d(args.num_topic)
        self.mean_bn.weight.requires_grad = False
        self.logvar_bn = nn.BatchNorm1d(args.num_topic)
        self.logvar_bn.weight.requires_grad = False

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + (eps * std)
        else:
            return mu

    def forward(self, input):
        e1 = F.softplus(self.fc11(input))
        e1 = F.softplus(self.fc12(e1))
        e1 = self.fc1_dropout(e1)
        mu = self.mean_bn(self.fc21(e1))
        logvar = self.logvar_bn(self.fc22(e1))
        z = self.reparameterize(mu, logvar)
        theta = F.softmax(z, dim=1)
        return theta, mu, logvar
