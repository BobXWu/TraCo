import torch
from torch import nn
import torch.nn.functional as F
from models.TPD import TPD
from models.Encoder import Encoder
from models.CDDecoder import CDDecoder
from models import utils


class TraCo(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.num_layers = args.num_topic_layers

        self.bottom_word_embeddings  = nn.init.trunc_normal_(torch.empty(args.vocab_size, args.model.embedding_dim), std=0.1)
        self.bottom_word_embeddings = nn.Parameter(F.normalize(self.bottom_word_embeddings))

        self.topic_embeddings_list = nn.ParameterList([])
        for num_topic in args.num_topic_list:
            topic_embeddings = torch.empty((num_topic, self.bottom_word_embeddings.shape[1]))
            nn.init.trunc_normal_(topic_embeddings, std=0.1)
            topic_embeddings = nn.Parameter(F.normalize(topic_embeddings))
            self.topic_embeddings_list.append(topic_embeddings)

        self.TPD = TPD(sinkhorn_alpha=args.model.sinkhorn_alpha, OT_max_iter=args.model.OT_max_iter)
        self.CDDecoder = CDDecoder(self.num_layers, args.vocab_size, args.model.bias_p, args.model.bias_topk)

        self.encoder = Encoder(args)

    def get_beta(self):
        beta_list = list()
        for layer_id, num_topic in enumerate(self.args.num_topic_list):
            topic_embeddings = self.topic_embeddings_list[layer_id]
            dist = utils.pairwise_euclidean_distance(topic_embeddings, self.bottom_word_embeddings)
            beta = F.softmax(-dist / self.args.model.beta_temp, dim=0)
            beta_list.append(beta)

        return beta_list

    def get_phi_list(self):
        transp_list = self.transp_list
        # multiply each transp with the topic size in the next layer.
        phi_list = [item * item.shape[1] for item in transp_list]
        return phi_list

    def get_theta(self, input_bow):
        theta_list = list()
        bottom_theta, mu, logvar = self.encoder(input_bow)
        phi_list = self.get_phi_list()
        # from bottom to top
        for layer_id in range(self.num_layers)[::-1]:
            if layer_id == self.num_layers - 1:
                theta_list.append(bottom_theta)
            else:
                last_theta = theta_list[-1]
                theta = torch.matmul(last_theta, phi_list[layer_id].T)
                theta_list.append(theta)

        theta_list = theta_list[::-1]

        if self.training:
            return theta_list, mu, logvar
        else:
            return theta_list

    def forward(self, input_bow):
        loss = 0.

        loss_TPD, self.transp_list = self.TPD(self.topic_embeddings_list, self.args.model.weight_loss_TPD)
        loss += loss_TPD

        theta_list, mu, logvar = self.get_theta(input_bow)
        loss_KL = self.compute_loss_KL(mu, logvar)
        loss_KL = loss_KL.mean()
        loss += loss_KL

        beta_list = self.get_beta()

        recon_loss = self.CDDecoder(input_bow, theta_list, beta_list)
        loss += recon_loss

        rst_dict = {
            'loss': loss,
        }

        return rst_dict

    def compute_loss_KL(self, mu, logvar, mu_prior=None):
        mu_dim = mu.shape[1]
        a = torch.ones((1, mu_dim), device=mu.device)
        if mu_prior is None:
            mu_prior = (torch.log(a).T - torch.mean(torch.log(a), 1)).T
        var_prior = (((1.0 / a) * (1 - (2.0 / mu_dim))).T + (1.0 / (mu_dim * mu_dim)) * torch.sum(1.0 / a, 1)).T

        var = logvar.exp()
        var_division = var / var_prior
        diff = mu - mu_prior
        diff_term = diff * diff / var_prior
        logvar_division = var_prior.log() - logvar
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(axis=1) - mu_dim)
        return KLD
