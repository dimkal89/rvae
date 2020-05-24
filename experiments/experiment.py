import os
import torch
import numpy as np

from itertools import chain
from rvae.geoml import nnj

from rvae.variational_inference.train import train_rvae, test_rvae, train_vae, test_vae
from rvae.utils.data_utils import get_mnist_loaders, get_fmnist_loaders
from rvae.models.vae import RVAE, VAE
from rvae.utils.save_utils import save_model


class Experiment():
    def __init__(self, args):
        if not os.path.exists(args.data_dir):
            os.makedirs(args.data_dir)
        if args.dataset.lower() == "mnist":
            self.train_loader, self.test_loader = get_mnist_loaders(args.data_dir, args.batch_size)
        elif args.dataset.lower() == "fmnist":
            self.train_loader, self.test_loader = get_fmnist_loaders(args.data_dir, args.batch_size)
        
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        if not os.path.exists(args.res_dir):
            os.makedirs(args.res_dir)
            # create graph directory
            os.makedirs(args.res_dir + "graphs/")
            # create model samples directory
            os.makedirs(args.res_dir + "samples/")
        
        if args.seed is not None:
            assert(type(args.seed) == int) 
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)

        if args.model.lower() == "rvae":
            self.model = RVAE(784, args.latent_dim, args.batch_size, args.num_centers, 
                              args.enc_layers, args.dec_layers, nnj.Softplus, nnj.Sigmoid,
                              args.rbf_beta, args.rec_b)
        elif args.model.lower() == "vae":
            self.model = VAE(784, args.latent_dim, args.num_centers, args.num_components,
                             args.enc_layers, args.dec_layers, nnj.Softplus, nnj.Sigmoid,
                             args.rbf_beta, args.rec_b)
        
        self.batch_size = args.batch_size
        self.mu_epochs = args.mu_epochs
        self.sigma_epochs = args.sigma_epochs
        self.warmup_learning_rate = args.warmup_learning_rate
        self.sigma_learning_rate = args.sigma_learning_rate
        self.log_invl = args.log_invl
        self.save_invl = args.save_invl
        self.device = args.device
        
    def train(self):
        self.model = self.model.to(self.device)
        # ================= RVAE =================
        if isinstance(self.model, RVAE):
            warmup_optimizer = torch.optim.Adam(
                chain(
                    self.model.encoder.parameters(),
                    self.model.q_mu.parameters(),
                    self.model.q_t.parameters(),
                    self.model.p_mu.parameters()), 
                lr=self.warmup_learning_rate
            )
            sigma_optimizer = torch.optim.Adam(
                chain(
                    self.model.p_sigma.parameters(), 
                    [self.model.pr_means, self.model.pr_t]),
                lr=self.sigma_learning_rate
            )

            # encoder/decoder mean optimization
            for epoch in range(1, self.mu_epochs + 1):
                loss = train_rvae(epoch, self.train_loader, self.batch_size, self.model, 
                                  warmup_optimizer, self.log_invl, self.device)
                print("\tEpoch: {} (warmup phase), negative ELBO: {:.3f}".format(epoch, loss))
            
            self.model.switch = False
            self.model._mean_warmup = False
            self.model._update_latent_codes(self.train_loader)
            self.model._update_RBF_centers(beta=0.01)
            self.model._initialize_prior_means()
            
            # decoder sigma/prior parameters optimization
            for epoch in range(1, self.sigma_epochs + 1):
                loss = train_rvae(epoch, self.train_loader, self.batch_size, self.model, 
                                  sigma_optimizer, self.log_invl, self.device)
                print("\tEpoch: {} (sigma optimization), negative ELBO: {:.3f}".format(epoch, loss))
        
        # ================= VAE =================
        else:
            warmup_optimizer = torch.optim.Adam(
                chain(
                    self.model.encoder.parameters(),
                    self.model.q_mu.parameters(),
                    self.model.q_var.parameters(),
                    self.model.decoder.parameters(),
                    self.model.p_mu.parameters()), 
                lr=self.warmup_learning_rate
            )
            sigma_optimizer = torch.optim.Adam(
                self.model.p_sigma.parameters(),
                lr=self.sigma_learning_rate
            )

            # encoder/decoder mean optimization
            for epoch in range(1, self.mu_epochs + 1):
                loss = train_vae(epoch, self.train_loader, self.batch_size, self.model, 
                                 warmup_optimizer, self.log_invl, self.device)
                print("\tEpoch: {} (warmup phase), negative ELBO: {:.3f}".format(epoch, loss))

            self.model._update_latent_codes(self.train_loader)
            self.model._update_RBF_centers(beta=0.01)

            for epoch in range(1, self.sigma_epochs + 1):
                loss = train_vae(epoch, self.train_loader, self.batch_size, self.model, 
                                 sigma_optimizer, self.log_invl, self.device)
                print("\tEpoch: {} (sigma optimization), negative ELBO: {:.3f}".format(epoch, loss))
                
    def test(self):
        if isinstance(self.model, RVAE):
            loss = test_rvae(self.test_loader, self.batch_size,
                             self.model, self.device)
        else:
            loss = test_vae(self.test_loader, self.batch_size, 
                            self.model, self.device)
        
        print("Test set negative ELBO: {:.3f}".format(loss))
