import os
import torch
import numpy as np

from itertools import chain

from rvae.geoml import nnj
from rvae.variational_inference.train import train_rvae, test_rvae, train_vae, test_vae
from rvae.utils.data_utils import get_mnist_loaders, get_fmnist_loaders, get_kmnist_loaders
from rvae.models.vae import RVAE, VAE
from rvae.utils.save_utils import save_model, load_model


class Experiment():
    def __init__(self, args):
        self.dataset = args.dataset.lower()

        if not os.path.exists(args.data_dir):
            os.makedirs(args.data_dir)
        if self.dataset == "mnist":
            self.train_loader, self.test_loader = get_mnist_loaders(args.data_dir, args.batch_size)
            in_dim = 784
        elif self.dataset == "fmnist":
            self.train_loader, self.test_loader = get_fmnist_loaders(args.data_dir, args.batch_size)
            in_dim = 784
        elif self.dataset == "kmnist":
            self.train_loader, self.test_loader = get_kmnist_loaders(args.data_dir, args.batch_size)
            in_dim = 784
        
        self.rvae_save_dir = os.path.join(args.save_dir, "RVAE/")
        self.vae_save_dir = os.path.join(args.save_dir, "VAE/")
        if not os.path.exists(self.rvae_save_dir):
            os.makedirs(self.rvae_save_dir)
        if not os.path.exists(self.vae_save_dir):
            os.makedirs(self.vae_save_dir)

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
            self.model = RVAE(in_dim, args.latent_dim, args.num_centers, args.enc_layers, 
                              args.dec_layers, nnj.Softplus, nnj.Sigmoid, args.rbf_beta, args.rec_b)
        elif args.model.lower() == "vae":
            self.model = VAE(in_dim, args.latent_dim, args.num_centers, args.num_components,
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
                loss, _, _ = train_rvae(epoch, self.train_loader, self.batch_size, self.model, 
                                        warmup_optimizer, self.log_invl, self.device)
                print("\tEpoch: {} (warmup phase), negative ELBO: {:.3f}".format(epoch, loss))

            # warmup checkpoint            
            savepath = os.path.join(self.rvae_save_dir, self.dataset+"_warmup")
            save_model(self.model, sigma_optimizer, 0, None, savepath)

            self.model.switch = False
            self.model._update_latent_codes(self.train_loader)
            self.model._update_RBF_centers(beta=0.01)
            self.model._mean_warmup = False
            self.model._initialize_prior_means()
            
            # decoder sigma/prior parameters optimization
            for epoch in range(1, self.sigma_epochs + 1):
                loss, _, _ = train_rvae(epoch, self.train_loader, self.batch_size, self.model, 
                                        sigma_optimizer, self.log_invl, self.device)
                print("\tEpoch: {} (sigma optimization), negative ELBO: {:.3f}".format(epoch, loss))

            savepath = os.path.join(self.rvae_save_dir,
                                    self.dataset+"_epoch"+str(epoch)+"ckpt")
            save_model(self.model, sigma_optimizer, epoch, loss, savepath)

        # ================= VAE =================
        else:
            warmup_optimizer = torch.optim.Adam(
                chain(self.model.encoder.parameters(),
                      self.model.q_mu.parameters(),
                      self.model.q_var.parameters(),
                      self.model.decoder.parameters(),
                      self.model.p_mu.parameters()), 
                      lr=self.warmup_learning_rate
            )

            if self.model.num_components > 1:
                sigma_optimizer = torch.optim.Adam(
                    chain(self.model.p_sigma.parameters(), self.model.means.parameters()),
                    lr=self.sigma_learning_rate
                )
            else:
                sigma_optimizer = torch.optim.Adam(
                    self.model.p_sigma.parameters(),
                    lr=self.sigma_learning_rate
                )

            # encoder/decoder mean optimization
            for epoch in range(1, self.mu_epochs + 1):
                loss, _, _ = train_vae(epoch, self.mu_epochs, self.train_loader, self.batch_size, self.model, 
                                       warmup_optimizer, self.log_invl, self.device)
                print("\tEpoch: {} (warmup phase), negative ELBO: {:.3f}".format(epoch, loss))

            # warmup checkpoint            
            savepath = os.path.join(self.rvae_save_dir, self.dataset+"_warmup")
            save_model(self.model, sigma_optimizer, 0, None, savepath)

            self.model.switch = False
            self.model._update_latent_codes(self.train_loader)
            self.model._update_RBF_centers(beta=0.01)

            for epoch in range(1, self.sigma_epochs + 1):
                loss, _, _ = train_vae(epoch, 1, self.train_loader, self.batch_size, self.model, 
                                       sigma_optimizer, self.log_invl, self.device)
                print("\tEpoch: {} (sigma optimization), negative ELBO: {:.3f}".format(epoch, loss))

                if epoch % self.save_invl == 0:
                    savepath = os.path.join(self.vae_save_dir, 
                                            self.dataset+"_K"+str(self.model.num_components)+"epoch"+str(epoch)+".ckpt")
                    save_model(self.model, sigma_optimizer, epoch, loss, savepath)
                
    def eval(self, pretrained_path=None):
        # load checkpoint
        if pretrained_path is not None:
            placeholder_optimizer = torch.optim.Adam(
                self.model.p_sigma.parameters(),
                lr=1e-3
            )
            load_model(pretrained_path, self.model, placeholder_optimizer)

        if isinstance(self.model, RVAE):
            loss, log_cond, KL = test_rvae(self.test_loader, self.batch_size, self.model, self.device)
        else:
            loss, log_cond, KL = test_vae(self.test_loader, self.batch_size, self.model, self.device)
        
        print("Test set negative ELBO: {:.3f}, negative conditional: {:.3f}, KL: {:.3f}".format(loss, log_cond, KL))
