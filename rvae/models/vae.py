import sys
import torch
import numpy as np

from torch import nn
from sklearn.cluster import KMeans
from ..geoml import nnj
from .templates import MLP, NonLinear
from ..misc import multibrownian_motion_sample

 
class RVAE(nn.Module):
    """
    
    IMPORTANT NOTES - READ:
        -   The smaller the b value is in nnj.Reciprocal (in the RBF net definition), the better
            the variance estimates in the regions where we have latent codes (within the manifold).
        -   The lower the beta value is in nnj.RBF, the looser the "borders" around your latent codes
            is going to be. This makes sense as beta is a multiplicative factor on the pairwise distances
            in the RBF kernel.
    """
    def __init__(self, in_dim, latent_dim, batch_size, num_centers, enc_layers, 
                 dec_layers, act, out_fn, rbf_beta, rec_b):
        super(RVAE, self).__init__()
        self.batch_size = batch_size
        self.num_centers = num_centers
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self._mean_warmup = True
        self.switch = True
        self.encoder = MLP(in_dim, enc_layers, act, None)
        self.q_mu = nn.Sequential(
            nn.Linear(enc_layers[-1], latent_dim)
        )
        self.q_t = nn.Sequential(
            nn.Linear(enc_layers[-1], 1),
            nn.Softplus(),
            nn.Hardtanh(min_val=1e-4, max_val=5.)
        )
        self.dummy_pmu = MLP(latent_dim, dec_layers, act, in_dim, out_fn)
        self.p_mu = MLP(latent_dim, dec_layers, act, in_dim, out_fn)
        self.p_sigma = nnj.Sequential(
            nnj.RBF(self.latent_dim, num_points=num_centers, beta=rbf_beta),
            nnj.PosLinear(num_centers, self.in_dim, bias=False),
            nnj.Reciprocal(b=rec_b),
            nnj.Sqrt()
        )
        self._latent_codes = None
        self.pr_means = torch.nn.Parameter(torch.zeros(latent_dim), requires_grad=True)
        self.pr_t = torch.nn.Parameter(torch.ones(1), requires_grad=True)

    def encode(self, x):
        h = self.encoder(x)
        q_mu = self.q_mu(h)
        q_t = self.q_t(h)

        eps = torch.randn_like(q_mu)

        # reparameterize
        z = (q_mu + q_t.sqrt() * eps).view(-1, self.latent_dim)
        
        return z, q_mu, q_t

    def _update_latent_codes(self, data_loader):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        codes = []
        for _, (data, labels) in enumerate(data_loader):
            dim1, dim2 = data.shape[-2], data.shape[-1]
            z, _, _ = self.encode(data.view(-1, dim1 * dim2).to(device))
            codes.append(z)
        self._latent_codes = torch.cat(codes, dim=0).view(-1, self.latent_dim)
    
    def _update_RBF_centers(self, beta=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        kmeans = KMeans(n_clusters=self.num_centers)
        kmeans.fit(self._latent_codes.detach().cpu().numpy())
        self.p_sigma._modules['0'].points.data = torch.from_numpy(kmeans.cluster_centers_.astype(np.float32)).to(device)
        self.p_sigma._modules['0'].beta = beta

    def _initialize_prior_means(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        kmeans = KMeans(n_clusters=1)
        kmeans.fit(self._latent_codes.detach().cpu().numpy())
        self.pr_means.data = torch.from_numpy(kmeans.cluster_centers_.astype(np.float32)).to(device)

    def decode(self, z, jacobian):
        if self._mean_warmup:
            if jacobian:
                mu, J_mu = self.p_mu(z, jacobian)
                sigma, J_sigma = self.p_sigma(z, jacobian)

                J_mu = torch.einsum("bij,bkj->bij", J_mu, J_mu)
                J_sigma = torch.einsum("bij,bkj->bij", J_sigma, J_sigma)
                
                return mu, sigma, J_mu
            else:
                mu = self.p_mu(z, jacobian)
                sigma = self.p_sigma(z, jacobian)

                return mu, sigma
        else:
            if jacobian:
                sigma, J_sigma = self.p_sigma(z, jacobian)
                mu, J_mu = self.p_mu(z, jacobian)
                
                # Get quadratic forms of Jacobians
                J_mu = torch.einsum("bij,bkj->bij", J_mu, J_mu)
                J_sigma = torch.einsum("bij,bkj->bij", J_sigma, J_sigma)
                # Compute metric
                G = J_mu + J_sigma

                return mu, sigma, G
            else:
                mu = self.p_mu(z)
                sigma = self.p_sigma(z)
                return mu, sigma

    def generate(self, n_steps, epoch, device, viz=False):
        """Generate samples from a Brownian motion on the manifold.
        """
        self.eval()
        k = 1/self.n_components * torch.ones(self.n_components)
        z = multibrownian_motion_sample(k, self.pr_means, self.pr_t, self.latent_dim, n_steps, self) 
        x_mu, _ = self.decode(z[::n_steps//100], False)
        
        # if viz:
        #     assert z.shape[1] == 2
        #     plot_brownian_motion(self, z, "./vae_example/results/graphs/bm_epoch_"+str(epoch)+".pdf", device, True)
        
        return x_mu

    def metric(self, z):
        if z.dim() == 1:
            z = z.unsqueeze(0)

        _, J_mu = self.p_mu(z, True)
        J_mu = torch.einsum("bji,bjk->bik", J_mu, J_mu)
        _, J_sigma = self.p_sigma(z, True)
        J_sigma = torch.einsum("bji,bjk->bik", J_sigma, J_sigma)

        return J_mu + J_sigma
            
    def forward(self, x, jacobian=False):
        z, q_mu, q_var = self.encode(x)
        if jacobian:
            p_mu, p_sigma, G = self.decode(z, jacobian)
            return p_mu, p_sigma, z, q_mu, q_var, G
        else:
            p_mu, p_sigma = self.decode(z, jacobian)
            return p_mu, p_sigma, z, q_mu, q_var   


class VAE(nn.Module):
    def __init__(self, in_dim, latent_dim, num_centers, num_components, enc_layers, 
                 dec_layers, act, out_fn, rbf_beta, rec_b):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_centers = num_centers
        self.num_components = num_components
        self.switch = True
        self.encoder = MLP(in_dim, enc_layers, act, None)
        self.q_mu = nn.Sequential(
            nn.Linear(enc_layers[-1], latent_dim)
        )
        self.q_var = nn.Sequential(
            nn.Linear(enc_layers[-1], latent_dim),
            nn.Softplus(),
            nn.Hardtanh(min_val=1e-4, max_val=5.)
        )
        self.decoder = MLP(latent_dim, dec_layers, act, None)
        self.p_mu = nn.Sequential(
            nn.Linear(dec_layers[-1], in_dim),
            out_fn()
        )
        self.p_sigma = nnj.Sequential(
            nnj.RBF(self.latent_dim, num_points=num_centers, beta=rbf_beta),
            nnj.PosLinear(num_centers, in_dim, bias=False),
            nnj.Reciprocal(b=rec_b),
            nnj.Sqrt()
        )
        self._latent_codes = None

        # if num_components > 1 assume VampPrior
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if num_components > 1:
            self.means = NonLinear(num_components, in_dim, activation=nn.Hardtanh(min_val=0., max_val=1.))
            # self.means.linear.weight.normal_(mean=0., std=0.01)
            self.dummy_input = torch.tensor(torch.eye(num_components, num_components), requires_grad=False, device=device)

    def encode_prior(self):
        u = self.means(self.dummy_input)
        return self.encode(u)

    def encode(self, x):
        h = self.encoder(x)
        return self.q_mu(h), self.q_var(h)

    def reparameterize(self, mu, var):
        eps = torch.randn_like(var)
        return mu + eps * var.sqrt()

    def _update_latent_codes(self, data_loader):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        codes = []
        for _, (data, labels) in enumerate(data_loader):
            dim1, dim2 = data.shape[-2], data.shape[-1]
            q_mu, q_var = self.encode(data.view(-1, dim1 * dim2).to(device))
            z = self.reparameterize(q_mu, q_var)
            codes.append(z)
        self._latent_codes = torch.cat(codes, dim=0).view(-1, self.latent_dim)

    def _update_RBF_centers(self, beta=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        kmeans = KMeans(n_clusters=self.num_centers)
        kmeans.fit(self._latent_codes.detach().cpu().numpy())
        self.p_sigma._modules['0'].points.data = torch.from_numpy(kmeans.cluster_centers_.astype(np.float32)).to(device)
        self.p_sigma._modules['0'].beta = beta

    def decode(self, z):
        h = self.decoder(z)
        mu = self.p_mu(h)
        sigma = self.p_sigma(z)

        return mu, sigma**2

    def generate(self, batch_size, epoch):
        z = torch.randn(batch_size, self.latent_dim)
        p_mu, _ = self.decode(z)
        return p_mu

    def forward(self, x):
        q_mu, q_var = self.encode(x)
        z = self.reparameterize(q_mu, q_var)
        p_mu, p_var = self.decode(z)

        if self.num_components > 1:
            pr_mu, pr_var = self.encode_prior()
        else:
            pr_mu = torch.zeros_like(q_mu)
            pr_var = torch.ones_like(q_var)

        return p_mu, p_var, z, q_mu, q_var, pr_mu, pr_var
