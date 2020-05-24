import numpy as np
from torch import nn
from sklearn.datasets import make_circles
from nn_models import MLP
from torch.optim import Adam
from geoml import nnj
from data_utils import get_circle_loaders
from debugging_utils import visualize_variance
from losses import riemannian_loss
from itertools import chain
import torch
import matplotlib.pyplot as plt
# fix random seed
torch.manual_seed(1989)
np.random.seed(1989)

class RBFVAE(nn.Module):
    """
    
    IMPORTANT NOTES - READ:
        -   The smaller the b value is in nnj.Reciprocal (in the RBF net definition), the better
            the variance estimates in the regions where we have latent codes (within the manifold).
        -   The lower the beta value is in nnj.RBF, the looser the "borders" around your latent codes
            is going to be. This makes sense as beta is a multiplicative factor on the pairwise distances
            in the RBF kernel.
    """
    def __init__(self, in_dim, latent_dim, n_centers, n_components, enc_layers, dec_layers, act, out_fn, rbf_beta, rec_b, device, trainable):
        super(RBFVAE, self).__init__()
        self.device = device
        self.n_components = n_components
        self.n_centers = n_centers
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self._mean_warmup = True
        self.switch = True
        self.encoder = MLP(in_dim, enc_layers[:-1], enc_layers[-1], act)
        self.q_mu = nn.Sequential(
            nn.Linear(enc_layers[-1], latent_dim)
        )
        self.q_t = nn.Sequential(
            nn.Linear(enc_layers[-1], 1),
            nn.Softplus(),
            nn.Hardtanh(min_val=1e-4, max_val=5.)
        )
        self.dummy_pmu = MLP(latent_dim, dec_layers, in_dim, act, out_fn)
        self.p_mu = MLP(latent_dim, dec_layers, in_dim, act, out_fn)
        self.p_sigma = MLP(latent_dim, dec_layers, in_dim, act, nnj.Softplus)
        if self.n_components == 1:
            self.pr_means = torch.nn.Parameter(torch.zeros(n_components, latent_dim), requires_grad=trainable)
        else:
            self.pr_means = torch.nn.Parameter(torch.randn(n_components, latent_dim, device=device), requires_grad=trainable)
        self.pr_t = torch.nn.Parameter(torch.ones(n_components, 1), requires_grad=trainable)

    def encode(self, x, mc_samples):
        h = self.encoder(x)
        q_mu = self.q_mu(h)
        q_t = self.q_t(h)

        eps = torch.randn(q_mu.shape[0], mc_samples, q_mu.shape[1], device=self.device).view(-1, self.latent_dim)
        q_mu = q_mu.expand_as(eps).view(-1, self.latent_dim)
        q_t = q_t.expand(q_mu.shape[0], 1)

        # reparameterize
        z = (q_mu + q_t.sqrt() * eps).view(-1, self.latent_dim)
        # g_inv = self.metric(z).inverse()

        # z = q_mu + torch.einsum("bij,bi->bi", q_t.sqrt().unsqueeze(1) * g_inv, eps)

        return z, q_mu, q_t
    
    # def _update_RBF_centers(self, x, beta=None):
    #     if torch.cuda.is_available():
    #         x = x.to("cuda")
    #     z, _, _ = self.encode(x, mc_samples=1)

    #     kmeans = KMeans(n_clusters=self.n_centers)
    #     kmeans.fit(z.detach().cpu().numpy())
    #     self.p_sigma._modules['0'].points.data = torch.from_numpy(kmeans.cluster_centers_.astype(np.float32)).to(self.device)
    #     self.p_sigma._modules['0'].beta = beta

    # def _initialize_prior_means(self, x):
    #     if torch.cuda.is_available():
    #         x = x.to("cuda")
    #     z, _, _ = self.encode(x, mc_samples=1)

    #     kmeans = KMeans(n_clusters=self.n_components)
    #     kmeans.fit(z.detach().cpu().numpy())
    #     self.pr_means.data = torch.from_numpy(kmeans.cluster_centers_.astype(np.float32)).to(self.device)

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

    # def generate(self, n_steps, epoch, device, viz=False):
    #     """Generate samples from a Brownian motion on the manifold.
    #     """
    #     self.eval()
    #     k = 1/self.n_components * torch.ones(self.n_components)
    #     z = multibrownian_motion_sample(k, self.pr_means, self.pr_t * 10, self.latent_dim, n_steps, self) 
    #     x_mu, _ = self.decode(z[::n_steps//100], False)
        
    #     if viz:
    #         assert z.shape[1] == 2
    #         brownian_motion_plot(self, z, "./vae_example/results/graphs/bm_epoch_"+str(epoch)+".png", device, True)
        
    #     return x_mu

    def metric(self, z):
        if z.dim() == 1:
            z = z.unsqueeze(0)

        _, J_mu = self.p_mu(z, True)
        J_mu = torch.einsum("bji,bjk->bik", J_mu, J_mu)
        _, J_sigma = self.p_sigma(z, True)
        J_sigma = torch.einsum("bji,bjk->bik", J_sigma, J_sigma)

        return J_mu + J_sigma
            
    def forward(self, x, mc_samples=1, jacobian=False):
        z, q_mu, q_var = self.encode(x, mc_samples)
        if jacobian:
            p_mu, p_sigma, G = self.decode(z, jacobian)
            return p_mu, p_sigma, z, q_mu, q_var, G
        else:
            p_mu, p_sigma = self.decode(z, jacobian)
            return p_mu, p_sigma, z, q_mu, q_var

# embedding = Sequential(Linear(2, 100), ReLU())
# data = make_circles(3200, factor=0.99)
# data = data[0]
# plt.scatter(data[2500:, 0], data[2500:, 1], s=0.7, c='b')
# plt.savefig("./circle_scatter.eps")
# plt.close()
# data = embedding(torch.from_numpy(data[0]).float())
# data = data.detach()
# torch.save(data[:2500], "./vae_example/data/circle_train.ptc")
# torch.save(data[2500:], "./vae_example/data/circle_test.ptc")

def train_rbfvae(epoch, train_loader, b_sz, model, optimizer, log_invl, device):
    model.train()
    train_loss = 0.
    n_batches = len(train_loader.dataset.data)//b_sz

    for i, data in enumerate(train_loader):
        prefactor = min(1, epoch/25)
        optimizer.zero_grad()
        if isinstance(data, list):
            data = data[0].to(device)
        else:
            data = data.to(device)
        p_mu, p_sigma, z, q_mu, q_var = model(data)
        model.dummy_pmu.load_state_dict(model.p_mu.state_dict())

        # XXX XXX DEBUGGING XXX XXX
        if model.switch:
            # p_sigma = 0.06 * p_sigma.clone()
            # p_sigma = torch.mean((q_mu - 0)**2)
            p_sigma = torch.tensor([1e-2])
            # p_sigma = torch.sqrt((p_mu - p_mu.mean()))
        
        k = 1/model.n_components * torch.ones(model.n_components)
        loss, log_pxz, kld = riemannian_loss(data, p_mu, p_sigma, z, q_mu, q_var.squeeze(), 
                                        model.pr_means, model.pr_t, k, model, prefactor)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if i % log_invl == 0:
            print("Epoch: {}, batch: {}, loss: {:.3f}, KL: {:.3f}, density: {:.3f}, q var: {:.4f}, p sigma: {:.4f}".format(
                  epoch, i, loss.item(), kld.item(), log_pxz.item(), q_var.mean().item(), p_sigma.pow(2).mean().item()
            ))
    
    avg_loss = train_loss/n_batches
    print("Avg epoch loss: {:.3f}".format(avg_loss))

    return avg_loss


def test_rbfvae(epoch, test_loader, b_sz, model, mc_samples, device, viz=False, viz_geodesics=False):
    model.eval()
    test_loss = 0
    test_kld = 0
    samples = []
    n_batches = len(test_loader.dataset.data)//b_sz
    iw_bound = True if mc_samples > 1 else False
    samples = []

    with torch.no_grad():
        for _, data in enumerate(test_loader):
            if isinstance(data, list):
                data = data[0].to(device)
            else:
                data = data.to(device)
            p_mu, p_sigma, z, q_mu, q_t = model(data)
            
            k = 1/model.n_components * torch.ones(model.n_components)
            loss = riemannian_loss(data, p_mu, p_sigma, z, q_mu, q_t, 
                                    model.pr_means, model.pr_t, k, model, 1, iw_bound)
            if type(loss) == tuple:
                test_loss += loss[0]    # ELBO
                test_kld += loss[2]     # KL div
                samples.append(z)
            else:
                test_loss += loss
            
        # # XXX XXX DEBUGGING XXX XXX
        samples.append(z)
        # show_reconstructions(test_loader, model, device)

    test_loss /= n_batches
    test_kld /= n_batches
    # if viz:
        # samples = torch.stack(samples[:-1]).view(-1, 2)
        # visualize_variance(model, 784, samples, "./vae_example/results/graphs/stddev_epoch_"+str(epoch)+".png", device, True)
        # metric_scatter(model, samples, True, "./vae_example/results/graphs/metric_epoch_"+str(epoch)+".png", device, viz_geodesics=viz_geodesics)
    if type(loss) == tuple:
        print("Test set loss: {:.3f}, pxz density: {:.3f}, KL div: {:.3f} \n".format(test_loss.item(), (test_loss-test_kld).item(), test_kld.item()))
    else:
        print("Test set IW-ELBO: {:.3f}".format(test_loss))
    
    samples = torch.stack(samples, dim=0)
    return test_loss, samples

def show_prior_means(model, data_loader, device, log_scale=True):
    sq_side = 4.
    # set up the grid
    x = np.arange(-sq_side, sq_side, 0.15)
    y = np.arange(-sq_side, sq_side, 0.15)
    xx, yy = np.meshgrid(x, y)
    coords = np.stack((xx.flatten(), yy.flatten()))
    coords = coords.transpose().astype(np.float32)
    coords = torch.from_numpy(coords).to(device)

    # evaluate metric on points on the grid
    model.eval()
    G = model.metric(coords)
    det = G.det().abs()
    
    # get the log of the canonical volume measure sqrt(det(G))
    det = det.detach_().cpu().numpy()
    if log_scale:
        measure = 1/2 * np.log(det)
    else:
        measure = np.sqrt(det)

    # plot background
    num_grid = xx.shape[0]
    measure = measure.reshape(num_grid, num_grid)
    plt.imshow(measure,
               interpolation='gaussian',
               origin='lower',
               extent=(-sq_side, sq_side,
                       -sq_side, sq_side),
               cmap=plt.cm.seismic,
               aspect='auto')
    plt.colorbar()
    # centers = model.p_sigma._modules['0'].points.detach().numpy()
    # plt.scatter(centers[:, 0], centers[:, 1], s=1.5, c='y')

    samples = []
    for _, data in enumerate(data_loader):
        x = data
        _, _, z, _, _ = model(x.to(device))
        samples.append(z)
    samples = torch.stack(samples, dim=0).view(-1, 2).detach().cpu().numpy()
    plt.scatter(samples[:, 0], samples[:, 1], s=0.6, c='y')

    # means = model.pr_means.detach().cpu().numpy()
    # plt.scatter(means[:, 0], means[:, 1], s=1.5, c='cyan')
    # plt.savefig("./circle_means_centroids.png")
    # plt.close()
    plt.show()

mu_epochs = 4
sigma_epochs = 150
prior_epochs = 10
b_sz = 100
log_invl = 100
RBF_beta = 50
init_beta = 1

embedding = nn.Sequential(nn.Linear(2, 100), nn.ReLU())
model = RBFVAE(100, 2, 256, 15, [64, 64], [64, 64], nnj.Tanh, None, rbf_beta=init_beta, rec_b=1e-9, device="cpu", trainable=True)
optimizer_mu = torch.optim.Adam(chain(model.encoder.parameters(), 
                                model.q_mu.parameters(), 
                                model.q_t.parameters(),
                                model.p_mu.parameters(), model.p_sigma.parameters()), 
                                lr=1e-3)
optimizer_sigma = torch.optim.Adam(model.p_sigma.parameters(), lr=1e-3)
optimizer_prior = torch.optim.Adam(model.parameters(), lr=1e-4)

train_loader, test_loader = get_circle_loaders("./vae_example/data/", b_sz)

for epoch in range(1, mu_epochs + 1):
    train_rbfvae(epoch, train_loader, b_sz, model, optimizer_mu, log_invl, "cpu")
    # show_prior_means(model, test_loader, "cpu")

model.switch = False
# model._update_RBF_centers(train_loader.dataset.data, RBF_beta)
for epoch in range(1, sigma_epochs + 1):
    train_rbfvae(epoch, train_loader, b_sz, model, optimizer_sigma, log_invl, "cpu")
show_prior_means(model, test_loader, "cpu")
    # if epoch % 2 == 0:
    #     show_prior_means(model, test_loader, "cpu")
_, samples = test_rbfvae(epoch, test_loader, b_sz, model, 1, "cpu")
visualize_variance(model, 100, samples.view(-1, 2), "./variance_RBF.pdf", "cpu")

# model._initialize_prior_means(train_loader.dataset.data)
model._mean_warmup = False
show_prior_means(model, test_loader, "cpu")
for epoch in range(1, prior_epochs + 1):
    train_rbfvae(epoch, train_loader, b_sz, model, optimizer_prior, log_invl, "cpu")

    if epoch % 2 == 0:
        _, samples = test_rbfvae(epoch, test_loader, b_sz, model, 1, "cpu")
        visualize_variance(model, 100, samples, "./variance_RBF.pdf", "cpu")
