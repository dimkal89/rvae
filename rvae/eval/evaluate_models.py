import torch
from vae import RBFVAE, VAE
from losses import riemannian_loss, euclidean_loss
from geoml import nnj
from data_utils import get_mnist_loaders
from utils import load_model
import numpy as np
from itertools import chain

def test_rbfvae(epoch, test_loader, b_sz, model, iw_samples, device, viz=False, viz_geodesics=False):
    model.eval()
    test_loss = 0
    test_kld = 0
    samples = []
    n_batches = len(test_loader.dataset.data)//b_sz
    iw_bound = True if iw_samples > 1 else False

    with torch.no_grad():
        for _, data in enumerate(test_loader):
            if isinstance(data, list):
                data = data[0].view(-1, 784).to(device)
            else:
                data = data.view(-1, 784).to(device)
            
            if iw_bound:
                log_w = torch.zeros((iw_samples, b_sz), device=device, dtype=torch.float)
                for j in range(iw_samples):
                    p_mu, p_sigma, z, q_mu, q_t = model(data)
                    k = 1/model.n_components * torch.ones(model.n_components)
                    loss = riemannian_loss(data, p_mu, p_sigma, z, q_mu, q_t, 
                                            model.pr_means, model.pr_t, k, model, 1, iw_bound)
                    log_w[j, :] = loss
                
                # log sum exp
                w_max = log_w.max(dim=0)[0]
                w = torch.log(torch.sum(torch.exp(log_w - w_max), dim=0) + 1e-5) + w_max
                batch_loss = -torch.mean(w)

                test_loss += batch_loss
            else:
                p_mu, p_sigma, z, q_mu, q_t = model(data)
                k = 1/model.n_components * torch.ones(model.n_components)
                loss = riemannian_loss(data, p_mu, p_sigma, z, q_mu, q_t, 
                                        model.pr_means, model.pr_t, k, model, 1, iw_bound)
                test_loss += loss[0]    # ELBO
                test_kld += loss[2]     # KL div
                samples.append(z)

    test_loss /= n_batches
    test_kld /= n_batches

    if iw_bound:
        print("Test set IW-ELBO: {:.3f}".format(test_loss))
    else:
        print("Test set loss: {:.3f}, pxz density: {:.3f}, KL div: {:.3f} \n".format(test_loss.item(), (test_loss-test_kld).item(), test_kld.item()))
    
    return test_loss, test_kld

def test_vae(epoch, test_loader, b_sz, model, iw_samples, log_invl, device):
    model.eval()
    avg_loss = 0
    test_kld = 0
    test_ll = 0
    n_batches = len(test_loader.dataset.data)//b_sz
    iw_bound = True if iw_samples > 1 else False

    with torch.no_grad():
        for _, data in enumerate(test_loader):
            if isinstance(data, list):
                data = data[0].view(-1, 784).to(device)
            else:
                data = data.view(-1, 784).to(device)
            
            # VampPrior
            if model.n_components > 1:
                p_mu, p_var, z, q_mu, q_var, pr_mu, pr_var = model(data)
                k = 1/model.n_components * torch.ones(model.n_components)
                loss = euclidean_loss(data, p_mu, p_var, z, q_mu, q_var, 
                                                pr_mu, pr_var, k, 1, iw_bound)
            # Regular VAE
            else:
                if iw_bound:
                    log_w = torch.zeros((iw_samples, b_sz), device=device, dtype=torch.float)
                    for j in range(iw_samples):
                        p_mu, p_var, z, q_mu, q_var, _, _ = model(data)
                        k = 1/model.n_components * torch.ones(model.n_components)
                        loss = euclidean_loss(data, p_mu, p_var, z, q_mu, q_var, 
                                            torch.zeros_like(q_mu), torch.ones_like(q_var), k, 1, iw_samples, iw_bound)
                        log_w[j, :] = loss
                
                    # log sum exp
                    w_max = log_w.max(dim=0)[0]
                    w = torch.log(torch.sum(torch.exp(log_w - w_max), dim=0) + 1e-5) + w_max
                    batch_loss = -torch.mean(w)

                    avg_loss += batch_loss
                else:
                    p_mu, p_var, z, q_mu, q_var, _, _ = model(data)
                    k = 1/model.n_components * torch.ones(model.n_components)
                    loss = euclidean_loss(data, p_mu, p_var, z, q_mu, q_var, 
                                        torch.zeros_like(q_mu), torch.ones_like(q_var), k, 1, iw_samples, iw_bound)
                    
                    avg_loss += loss[0]     # ELBO
                    test_ll += loss[1]
                    test_kld += loss[2]     # KL div

        avg_loss /= n_batches
        test_kld /= n_batches
        test_ll /= n_batches

        if iw_bound:
            print("Test set IW-ELBO: {:.3f}".format(avg_loss))
        else:
            print("Test set loss: {:.3f}, pxz density: {:.3f}, KL div: {:.3f} \n".format(avg_loss.item(), (avg_loss-test_kld).item(), test_kld.item()))

        return avg_loss, test_ll, test_kld

model_type = "VAE"
_, test_loader = get_mnist_loaders("./vae_example/data/", 100, True)

vae_2d = "./path/to/2D_VAE.ckpt"
vae_5d = "./path/to/5D_VAE.ckpt"
vae_10d = "./path/to/10D_VAE.ckpt"
rvae_2d = "./path/to/2D_RVAE.ckpt"
rvae_5d = "./path/to/5D_RVAE.ckpt"
rvae_10d = "./path/to/10D_RVAE.ckpt"

paths_vae = [vae_2d, vae_5d, vae_10d]
paths_rvae = [rvae_2d, rvae_5d, rvae_10d]

if model_type == "RVAE":
    for path in paths_rvae:
        if path is rvae_2d:
            print("R-VAE 2D \n")
            dim = 2
        elif path is rvae_5d:
            print("R-VAE 5D \n")
            dim = 5
        else:
            print("R-VAE 10D \n")
            dim = 10

        model = RBFVAE(784, dim, 100, 350, 1, [300, 300], [300, 300], nnj.ELU, nnj.Sigmoid, 0.01, 1e-9, "cpu", True) # 0.08 is a good value for beta
        model._mean_warmup = False
        model.switch = False
        optimizer_var = torch.optim.Adam(chain([model.pr_means, model.pr_t], model.p_sigma.parameters()), lr=1e-3)
        load_model(path, model, optimizer_var)

        elbos = np.zeros([10])
        kls = np.zeros([10]) 
        iw_elbo = np.zeros([10])
        for i in range(10):
            loss, kld = test_rbfvae(0, test_loader, 100, model, 1, "cpu", False)
            iw_loss, _ = test_rbfvae(0, test_loader, 100, model, 250, "cpu", False)
            elbos[i] = loss.detach().numpy()
            kls[i] = kld.detach.numpy()
            iw_elbo[i] = iw_elbo.detach().numpy()
        print("ELBO STATS")
        print("mean:", elbos.mean())
        print("std dev:", elbos.std())
        print("IW BOUND STATS")
        print("mean:", iw_elbo.mean())
        print("std dev:", iw_elbo.std())
else:
    for path in paths_vae:
        if path is vae_2d:
            print("VAE 2D \n")
            dim = 2
        elif path is vae_5d:
            print("VAE 5D \n")
            dim = 5
        else:
            print("VAE 10D \n")
            dim = 10

        model = VAE(784, dim, 100, 350, 1, [300, 300], [300, 300], nnj.ELU, nnj.Sigmoid, 0.01, 1e-9, "cpu")
        optimizer_var = torch.optim.Adam(model.p_sigma.parameters(), lr=1e-3)
        load_model(path, model, optimizer_var)

        elbos = np.zeros([10])
        kls = np.zeros([10]) 
        lls = np.zeros([10])
        for i in range(10):
            elbo, ll, kl = test_vae(0, test_loader, 100, model, 1, 100, "cpu")
            elbos[i] = elbo.detach().numpy()
            lls[i] = ll.detach().numpy()
            kls[i] = kl.detach().numpy()
        print("ELBO STATS")
        print("mean:", elbos.mean())
        print("std dev:", elbos.std())
        print("LL STATS")
        print("mean:", lls.mean())
        print("std dev:", lls.std())
        print("KL STATS")
        print("mean:", kls.mean())
        print("std dev:", kls.std())