import sys
import torch
import numpy as np

from .losses import elbo_rvae, elbo_vae


def train_rvae(epoch, train_loader, batch_size, model, optimizer, log_invl, device):
    model.train()
    train_loss = 0.
    train_kld = 0.
    n_batches = len(train_loader.dataset)//batch_size

    for i, (data, labels) in enumerate(train_loader):
        beta = 1
        optimizer.zero_grad()
        data = data.view(-1, 784).to(device)

        p_mu, p_sigma, z, q_mu, q_t = model(data)
        model.dummy_pmu.load_state_dict(model.p_mu.state_dict())

        if model.switch:
            p_sigma = torch.ones(1).to(device)
            # p_sigma = torch.tensor([1e-4]).to(device)
        
        loss, log_pxz, kld = elbo_rvae(data, p_mu, p_sigma, z, q_mu, q_t.squeeze(), model, beta)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if i % log_invl == 0:
            print("Epoch: {}, batch: {}, loss: {:.3f}, KL: {:.3f}".format(
                epoch, i, loss.item(), kld.item()
            ))
    
    train_loss /= n_batches
    train_kld /= n_batches

    return train_loss, train_kld


def test_rvae(test_loader, batch_size, model, device):
    model.eval()
    test_loss = 0
    test_kld = 0
    n_batches = len(test_loader.dataset)//batch_size

    with torch.no_grad():
        for _, (data, labels) in enumerate(test_loader):
            data = data.view(-1, 784).to(device)
            
            p_mu, p_sigma, z, q_mu, q_t = model(data)
            loss = elbo_rvae(data, p_mu, p_sigma, z, q_mu, q_t, model, 1.)
            test_loss += loss[0]    # ELBO
            test_kld += loss[2]     # KL div

    test_loss /= n_batches
    test_kld /= n_batches
    
    return test_loss, test_kld


def train_vae(epoch, train_loader, batch_size, model, optimizer, log_invl, device):
    model.train()
    train_loss = 0.
    train_kld = 0.
    n_batches = len(train_loader.dataset)//batch_size

    for i, (data, labels) in enumerate(train_loader):
        beta = min(epoch/200, 1)
        optimizer.zero_grad()
        data = data.view(-1, 784).to(device)
        
        p_mu, p_var, z, q_mu, q_var, pr_mu, pr_var = model(data)
        if model.switch:
            p_var = torch.ones(1).to(device)
            # p_var = torch.tensor([1e-4]).to(device)
        vampprior = True if model.num_components > 1 else False
        loss, log_pxz, kld = elbo_vae(data, p_mu, p_var, z, q_mu, q_var, pr_mu, pr_var, beta, vampprior)
        loss.backward()
        train_loss += loss.item()
        train_kld += kld
        optimizer.step()

        if i % log_invl == 0:
            print("Epoch: {}, batch: {}, loss: {:.3f}, KL: {:.3f}".format(
                epoch, i, loss.item(), kld.item()
            ))
    
    train_loss /= n_batches
    train_kld /= n_batches

    return train_loss, train_kld 


def test_vae(test_loader, b_sz, model, device):
    model.eval()
    test_loss = 0
    test_kld = 0
    n_batches = len(test_loader.dataset)//b_sz

    with torch.no_grad():
        for _, (data, labels) in enumerate(test_loader):
            data = data.view(-1, 784).to(device)
            
            p_mu, p_var, z, q_mu, q_var, pr_mu, pr_var = model(data)
            vampprior = True if model.num_components > 1 else False
            loss = elbo_vae(data, p_mu, p_var, z, q_mu, q_var, pr_mu, pr_var, 1, vampprior)
            test_loss += loss[0]     # ELBO
            test_kld += loss[2]     # KL div

        test_loss /= n_batches
        test_kld /= n_batches

        return test_loss, test_kld
