import torch

from torch.distributions import Normal, kl_divergence
from ..misc import log_bm_krn, log_multibm_krn, log_gauss_mix


def elbo_rvae(data, p_mu, p_sigma, z, q_mu, q_t, model, beta):
    if model._mean_warmup:
        return -Normal(p_mu, p_sigma).log_prob(data).sum(-1).mean(), torch.zeros(1), torch.zeros(1)
    else:
        pr_mu, pr_t = model.pr_means, model.pr_t

        log_pxz = Normal(p_mu, p_sigma).log_prob(data).sum(-1)
        log_qzx = log_bm_krn(z, q_mu, q_t, model)
        log_pz = log_bm_krn(z, pr_mu.expand_as(z), pr_t, model)

        KL = log_qzx - log_pz

        return (-log_pxz + beta * KL.abs()).mean(), -log_pxz.mean(), KL.mean()


def elbo_vae(data, p_mu, p_var, z, q_mu, q_var, pr_mu, pr_var, beta, vampprior=False):
    log_pxz = Normal(p_mu, p_var.sqrt()).log_prob(data).sum(-1)
    log_qzx = Normal(q_mu, q_var.sqrt()).log_prob(z).sum(-1)

    if vampprior:
        log_pz = log_gauss_mix(z, pr_mu, pr_var)
    else:
        log_pz = Normal(pr_mu, pr_var).log_prob(z)

    KL = log_qzx - log_pz

    return (-log_pxz + beta * KL).mean(), -log_pxz.mean(), KL.mean()
