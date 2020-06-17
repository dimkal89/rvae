import sys
import torch
import numpy as np
from math import pi
from torch.distributions import Normal, Categorical
from .geoml.curve import CubicSpline
import math


class DistSqKL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, net, p0, p1):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        b_sz = p0.shape[0]
        with torch.no_grad():
            with torch.enable_grad():
                crv, energy = connecting_geodesic(net, p0, p1, max_iter=6, n_nodes=5, eval_grid=16, l_rate=1e-3)
                lm0 = crv.deriv(torch.zeros(1, device=device)).view(b_sz, -1)
                lm1 = crv.deriv(torch.ones(1, device=device)).view(b_sz, -1)
                ctx.save_for_backward(lm0, lm1)
        net.p_sigma.zero_grad()
        net.dummy_pmu.zero_grad()
        return energy
    
    @staticmethod
    def backward(ctx, grad_output):
        if grad_output.dim() == 1:
            grad_output.unsqueeze_(1)
        lm0, lm1 = ctx.saved_tensors
        return None, 2 * grad_output * lm0, 2 * grad_output * lm1


def curve_energy(c, model, eval_pts):
    """Computes curve energy (in ambient/embedding space) with
    Riemann sums.
    
    params:
        c:              geoml.curve.CubicSpline object - the curve in latent space
        model:          nn.Module object - the VAE containing the decoder mu/sigma
                        functions
        eval_pts:       int - the number of (ordered) discrete points representing 
                        the curve
    """
    c = c.view(-1, model.latent_dim)
    mu = model.dummy_pmu(c, False)
    mu = mu.view(-1, eval_pts, model.in_dim)
    delta_mu = (mu[:, 1:, :] - mu[:, :-1, :])
    sigma = model.p_sigma(c, False)
    sigma = sigma.view(-1, eval_pts, model.in_dim)
    delta_sig = (sigma[:, :-1, :] - sigma[:, 1:, :])

    d_mu = delta_mu.pow(2).sum(1)
    d_sig = delta_sig.pow(2).sum(1)
    
    return 0.5 * torch.sum(d_mu + d_sig, dim=-1)


def connecting_geodesic(net, p0, p1, optim=torch.optim.SGD, max_iter=25, n_nodes=16, eval_grid=5, l_rate=1e-3):
    """Computes the logmap of the geodesic with endpoints 
    p0, p1 \in M by minimizing the curve energy.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # The line below is written assuming p1 is the mean
    curve = CubicSpline(p0, p1, num_nodes=n_nodes, device=device)
    
    # the following code is lifted from 
    # geoml.geodesics.geodesic_minimizing_energy()
    alpha = torch.linspace(0, 1, eval_grid, device=device).reshape((-1, 1))
    if optim == torch.optim.SGD:
        opt = optim([curve.parameters], momentum=0.99, lr=l_rate, nesterov=True)
    else:
        opt = optim([curve.parameters], lr=l_rate)

    if net._mean_warmup:
        curve_energies = curve_energy(curve(alpha), net, eval_grid)
    else:
        for _ in range(max_iter):
            opt.zero_grad()
            curve_energies = curve_energy(curve(alpha), net, eval_grid)
            loss = curve_energies.sum()
            loss.backward()
            opt.step()
            if torch.max(torch.abs(curve.parameters.grad)) < 1e-4:
                break
    
    return curve, curve_energies.detach_()


def log_bm_krn(x, y, t, model):
    """Log pdf of a Brownian motion (BM) transition kernel.
    
    params:
        x:      torch.tensor object - a point on the manifold
        y:      torch.tensor object - a point on the manifold,
                typically interpreted as a "mean".
        t:      float - the time for which the BM occur
        model:  nn.Module object - the model containing the embedding 
                mapping to the ambient space
    """
    d = x.size(1)
    t = t.squeeze()
    _, logdet_x = model.metric(x).slogdet()
    _, logdet_y = model.metric(y).slogdet()
    log_H = (logdet_x - logdet_y)/2
    l_sq = DistSqKL.apply(model, x, y)

    return -d/2 * torch.log(2 * pi * t) + log_H - l_sq/(2 * t)


def brownian_motion_sample(num_steps, num_samples, dim, t, init_point, model):
    """Returns the points of a discretized Brownian motion (BM)
    on a manifold (a.k.a. latent space).

    params:
        num_steps:      int - the number of time steps for which 
                        the BM will run
        num_samples:    int - the number of samples that will be
                        returned
        dim:            int - the dimensionality of the manifold/
                        latent space
        t:              float - the time for which the BM will run
        init_point:     torch.Tensor - the initial point of the BM
        model:          torch.nn.Module - the model containing the
                        embedding
    """ 
    if init_point is None:
        init_point = torch.zeros(num_samples, dim)
    samples = [init_point.squeeze()]

    if num_samples > 1:
        samples[0] = samples[0].expand(num_samples, dim)
    
    for _ in range(num_steps - 1):
        g = model.metric(samples[-1])
        cov_mat = t/num_steps * g.squeeze().inverse()
        mvn = torch.distributions.multivariate_normal.MultivariateNormal(samples[-1], covariance_matrix=cov_mat)
        samples.append(mvn.sample().squeeze())
    
    return torch.cat(samples).view(num_steps, num_samples, dim).squeeze()


def log_gauss_mix(x, mu, var):
    # number of components
    K = mu.shape[0]

    x_xp = x.unsqueeze(1)
    mu_xp = mu.unsqueeze(0)
    var_xp = var.unsqueeze(0)

    a = log_Normal_diag(x_xp, mu_xp, torch.log(var_xp + 1e-5), dim=2) - math.log(K)
    a_max, _ = torch.max(a, 1)  # MB x 1

    # calculate log-sum-exp
    log_mix = a_max + torch.log(torch.sum(torch.exp(a - a_max.unsqueeze(1)), 1))

    return log_mix


def log_Normal_diag(x, mean, log_var, average=False, dim=None):
    log_normal = -0.5 * ( log_var + torch.pow( x - mean, 2 ) / torch.exp( log_var ) )
    if average:
        return torch.mean( log_normal, dim )
    else:
        return torch.sum( log_normal, dim )


def linear_interpolation(p0, p1, n_points):
    dim = p0.shape[-1]
    c_pts = torch.zeros([n_points, dim])
    c_pts[0] = p0
    c_pts[-1] = p1 
    for i in range(1, (n_points + 1) - 2):
        c_pts[i] = c_pts[i - 1] + 1/n_points * (p1 - p0)
    
    return c_pts
