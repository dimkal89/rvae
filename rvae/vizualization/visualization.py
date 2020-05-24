import torch
import numpy as np
from matplotlib import pyplot as plt
from rvae.misc import connecting_geodesic, linear_interpolation
from torchvision.utils import save_image


def plot_variance(model, var_dim, samples, savepath, device, log_scale=True):
    # set up the grid
    x = np.arange(-30, 30, 0.5)
    y = np.arange(-30, 30, 0.5)
    xx, yy = np.meshgrid(x, y)
    coords = np.stack((xx.flatten(), yy.flatten()))
    coords = coords.transpose().astype(np.float32)
    coords = torch.from_numpy(coords)

    # evaluate variance function on points on the grid
    sigma = list()
    n_batches = len(coords)//100
    for b in range(n_batches):
        batch_coords = coords[b * 100: (b + 1) * 100].to(device)
        batch_sigma = model.p_sigma(batch_coords, False)
        sigma.append(batch_sigma)
    sigma = torch.stack(sigma)
    sigma = sigma.view(coords.size(0), var_dim)

    # log the variance for clearer visual results
    sigma = sigma.detach().cpu().numpy()
    if log_scale:
        sigma = np.sum(np.log(sigma), axis=1)
    else:
        # NOTE: check that all of this is correct, it's probably not
        sigma = np.sum(np.log(sigma), axis=1)
        sigma = np.exp(sigma)
        sigma = np.clip(sigma, a_min=1e-5, a_max=1e6)
        # var = np.sqrt(var).prod(1)
    
    # plot
    num_grid = xx.shape[0]
    sigma = sigma.reshape(num_grid, num_grid)
    plt.imshow(sigma,
               interpolation='gaussian',
               origin='lower',
               extent=(-30, 30,
                       -30, 30),
               cmap=plt.cm.seismic,
               aspect='auto')
    if log_scale:
        plt.title("Log std dev")
    else:
        plt.title("Std dev")
    plt.colorbar()

    samples = samples.detach().cpu().numpy()
    plt.scatter(samples[:, 0], samples[:, 1], s=0.9, c='g')
    plt.savefig(savepath)
    plt.close()


def plot_latent_space(model, data_loader, save_dir, device, log_scale=True):
    # set up the grid
    side = 130
    step = 1
    x = np.arange(-side, side, step)
    y = np.arange(-side, side, step)
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
               extent=(-side, side,
                       -side, side),
               cmap=plt.cm.RdGy,
               aspect='auto')
    plt.colorbar()

    samples = []
    labels = []
    for _, data in enumerate(data_loader):
        x, y = data[0], data[1]
        _, _, z, _, _ = model(x.view(-1, 784).to(device))
        samples.append(z)
        labels.append(y)
    
    samples = torch.stack(samples, dim=0).view(-1, 2).detach().cpu().numpy()
    labels = torch.stack(labels).view(-1).detach().cpu().numpy()
    plt.scatter(samples[:, 0], samples[:, 1], s=0.7, c=labels)
    plt.savefig(save_dir)
    plt.close()


def plot_brownian_motion(model, bm_samples, save_dir, device, log_scale):
    sq = 120
    step = 1
    # set up the grid
    x = np.arange(-sq, sq, step)
    y = np.arange(-sq, sq, step)
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
               extent=(-sq, sq,
                       -sq, sq),
               cmap=plt.cm.RdGy,
               aspect='auto')
    
    # plot brownian motion
    bm_samples = bm_samples.detach_().cpu().numpy()
    plt.plot(bm_samples[:, 0], bm_samples[:, 1], color='k')

    # plt.title("Latent prior sampling")
    plt.colorbar()
    plt.savefig(save_dir)
    plt.close()


def show_reconstructions(data_loader, model, save_dir, device):
    x = next(iter(data_loader))
    if isinstance(x, list):
        x = x[0]
    out_tup = model(x.view(-1, 784).to(device))
    p_mu = out_tup[0]
    save_image(p_mu.view(-1, 1, 28, 28).cpu(), save_dir + "reconstruction.jpg", nrow=10)


def plot_geodesics(model, data_loader, save_dir, device, log_scale=True):
    samples = []
    labels = []
    for _, data in enumerate(data_loader):
        x, y = data
        z, _, _ = model.encode(x.view(-1, 784), 1)
        samples.append(z)
        labels.append(y)

    samples = torch.stack(samples, dim=0).view(-1, 2)
    labels = torch.stack(labels).view(-1)
    idcs = np.random.choice(samples.shape[0], 2)

    model.p_sigma._modules['0'].beta = 0.07
    
    c, _ = connecting_geodesic(model, samples[idcs[0]].unsqueeze(0), samples[idcs[1]].unsqueeze(0), n_nodes=32, eval_grid=64, max_iter=500, l_rate=1e-3)
    c_pts = c(torch.arange(start=0, end=1, step=0.05))
    c_pts = c_pts.detach().cpu().numpy()

    model.p_sigma._modules['0'].beta = 0.01

    side = 110
    step = 1

    x = np.arange(-side, side, step)
    y = np.arange(-side, side, step)
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
               extent=(-side, side,
                       -side, side),
               cmap=plt.cm.RdGy,
               aspect='auto')
    plt.colorbar()

    # latent space scatter plot
    samples = samples.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    plt.scatter(samples[:, 0], samples[:, 1], s=0.9, c=labels)

    p_mu, _ = model.decode(torch.from_numpy(c_pts).float(), False)
    save_image(p_mu.view(-1, 1, 28, 28).detach().cpu(), save_dir + "geodesic_img.jpeg", nrow=20)

    lin_pts = linear_interpolation(torch.from_numpy(c_pts[0]), torch.from_numpy(c_pts[-1]), n_points=20)
    p_mu, _ = model.decode(lin_pts, False)
    save_image(p_mu.view(-1, 1, 28, 28).detach().cpu(), save_dir + "linear_img.jpeg", nrow=20)

    plt.plot(c_pts[:, 0], c_pts[:, 1], c='k')
    plt.plot(lin_pts[:, 0], lin_pts[:, 1], c='r')
    plt.show()

    return c_pts, lin_pts