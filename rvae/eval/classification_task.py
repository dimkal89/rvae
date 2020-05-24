from sklearn import neighbors, metrics
from sklearn.neural_network import MLPClassifier
from vae import RBFVAE, VAE
from data_utils import get_mnist_loaders
import numpy as np
from geoml import nnj
from torch import nn
import torch
from utils import load_model
from itertools import chain

train_loader, test_loader = get_mnist_loaders("./data/", 100, True)
# set saved VAE checkpoint path
vae_path = "./path/to/VAE.ckpt"
# set saved RVAE checkpoint path
rvae_path = "./path/to/RVAE.ckpt"
# set latent dimensions 
latent_dim = 10

# load appropriate model
model = VAE(784, latent_dim, 350, 1, [300, 300], [300, 300], nnj.ELU, nnj.Sigmoid, 0.01, 1e-9, "cpu")
optimizer_var = torch.optim.Adam(model.p_sigma.parameters(), lr=1e-3)
load_model(vae_path, model, optimizer_var)

# model = RBFVAE(784, latent_dim, 400, 1, [300, 300], [300, 300], nnj.ELU, nnj.Sigmoid, 0.1, 1e-9, "cpu", True)
# model._mean_warmup = False
# model.switch = False
# optimizer_var = torch.optim.Adam(chain(model.p_sigma.parameters(), [model.pr_means, model.pr_t]), lr=1e-4)
# load_model(rvae_path, model, optimizer_var)

z_train = []
z_test = []
y_train = []
y_test = []

with torch.no_grad():
    # get training embeddings
    for _, data in enumerate(train_loader):
        x, y = data
        if isinstance(model, VAE):
            z, _, _, _, _ = model.encode(x.view(-1, 784), 1)
        else: 
            z, _, _ = model.encode(x.view(-1, 784), 1)
        z_train.append(z)
        y_train.append(y)
    
    z_train = torch.stack(z_train, dim=0).view(-1, latent_dim).detach().numpy()
    y_train = torch.stack(y_train).view(-1).detach().numpy()

    # get test embeddings
    for _, data in enumerate(test_loader):
        x, y = data
        if isinstance(model, VAE):
            z, _, _, _, _ = model.encode(x.view(-1, 784), 1)
        else: 
            z, _, _ = model.encode(x.view(-1, 784), 1)
        z_test.append(z)
        y_test.append(y)

    z_test = torch.stack(z_test, dim=0).view(-1, latent_dim).detach().numpy()
    y_test = torch.stack(y_test).view(-1).detach().numpy()

print("classification phase")

digit_f1 = np.zeros([5, 10])
avg_f1 = np.zeros([5])

for i in range(5):
    clf = MLPClassifier(batch_size=64, max_iter=100)
    clf.fit(z_train, y_train)
    y_pred = clf.predict(z_test)
    f1 = metrics.f1_score(y_test, y_pred, average=None)
    acc = metrics.accuracy_score(y_test, y_pred)

    digit_f1[i] = f1
    avg_f1[i] = f1.mean()

print(digit_f1.mean(axis=0))
print(avg_f1.mean())
print(avg_f1.std())
