import argparse

from experiment import Experiment


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="../data/", type=str, help="root directory all the data will be stored in")
parser.add_argument("--save_dir", default="../saved_models/", type=str, help="root directory model checkpoints will be saved in")
parser.add_argument("--res_dir", default="../results/", type=str, help="root directory results will be saved in")
parser.add_argument("--dataset", default="mnist", help="mnist | fmnist | kmnist")
parser.add_argument("--enc_layers", nargs="+", type=int, help="encoder layers", default="100")
parser.add_argument("--dec_layers", nargs="+", type=int, help="decoder layers", default="100")
parser.add_argument("--model", default="RVAE", help="RVAE | VAE")
parser.add_argument("--batch_size", default=100, type=int, help="batch size")
parser.add_argument("--warmup_learning_rate", default=1e-3, type=float, help="p_mu learning rate")
parser.add_argument("--sigma_learning_rate", default=1e-3, type=float, help="p_sigma learning rate")
parser.add_argument("--mu_epochs", default=100, type=int, help="number of training epochs (decoder mu)")
parser.add_argument("--sigma_epochs", default=100, type=int, help="number of training epochs (decoder sigma)")
parser.add_argument("--device", default="cpu", type=str, help="cuda | cpu")
parser.add_argument("--seed", default=None, help="random seed")
parser.add_argument("--log_invl", default=100, type=int, help="the interval in which training stats will be reported")
parser.add_argument("--save_invl", default=25, type=int, help="the interval in which model weights will be saved")
parser.add_argument("--latent_dim", default=2, type=int, help="dimensionality of latent space")
parser.add_argument("--num_centers", default=64, type=int, help="number of centers for the RBF regularization in the decoder sigma net")
parser.add_argument("--rbf_beta", default=0.01, type=float, help="rbf layer beta parameter")
parser.add_argument("--rec_b", default=1e-9, type=float)
parser.add_argument("--num_components", default=128, type=int, help="number of components for the prior")
parser.add_argument("--ckpt_path", default=None, type=str)
args = parser.parse_args()

experiment = Experiment(args)

if args.ckpt_path is None:
    experiment.train()
    experiment.eval()
else:
    experiment.eval(args.ckpt_path)
