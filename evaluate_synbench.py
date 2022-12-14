import os
import argparse
import torch
import numpy as np
import seaborn as sns
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import timm
import torch.nn as nn
from torch.utils.data import Dataset
import cvxpy as cp
from torchvision import transforms as pth_transforms
from collections import defaultdict
from utils import yaml_config_hook


def func_bound_correct(a):
    return 1/(a*scipy.stats.norm.ppf(a)) * np.sqrt(1/(2*np.pi)) * np.exp(-scipy.stats.norm.ppf(a)**2/2) + 1


class Gaussian_Dataset(Dataset):
    def __init__(self, mu, Sigma, cls_prior, num_realization, transform=None):
        # generating symmetric mean gaussians with the same covariance matrix and some class prior
        self.mu = mu
        self.Sigma = Sigma
        self.cls_prior = cls_prior
        self.labels = torch.bernoulli(cls_prior*torch.ones(num_realization))

        self.samples = []
        self.transform = transform

        ## since the covariance matrix is a diagonal matrix, we generate the gaussians individually
        for i in range(self.labels.shape[0]):
            if self.labels[i]== 1:
                self.samples.append(torch.normal(mean=self.mu+0.5, std=torch.sqrt(torch.diag(self.Sigma))).unsqueeze(0))
            else:
                self.samples.append(torch.normal(mean=-self.mu+0.5, std=torch.sqrt(torch.diag(self.Sigma))).unsqueeze(0)) 


        self.samples = torch.cat(self.samples, out=torch.Tensor(len(self.samples),self.samples[0].shape[1]))

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.transform:
            sample = self.transform(sample.view(3,224,224))
        return sample, int(self.labels[idx])


def inference(loader, model):
    feature_vector = []
    labels_vector = []
    for _, (x, y) in enumerate(loader):
        x = x.to(args.device).view(y.shape[0],3,224,224)

        with torch.no_grad():
            h = model(x)
        h = h.detach()

        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    return feature_vector, labels_vector


def get_bound_rep(loader, mu_1, mu_2, Sigma, epss=[0,0.2]):
    bound = []
    accuracy = []

    F, D, _ = torch.svd(Sigma, some=True)
    F1 = F[:,D>1e-6].float()
    D1 = D[D>1e-6].float()
    D1_inv = torch.diag(D1**(-1))
    
    mu = torch.matmul(F1.t(), (mu_1 - mu_2)/2)
    mu_mu = ((mu_1 + mu_2)/2).to(args.device)
    F1 = F1.to(args.device)
    
    for eps in epss:
        print(eps)
        mu = mu.cpu()
        z_S, _ = qcqp_solve(mu, D1_inv, float(eps) * torch.norm(mu, p=2))

        w_0 = torch.matmul(D1_inv,mu-torch.tensor(z_S).float())
        w_0 = w_0.to(args.device)
        W_0 = torch.matmul(F1, w_0) 
        
        mu = mu.to(args.device)

        bound_epoch = 0
        accuracy_epoch = 0
        for _, (x, y) in enumerate(loader):

            x = x.to(args.device)
            y = y.to(args.device)

            with torch.no_grad():
                xbar = x.view(-1,mu_mu.shape[0]) - mu_mu

                margin = torch.matmul(xbar, W_0)
                bb = torch.abs(margin)/torch.abs(torch.matmul(mu.t(),w_0))
                bound_epoch += (bb[(margin > 0) * (y==1) + (margin < 0) * (y==0)]).mean()
                accuracy_epoch += ((margin > 0) * (y==1) + (margin < 0) * (y==0)).sum().item()/ y.size(0)

        bound.append((bound_epoch/len(loader)).detach().cpu().numpy())
        accuracy.append((accuracy_epoch/len(loader)))
        
    return bound, accuracy


def qcqp_solve(mu, D, epsilon):

    mu = mu.numpy()
    D = D.numpy()
    epsilon = epsilon.numpy()

    n = mu.shape[0]

    # Define and solve the CVXPY problem.
    z = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.quad_form(mu-z, D)),
                    #  [cp.SOC(epsilon, z)])
                    [cp.pnorm(z) <= epsilon])
    prob.solve(solver='SCS')

    return z.value, prob.value



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SynBench")
    config_ = yaml_config_hook("./config.yaml")
    for k, v in config_.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    realization = args.num_realization
    epsilons = list(args.epsilon.split())
    print(epsilons)

    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if args.device == 'cuda':
            torch.cuda.manual_seed(args.seed)

    save_dir = './test_' + args.fm_name + '/results' 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    dct_bounds = defaultdict(list)
    dct_accuracy = defaultdict(list)
    dct_bounds['raw'] = []
    dct_accuracy['raw'] = []
    for eps in epsilons:
        dct_bounds['rep_{}'.format(eps)] = []
        dct_accuracy['rep_{}'.format(eps)] = []


    if args.calculate:
        encoder = timm.create_model(args.fm_name, pretrained=True, num_classes=0).to(args.device)
        encoder.eval()
        n_features = encoder.num_features        

        mu_scales = np.arange(0.1, 5.0, 0.1)
        mu = torch.ones(3*224*224)/np.sqrt(3*224*224)
        Sigma = torch.eye(3*224*224)

        train_transform = pth_transforms.Compose([
            pth_transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)), 
        ])
        val_transform = pth_transforms.Compose([
            pth_transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
        ])
        for scale in mu_scales:
            # synthetic data generation
            train_dataset = Gaussian_Dataset(mu*scale, Sigma, args.cls_prior, realization, transform=train_transform) 
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.logistic_batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=args.workers,
            )
            test_dataset = Gaussian_Dataset(mu*scale, Sigma, args.cls_prior, realization, transform=val_transform) 
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=args.logistic_batch_size,
                shuffle=False,
                drop_last=True,
                num_workers=args.workers,
            )
            # compute the reference accuracy vs bound in the raw data space
            accuracy = scipy.stats.norm.cdf(torch.norm(mu*scale, p=2))
            dct_bounds['raw'].append(func_bound_correct(accuracy))
            dct_accuracy['raw'].append(accuracy)
            
            # find the bayes classifiers in the representatin space
            train_X, train_y = inference(train_loader, encoder)
            selected_data = train_X[train_y==1]
            mu_1 = np.mean(selected_data, axis=0)
            cov = np.cov(selected_data, rowvar=False) * (selected_data.shape[0]-1)
            selected_data = train_X[train_y==0]
            mu_2 = np.mean(selected_data, axis=0)
            cov += np.cov(selected_data, rowvar=False) * (selected_data.shape[0]-1)
            cov = cov/(train_X.shape[0]-2)

            test_X, test_y = inference(test_loader, encoder)
            features = torch.utils.data.TensorDataset(
                torch.from_numpy(test_X), torch.from_numpy(test_y)
            )
            feature_loader = torch.utils.data.DataLoader(
                features, batch_size=args.logistic_batch_size, shuffle=False
            )

            bound, accuracy = get_bound_rep(feature_loader, torch.tensor(mu_1), torch.tensor(mu_2), torch.tensor(cov), epsilons)
            
            for i in range(len(epsilons)):
                dct_bounds['rep_{}'.format(epsilons[i])].append(bound[i])
                dct_accuracy['rep_{}'.format(epsilons[i])].append(accuracy[i])

        np.save(save_dir + '/mu_scales', mu_scales)
        np.save(save_dir + '/accuracy_raw', dct_accuracy['raw'])
        np.save(save_dir + '/bounds_raw', dct_bounds['raw'])
        for i in range(len(epsilons)):
            np.save(save_dir + '/accuracy_rep_{}'.format(epsilons[i]), dct_accuracy['rep_{}'.format(epsilons[i])])
            np.save(save_dir + '/bounds_rep_{}'.format(epsilons[i]), dct_bounds['rep_{}'.format(epsilons[i])])


    mu_scales = np.load(save_dir + '/mu_scales.npy')
    dct_accuracy['raw'] = np.load(save_dir + '/accuracy_raw.npy')
    dct_bounds['raw'] = np.load(save_dir + '/bounds_raw.npy')
    for i in range(len(epsilons)):
        dct_accuracy['rep_{}'.format(epsilons[i])] = np.load(save_dir + '/accuracy_rep_{}.npy'.format(epsilons[i]))
        dct_bounds['rep_{}'.format(epsilons[i])] = np.load(save_dir + '/bounds_rep_{}.npy'.format(epsilons[i]))
    sz = mu_scales.shape[0]

    xgrid = np.linspace(0.6, 1.0, 100).reshape(100,1)

    # plot the expected bound vs threshold accuracy curve
    fig, ax = plt.subplots()
    ax.plot(xgrid, (np.repeat(np.array(dct_bounds['raw']).reshape(1,sz), 100, axis=0) * (np.array(dct_accuracy['raw'])>xgrid)).sum(axis=1)/sz, '-', color = 'crimson', label='Raw data w/ robust/std. Bayes')
    for i in range(len(epsilons)):
        ax.plot(xgrid, (np.repeat(np.array(dct_bounds['rep_{}'.format(epsilons[i])]).reshape(1,sz), 100, axis=0) * (np.array(dct_accuracy['rep_{}'.format(epsilons[i])])>xgrid)).sum(axis=1)/sz, '-', label='Rep. data w/ {}-robust Bayes'.format(epsilons[i]))
    ax.legend()
    ax.grid(True, linestyle='-.')
    ax.tick_params(labelsize='medium', width=3)
    plt.ylim(0,1.2)
    plt.xlabel("threshold accuracy $a_t$")
    plt.ylabel("expected bound $E_{\Theta,\epsilon}(\Theta,\epsilon,a_t)$")
    plt.show()
    plt.savefig(save_dir + '/plot_bnd_acc.png')

    # calculate synbench-score
    temp=np.array([0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99])
    raw_ref = (np.repeat(((np.repeat(np.array(dct_bounds['raw']).reshape(1,sz), 100, axis=0) * (np.array(dct_accuracy['raw'])>xgrid)).sum(axis=1)/sz).reshape(100,1), temp.shape[0], axis=1) * (np.repeat(xgrid, temp.shape[0], axis=1) >= temp.reshape(1,temp.shape[0]))).sum(axis=0)
    for i in range(len(epsilons)):
        print("Epsilon={}, from ts=0.6 to 0.99".format(epsilons[i]))
        eps0 = (np.repeat(((np.repeat(np.array(dct_bounds['rep_{}'.format(epsilons[i])]).reshape(1,sz), 100, axis=0) * (np.array(dct_accuracy['rep_{}'.format(epsilons[i])])>xgrid)).sum(axis=1)/sz).reshape(100,1), temp.shape[0], axis=1) * (np.repeat(xgrid, temp.shape[0], axis=1) >= temp.reshape(1,temp.shape[0]))).sum(axis=0)
        print(eps0/raw_ref)
    
    