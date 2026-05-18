import torch
import gpytorch
import numpy as np

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.Mean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.task_covar = gpytorch.kernels.IndexKernel(num_tasks=num_tasks, rank=num_tasks)

    def forward(self, x, i):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        covar = covar.mul(self.task_covar(i))
        return gpytorch.distributions.MultivariateNormal(mean, covar)

def train_mtgp(train_X, train_Y, num_tasks, n_inducing=None, lr=0.01, iterations=500):
    n = train_X.shape[0]
    task_indices = torch.arange(num_tasks).repeat(n, 1).t().reshape(-1)
    X_repeated = train_X.repeat_interleave(num_tasks, dim=0)
    Y_flat = train_Y.T.reshape(-1, 1)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = MultitaskGPModel(X_repeated, Y_flat.squeeze(), likelihood, num_tasks)
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for i in range(iterations):
        optimizer.zero_grad()
        output = model(X_repeated, task_indices)
        loss = -mll(output, Y_flat.squeeze())
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            print(f"    Iter {i+1}/{iterations}, loss: {loss.item():.4f}")
    return model, likelihood

def predict(model, likelihood, test_X, num_tasks):
    test_indices = torch.arange(num_tasks)
    X_test = test_X.repeat(num_tasks, 1)
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        pred = likelihood(model(X_test, test_indices))
        mean = pred.mean.cpu().numpy()
    return mean
