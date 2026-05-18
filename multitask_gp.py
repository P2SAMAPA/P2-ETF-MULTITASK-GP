import torch
import gpytorch
import numpy as np
from sklearn.preprocessing import StandardScaler

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks, num_macros):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        # Covariance module: time kernel (RBF) + task kernel (coregionalization)
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(),
            num_tasks=num_tasks,
            rank=num_tasks
        )
        # Optionally add macro influence: we can extend input with macro features
        # Instead, we'll train on combined input (time + macro) but keep simple for now

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

def train_mtgp(train_X, train_Y, num_tasks, num_macros, n_inducing=100, lr=0.01, iterations=500):
    # train_X: (n_samples, n_features) – here we use only time index and macro values
    # train_Y: (n_samples, num_tasks) – returns for each ETF
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
    model = MultitaskGPModel(train_X, train_Y, likelihood, num_tasks, num_macros)
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for i in range(iterations):
        optimizer.zero_grad()
        output = model(train_X)
        loss = -mll(output, train_Y)
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            print(f"    Iter {i+1}/{iterations}, loss: {loss.item():.4f}")
    return model, likelihood

def predict(model, likelihood, test_X):
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(test_X))
        mean = pred.mean.numpy()
        lower, upper = pred.confidence_region()
    return mean, lower, upper
