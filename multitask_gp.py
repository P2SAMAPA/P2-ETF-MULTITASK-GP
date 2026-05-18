import torch
import gpytorch
import numpy as np

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=num_tasks, rank=num_tasks
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

def train_mtgp(train_X, train_Y, num_tasks, num_macros, n_inducing=100, lr=0.01, iterations=500):
    # train_X: (n, d) where d = 1 + num_macros
    # train_Y: (n, num_tasks)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
    model = MultitaskGPModel(train_X, train_Y, likelihood, num_tasks)
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

def predict(model, likelihood, test_X, num_tasks):
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(test_X))
        mean = pred.mean.cpu().numpy()
    # mean shape: (num_tasks,) because test_X is a single point (1, d)
    return mean
