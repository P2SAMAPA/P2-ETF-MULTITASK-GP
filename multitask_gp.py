import torch
import gpytorch
import numpy as np

class MultitaskGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, num_tasks, num_macros, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_tasks])
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.Mean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel() + gpytorch.kernels.LinearKernel()
        )
        self.task_covar = gpytorch.kernels.IndexKernel(num_tasks=num_tasks, rank=num_tasks)

    def forward(self, x, i):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        covar = covar.mul(self.task_covar(i))
        return gpytorch.distributions.MultivariateNormal(mean, covar)

def train_mtgp(train_X, train_Y, num_tasks, num_macros, n_inducing=100, lr=0.01, iterations=500):
    n = train_X.shape[0]
    task_indices = torch.arange(num_tasks).repeat(n, 1).t().reshape(-1)
    X_repeated = train_X.repeat_interleave(num_tasks, dim=0)
    Y_flat = train_Y.T.reshape(-1, 1)
    inducing_points = train_X[:n_inducing]
    model = MultitaskGPModel(num_tasks, num_macros, inducing_points)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(Y_flat))
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
        mean = pred.mean.numpy().reshape(num_tasks, -1).squeeze()
    return mean
