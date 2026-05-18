import torch
import gpytorch
import numpy as np

class IndependentGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

def train_individual_gps(train_X, train_Y, num_tasks, lr=0.01, iterations=500):
    """
    train_X: torch tensor (n, d)
    train_Y: torch tensor (n, num_tasks)
    Returns lists of models and likelihoods.
    """
    models = []
    likelihoods = []
    for task in range(num_tasks):
        y_task = train_Y[:, task]
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = IndependentGPModel(train_X, y_task, likelihood)
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        for i in range(iterations):
            optimizer.zero_grad()
            output = model(train_X)
            loss = -mll(output, y_task)
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                print(f"    Task {task+1} iter {i+1}/{iterations}, loss: {loss.item():.4f}")
        models.append(model)
        likelihoods.append(likelihood)
    return models, likelihoods

def predict_individual(models, likelihoods, test_X, num_tasks):
    """
    Returns array of predicted means (length num_tasks).
    """
    preds = []
    for task in range(num_tasks):
        model = models[task]
        likelihood = likelihoods[task]
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            pred = likelihood(model(test_X))
            mean = pred.mean.cpu().numpy()
            preds.append(mean.item())
    return np.array(preds)
