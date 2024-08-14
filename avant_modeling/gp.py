import gpytorch
import torch


class torchGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPModel(torch.nn.Module):
    def __init__(self, train_x, train_y, device="cuda:0", train_lr=1e-1, train_epochs=100):
        super().__init__()
        train_x = train_x.to(device)
        train_y = train_y.to(device)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        self.model = torchGPModel(train_x, train_y, self.likelihood).to(device)
        
        self.likelihood.train()
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=train_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=33, gamma=1e-1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        for i in range(train_epochs):
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            print("GP epoch %d, loss: %.8f" % (i, loss))

        self.likelihood.eval()
        self.model.eval()

        self.trained_params = {
            "sigma_f": self.model.covar_module.outputscale.item(),
            "lengthscale": self.model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().tolist(),
            "sigma_n": self.likelihood.noise.item()
        }

    def fantasy_model(self, inputs, outputs):
        # self.model.get_fantasy_model
        self.model.set_train_data(inputs, outputs, False)

    def forward(self, x):
        with gpytorch.settings.fast_pred_var():
            return self.likelihood(self.model(x))