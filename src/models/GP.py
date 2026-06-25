import torch
import gpytorch
import numpy as np

# --- 1. Model Definition ---
class SpatiotemporalGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(SpatiotemporalGPModel, self).__init__(train_x, train_y, likelihood)
        
        # Constant baseline payout
        self.mean_module = gpytorch.means.ConstantMean()
        
        # Spatial Kernel: Matérn 5/2 operating ONLY on columns (0) and rows (1)
        spatial_kernel = gpytorch.kernels.MaternKernel(nu=2.5, active_dims=torch.tensor([0, 1]))
        
        # Temporal Kernel: RBF (Exponential Quadratic) operating ONLY on time (2)
        temporal_kernel = gpytorch.kernels.RBFKernel(active_dims=torch.tensor([2]))
        
        # Multiply them together and wrap in a ScaleKernel to capture overall amplitude
        self.covar_module = gpytorch.kernels.ScaleKernel(spatial_kernel * temporal_kernel)
        

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# --- 2. Training Pipeline ---
def build_and_train_gp(X_train, y_train, training_iterations=150):
    print("Converting arrays to PyTorch Tensors...")
    # Convert numpy arrays to float32 tensors
    train_x = torch.tensor(X_train, dtype=torch.float32)
    train_y = torch.tensor(y_train, dtype=torch.float32)
    
    # Initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = SpatiotemporalGPModel(train_x, train_y, likelihood)
    
    # Switch to training mode
    model.train()
    likelihood.train()
    
    # Use the Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    # "Loss" for GPs is the Negative Marginal Log Likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    print(f"Training GPyTorch Model for {training_iterations} iterations...")
    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        
    print("Training complete!")
    return model, likelihood


# --- 3. Tactical Inference ---
def predict_next_week(model, likelihood, target_week=20):
    # Switch to evaluation (predictive) mode
    model.eval()
    likelihood.eval()
    
    # Construct the Week 20 target grid
    X_new = []
    for tile_idx in range(20):
        x_coord, y_coord = tile_idx % 4, tile_idx // 4
        X_new.append([x_coord, y_coord, target_week])
    test_x = torch.tensor(X_new, dtype=torch.float32)

    print(f"Calculating predictive distribution for Week {target_week}...")
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Get the predictive distribution (includes both model uncertainty and noise)
        observed_pred = likelihood(model(test_x))
        
        # Extract mean and variance back to numpy arrays
        mu_pred = observed_pred.mean.numpy()
        var_pred = observed_pred.variance.numpy()

    # Exponentiate from log-space back to real payouts (Applying Jensen's correction)
    expected_rewards = np.exp(mu_pred + (var_pred / 2.0))
    standard_deviations = np.sqrt((np.exp(var_pred) - 1.0) * np.exp(2 * mu_pred + var_pred))

    # Upper Confidence Bound calculation
    kappa = 1.96 
    ucb_scores = expected_rewards + (kappa * standard_deviations)
    best_tile = np.argmax(ucb_scores)
    
    return best_tile, expected_rewards, ucb_scores