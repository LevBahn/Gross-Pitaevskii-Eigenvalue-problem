import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import os
from torch.autograd import grad
from scipy.special import hermite
from adabelief_pytorch import AdaBelief
from pytorch_optimizer import QHAdam, AdaHessian, Ranger21, SophiaH, Shampoo
from torch.optim.lr_scheduler import CosineAnnealingLR
from distributed_shampoo import AdamGraftingConfig, DistributedShampoo
import torch.nn.utils

# import matplotlib
# try:
#     matplotlib.use('Qt5Agg')
# except:
#     pass

import matplotlib.pyplot as plt

# Check if GPU is available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GrossPitaevskiiPINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) for solving the 1D Gross-Pitaevskii Equation.
    """

    def __init__(self, layers, hbar=1.0, m=1.0, mode=0, beta_init=1.0, alpha_init=1.0, decay_rate=0.001, gamma=1.0, mu=0.0):
        """
        Parameters
        ----------
        layers : list of int
            Neural network architecture, each entry defines the number of neurons in that layer.
        hbar : float, optional
            Reduced Planck's constant (default is 1.0).
        m : float, optional
            Mass of the particle (default is 1.0).
        mode : int, optional
            Mode number (default is 0).
        beta_init : float, optional
            Initial weight for previous predictions, default is 1.0.
        alpha_init : float, optional
            Initial weight for learned perturbations, default is 1.0.
        decay_rate : float, optional
            Rate at which beta decays over time, default is 0.001.
        """
        super().__init__()
        self.layers = layers
        self.network = self.build_network()
        self.hbar = hbar  # Planck's constant, fixed
        self.m = m  # Particle mass, fixed
        self.mode = mode  # Mode number (n)
        self.beta = beta_init
        self.alpha = alpha_init
        self.decay_rate = decay_rate
        self.iteration = 0
        self.gamma = gamma  # Interaction strength parameter
        self.mu = mu  # Store the chemical potential

    def update_alpha_beta(self):
        """
        Updates the weighting factors alpha and beta adaptively.
        """
        self.iteration += 1
        self.beta = max(0.5, self.beta * torch.exp(torch.tensor(-self.decay_rate * self.iteration)))
        self.alpha = 1.0 + (1.0 - self.beta)  # Makes α increase as β decreases

    def build_network(self):
        """
        Build the neural network with sine activation functions between layers.

        Returns
        -------
        nn.Sequential
            A PyTorch sequential model representing the neural network architecture.
        """
        layers = []
        for i in range(len(self.layers) - 1):
            layers.append(nn.Linear(self.layers[i], self.layers[i + 1]))
            if i < len(self.layers) - 2:
                layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def weighted_hermite(self, x, n):
        """
        Compute the weighted Hermite polynomial solution for the linear case (gamma = 0).
        Equation (34) in https://www.sciencedirect.com/science/article/abs/pii/S0010465513001318.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of spatial coordinates (collocation points) or boundary points.
        n : int
            Mode of ground state solution to Gross-Pitavskii equation (0 for base ground state)

        Returns
        -------
        torch.Tensor
            The weighted Hermite polynomial solution for the linear case (gamma = 0).
        """
        H_n = hermite(n)(x.cpu().detach().numpy())  # Hermite polynomial evaluated at x
        norm_factor = (2**n * math.factorial(n) * np.sqrt(np.pi))**(-0.5)
        weighted_hermite = norm_factor * torch.exp(-x**2 / 2) * torch.tensor(H_n, dtype=torch.float32).to(device)

        return weighted_hermite

    def fixed_mu_pde_loss(self, inputs, predictions, mu, gamma, potential_type, precomputed_potential=None,
                          prev_prediction=None, mode=0):
        """
        Modified PDE loss that matches Algorithm 2 from the paper more closely.
        """
        u = self.get_complete_solution(inputs, predictions, mode)

        # Don't normalize here - we want the natural scaling with μ

        # Recalculate derivatives
        u_x = torch.autograd.grad(outputs=u, inputs=inputs,
                                  grad_outputs=torch.ones_like(u),
                                  create_graph=True, retain_graph=True)[0]
        u_xx = torch.autograd.grad(outputs=u_x, inputs=inputs,
                                   grad_outputs=torch.ones_like(u_x),
                                   create_graph=True, retain_graph=True)[0]

        if precomputed_potential is not None:
            V = precomputed_potential
        else:
            V = self.compute_potential(inputs, potential_type)

        # Use gamma for the interaction term as per the paper
        residual = -0.5 * u_xx + V * u + gamma * u ** 3 - mu * u
        pde_loss = torch.mean(residual ** 2)
        return pde_loss, residual

    def forward(self, inputs):
        """
        Forward pass through the neural network.
        """
        return self.network(inputs)

    def compute_potential(self, x, potential_type="harmonic", **kwargs):
        """
        Compute a symmetric or asymmetric potential function for the 1D domain.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of spatial coordinates.
        potential_type : str, optional
            Type of potential. Options are: "harmonic".
        kwargs : dict
            Additional parameters specific to each potential type.

        Returns
        -------
        V : torch.Tensor
            Tensor of potential values at the input points.

        Raises
        ------
        ValueError
            If the potential type is not recognized.
        """
        if potential_type == "harmonic":
            omega = kwargs.get('omega', 1.0)  # Frequency for harmonic potential
            V = 0.5 * omega ** 2 * x ** 2
        elif potential_type == "gaussian":
            a = kwargs.get('a', 0.0)  # Center of the Gaussian
            V = torch.exp(-(x - a) ** 2)
        elif potential_type == "periodic":
            V0 = kwargs.get('V0', 1.0)  # Depth of the potential
            k = kwargs.get('k', 2 * np.pi / 5.0)  # Wave number for periodic potential
            V = V0 * torch.cos(k * x) ** 2
        else:
            raise ValueError(f"Unknown potential type: {potential_type}")

        return V

    def compute_thomas_fermi_approx(self, mu, x, gamma):
        """
        Calculate the Thomas–Fermi approximation for the given potential.

        Parameters
        ----------
        mu : float
            Chemical potential.
        x : torch.Tensor
            Spatial coordinates.
        gamma : float
            Interaction strength.

        Returns
        -------
        torch.Tensor
            Thomas–Fermi approximation of the wave function.
        """
        # For gamma = 0, return None (TF approximation not valid)
        if gamma == 0:
            return None

        # Compute potential
        V = self.compute_potential(x, "harmonic")

        # Calculate TF approximation: ψ_TF(x) = sqrt(max(0, (μ - V(x))/γ))
        zeros = torch.zeros_like(x)
        tf_approx = torch.sqrt(torch.maximum((mu - V) / gamma, zeros))

        # Normalize the TF approximation
        if torch.sum(tf_approx ** 2) > 0:
            # If dx is available
            if x.size(0) > 1:
                dx = x[1, 0] - x[0, 0]
                norm = torch.sqrt(torch.sum(tf_approx ** 2) * dx)
                tf_approx = tf_approx / norm

        return tf_approx

    def get_complete_solution(self, x, perturbation, mode=0):
        """
        Properly combine base solution with perturbation based on chemical potential.
        """
        base_solution = self.weighted_hermite(x, mode)

        # For μ=0, return just the base solution
        if self.mu == 0.0:
            return base_solution

        # For μ>0, scale the perturbation proportionally to μ
        # This is key to getting the widening effect as μ increases
        scaling_factor = self.mu / 5  # Adjust this based on paper's plots
        scaled_perturbation = perturbation * scaling_factor

        # The total solution grows with μ rather than being normalized to unit norm
        return base_solution + scaled_perturbation

    def boundary_loss(self, boundary_points, boundary_values, prev_prediction=None, mode=0):
        """
        Compute the boundary loss (MSE) for the boundary conditions.

        Parameters
        ----------
        boundary_points : torch.Tensor
            Input tensor of boundary spatial points.
        boundary_values : torch.Tensor
            Tensor of boundary values (for Dirichlet conditions).
        prev_prediction : GrossPitaevskiiPINN or None
            Previously trained model whose predictions are used as part of the training process.
            If None, the model starts training from scratch.
        mode: int
            Mode of ground state solution to Gross-Pitavskii equation. Default is 0.

        Returns
        -------
        torch.Tensor
            Mean squared error (MSE) at the boundary points.
        """
        u_pred = self.forward(boundary_points)
        full_u = self.get_complete_solution(boundary_points, u_pred, mode)
        return torch.mean((full_u - boundary_values) ** 2)

    def riesz_loss(self, predictions, inputs, eta, potential_type, precomputed_potential=None, prev_prediction=None, mode=0):
        """
        Compute the Riesz energy loss for the Gross-Pitaevskii equation.

        Parameters
        ----------
        predictions : torch.Tensor
            Predicted solution from the network.
        inputs : torch.Tensor
            Input tensor of spatial coordinates (collocation points).
        eta : float
            Interaction strength.
        potential_type : str
            Type of potential function to use.
        precomputed_potential : torch.Tensor
            Precomputed potential. Default is None.
        prev_prediction : GrossPitaevskiiPINN or None
            Previously trained model whose predictions are used as part of the training process.
            If None, the model starts training from scratch.
        mode: int
            Mode of ground state solution to Gross-Pitavskii equation. Default is 0.

        Returns
        -------
        torch.Tensor
            Riesz energy loss value.
        """

        u = self.get_complete_solution(inputs, predictions, mode)

        if not inputs.requires_grad:
            inputs = inputs.clone().detach().requires_grad_(True)
        u_x = torch.autograd.grad(outputs=u, inputs=inputs,
                                  grad_outputs=torch.ones_like(predictions),
                                  create_graph=True, retain_graph=True)[0]

        laplacian_term = torch.mean(u_x ** 2)  # Kinetic term
        if precomputed_potential is not None:
            V = precomputed_potential
        else:
            V = self.compute_potential(inputs, potential_type)
        potential_term = torch.mean(V * u ** 2)  # Potential term
        interaction_term = 0.5 * eta * torch.mean(u ** 4)  # Interaction term

        riesz_energy = 0.5 * (laplacian_term + potential_term + interaction_term)
        return riesz_energy

    def symmetry_loss(self, collocation_points, lb, ub):
        """
        Compute the symmetry loss to enforce u(x) = u((a+b)-x).

        Parameters
        ----------
        collocation_points : torch.Tensor
            Tensor of interior spatial points.
        lb : torch.Tensor
            Lower bound of interval.
        ub: torch.Tensor
            Upper bound of interval.

        Returns
        -------
        sym_loss : torch.Tensor
            The mean squared error enforcing symmetry u(x) = u((a+b)-x).
        """
        # Reflect points across the center of the domain
        x_reflected = (lb + ub) - collocation_points

        # Evaluate u(x) and u((a+b)-x)
        u_original = self.forward(collocation_points)
        u_reflected = self.forward(x_reflected)

        # Compute MSE to enforce symmetry
        sym_loss = torch.mean((u_original - u_reflected) ** 2)
        return sym_loss

    def compute_normalization_loss(self, u, x):
        """Compute normalization loss using proper numerical integration"""
        dx = x[1] - x[0]  # Assuming uniform grid
        integral = torch.sum(u ** 2) * dx
        return (integral - 1.0) ** 2

def initialize_weights(m):
    """
    Initialize the weights of the neural network layers using Xavier uniform initialization.

    Parameters
    ----------
    m : torch.nn.Module
        A layer of the neural network. If it is a linear layer, its weights and biases are initialized.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def prepare_training_data(N_u, N_f, lb, ub):
    """
    Prepare boundary and collocation points for training.

    Parameters
    ----------
    N_u : int
        Number of boundary points.
    N_f : int
        Number of collocation points.
    lb : np.ndarray
        Lower bounds of the domain.
    ub : np.ndarray
        Upper bounds of the domain.

    Returns
    -------
    collocation_points : np.ndarray
        Collocation points.
    boundary_points : np.ndarray
        Boundary points.
    boundary_values : np.ndarray
        Boundary values.
    """

    # Boundary of interval
    boundary_points = np.array([[lb], [ub]])
    boundary_values = np.zeros((2, 1))

    # Dynamically sample points inside the interval
    collocation_points = np.random.rand(N_f, 1) * (ub - lb) + lb

    return collocation_points, boundary_points, boundary_values


def normalize_wavefunction(u):
    u = u.clone()
    norm = torch.norm(u, p=2)
    return u / norm


def train_with_fixed_gamma(gamma_values, modes, X_train, N_u, N_f, lb, ub, layers, epochs, weights,
                           potential_type='harmonic'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mu_table = {}
    models_by_mode = {}

    boundary_points = torch.tensor([[lb], [ub]], dtype=torch.float32).to(device)
    boundary_values = torch.zeros((2, 1), dtype=torch.float32).to(device)

    # Sort gamma values to train from smaller to larger (helps convergence)
    gamma_values = sorted(gamma_values)

    for mode in modes:
        mu_logs = []
        prev_prediction = None
        models_by_gamma = {}

        for gamma in gamma_values:
            print(f"\nTraining for γ = {gamma:.2f}, mode = {mode}")
            model = GrossPitaevskiiPINN(layers, mode=mode, gamma=gamma).to(device)
            model.apply(initialize_weights)

            # Use Adam optimizer with a reasonable learning rate
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, factor=0.5,
                                                                   verbose=True)

            X_tensor = torch.tensor(X_train, dtype=torch.float32, requires_grad=True).to(device)
            lb_tensor = torch.tensor(lb, dtype=torch.float32).to(device)
            ub_tensor = torch.tensor(ub, dtype=torch.float32).to(device)

            # Track mu values during training
            mu_history = []
            loss_history = []

            for epoch in range(epochs):
                optimizer.zero_grad()
                u_pred = model.forward(X_tensor)

                # Get PDE loss and compute mu value
                pde_loss, _, lambda_pde = model.pde_loss(X_tensor, u_pred, gamma, potential_type, None, prev_prediction,
                                                         mode)
                boundary_loss = model.boundary_loss(boundary_points, boundary_values, prev_prediction, mode)

                # Get full solution and normalize
                full_u = model.get_complete_solution(X_tensor, u_pred, mode)
                dx = X_train[1, 0] - X_train[0, 0]  # Assuming uniform grid
                norm_squared = torch.sum(full_u ** 2) * dx
                norm_loss = (norm_squared - 1.0) ** 2

                sym_loss = model.symmetry_loss(X_tensor, lb_tensor, ub_tensor)

                # Calculate Riesz energy loss
                riesz_loss = model.riesz_loss(u_pred, X_tensor, gamma, potential_type, None, prev_prediction, mode)

                # Use all loss components with appropriate weights
                total_loss = (
                        weights[0] * boundary_loss +
                        weights[1] * riesz_loss / (ub - lb) +
                        weights[2] * pde_loss / (ub - lb) +
                        weights[3] * norm_loss +
                        weights[4] * sym_loss
                )

                # Store history for plotting
                if epoch % 100 == 0:
                    mu_history.append(lambda_pde.item())
                    loss_history.append(total_loss.item())

                total_loss.backward()

                # Clip gradients for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step(total_loss)

                if epoch % 500 == 0:
                    print(f"Epoch {epoch}, Loss: {total_loss.item():.6f}, μ: {lambda_pde.item():.4f}")

            # Use the average of the last few mu values for stability
            final_mu = sum(mu_history[-10:]) / 10 if mu_history else 0
            mu_logs.append((gamma, final_mu))
            models_by_gamma[gamma] = model

            # For γ > 0, use the previous model's predictions to help convergence
            if gamma > 0:
                prev_prediction = make_prev_prediction(model, mode)

        mu_table[mode] = mu_logs
        models_by_mode[mode] = models_by_gamma

    return models_by_mode, mu_table


def plot_loss_history(loss_histories, etas, optimizer_names, save_path='plots/loss_history.png', potential_type='gaussian'):
    """
    Plot the training loss history for all optimizers and interaction strengths (etas) in a single plot.

    Parameters
    ----------
    loss_histories : dict
        Dictionary where keys are eta values, and values are dictionaries mapping optimizer names to loss history lists.
    etas : list of float
        List of eta values used in training.
    optimizer_names : list of str
        List of optimizer names used during training.
    save_path : str, optional
        File path to save the plot, by default 'plots/loss_history.png'.
    potential_type : str, optional
        Type of potential function to use. Default is 'gaussian'.

    Returns
    -------
    None
    """

    # Ensure the plots directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 6))

    for optimizer_name in optimizer_names:
        for eta in etas:
            if eta in loss_histories and optimizer_name in loss_histories[eta]:
                loss = loss_histories[eta][optimizer_name]
                plt.plot(np.arange(len(loss)), loss, marker='o', label=f"{optimizer_name}, η={eta}")

    plt.xlabel('Training step (x 100)', fontsize="xx-large")
    plt.ylabel('Total Loss', fontsize="xx-large")
    plt.yscale('log')
    plt.title(f'Loss History for Different Interaction Strengths ($\\eta$) for {potential_type.capitalize()} Potential',
              fontsize="xx-large")
    plt.legend(fontsize="large")
    plt.grid(True)

    # Set larger tick sizes
    plt.xticks(fontsize="x-large")
    plt.yticks(fontsize="x-large")

    # Save the plot
    save_path = save_path.format(potential_type=potential_type)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_lambda_pde(lambda_pde_histories, etas, optimizer_names, save_path="plots/lambda_pde.png", potential_type="harmonic"):
    """
    Plot the evolution of λ_PDE over training iterations for all optimizers and interaction strengths in one figure.

    Parameters
    ----------
    lambda_pde_histories : dict
        Dictionary where keys are eta values, and values are dictionaries mapping optimizer names to lambda_pde history lists.
    etas : list of float
        List of eta values used in training.
    optimizer_names : list of str
        List of optimizer names used during training.
    save_path : str, optional
        File path to save the plot, by default 'plots/lambda_pde.png'.
    potential_type : str, optional
        Type of potential function to use. Default is 'gaussian'.

    Returns
    -------
    None
    """

    # Ensure the plots directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 6))

    for optimizer_name in optimizer_names:
        for eta in etas:
            if eta in lambda_pde_histories and optimizer_name in lambda_pde_histories[eta]:
                lambda_pde = lambda_pde_histories[eta][optimizer_name]
                plt.plot(np.arange(len(lambda_pde)), lambda_pde, marker='o', label=f"{optimizer_name}, η={eta}")

    plt.xlabel('Training step (x 100)', fontsize="xx-large")
    plt.ylabel(r"$\lambda_{PDE}$", fontsize="xx-large")
    plt.title(
        r"$\lambda_{PDE}$ for Different Interaction Strengths ($\eta$) for " + f"{potential_type.capitalize()} Potential",
        fontsize="xx-large")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()

    # Set larger tick sizes
    plt.xticks(fontsize="x-large")
    plt.yticks(fontsize="x-large")

    # Save the plot
    save_path = save_path.format(potential_type=potential_type)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def compute_gamma(model, X_tensor, mu, mode):
    """
    Compute gamma parameter based on the model's current prediction,
    following Algorithm 2 from the paper.
    """
    model.eval()
    with torch.no_grad():
        u_pred = model.forward(X_tensor)
        full_u = model.get_complete_solution(X_tensor, u_pred, mode)

        # Calculate dx for proper integration
        dx = X_tensor[1] - X_tensor[0]

        # Calculate intensities
        u_squared = full_u ** 2
        norm_squared = torch.sum(u_squared) * dx

        # Compute gamma according to equation in Algorithm 2
        if mu == 0.0:
            gamma = 0.0
        else:
            gamma = mu / (torch.sum(u_squared ** 2) / norm_squared ** 2 * dx)

    return gamma


def make_prev_prediction(model, mode):
    model.eval()
    def _predict(x):
        with torch.no_grad():
            u_pred = model.forward(x)
            return model.get_complete_solution(x, u_pred, mode)
    return _predict


def train_with_mu_and_mode(mu_values, modes, X_train, N_u, N_f, lb, ub, layers, epochs, weights,
                           potential_type='harmonic'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gamma_table = {}
    models_by_mode = {}

    boundary_points = torch.tensor([[lb], [ub]], dtype=torch.float32).to(device)
    boundary_values = torch.zeros((2, 1), dtype=torch.float32).to(device)

    for mode in modes:
        gamma_logs = []
        prev_prediction = None
        models_by_mu = {}

        for mu in mu_values:
            print(f"\nTraining for μ = {mu:.2f}, mode = {mode}")
            model = GrossPitaevskiiPINN(layers, mode=mode, mu=mu).to(device)
            model.apply(initialize_weights)

            # Use a more appropriate optimizer and learning rate according to the paper
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, factor=0.5,
                                                                   verbose=True)

            X_tensor = torch.tensor(X_train, dtype=torch.float32, requires_grad=True).to(device)
            lb_tensor = torch.tensor(lb, dtype=torch.float32).to(device)
            ub_tensor = torch.tensor(ub, dtype=torch.float32).to(device)

            # Initialize gamma parameter dynamically - will be adjusted during training
            gamma = 0.0 if mu == 0.0 else 1.0  # Start with a reasonable initial value

            for epoch in range(epochs):
                optimizer.zero_grad()
                u_pred = model.forward(X_tensor)

                # Get full solution (neural network output + weighted Hermite polynomial)
                full_u = model.get_complete_solution(X_tensor, u_pred, mode)

                # If mu > 0, adaptively compute gamma parameter as in Algorithm 2
                # This ensures we get the correct nonlinear interaction strength
                if mu > 0.0:
                    # Calculate gamma according to normalizing mean-field energy
                    dx = X_train[1, 0] - X_train[0, 0]  # Assuming uniform grid

                    # Calculate modified normalization (Equation 13 in the paper)
                    u_squared = full_u ** 2
                    norm_squared = torch.sum(u_squared) * dx

                    # Update gamma as per Algorithm 2
                    gamma = mu / (torch.sum(u_squared ** 2) / norm_squared ** 2 * dx)

                    # Normalize the wavefunction to have unit norm
                    full_u = full_u / torch.sqrt(norm_squared)

                # Compute PDE loss with fixed mu and computed gamma
                pde_loss, _ = model.fixed_mu_pde_loss(X_tensor, u_pred, mu, gamma, potential_type, None,
                                                      prev_prediction, mode)

                # Compute boundary loss (Dirichlet boundary conditions)
                boundary_loss = model.boundary_loss(boundary_points, boundary_values, prev_prediction, mode)

                # Compute normalization loss to ensure proper norm
                dx = X_train[1, 0] - X_train[0, 0]  # Assuming uniform grid
                norm_squared = torch.sum(full_u ** 2) * dx
                norm_loss = (norm_squared - 1.0) ** 2

                # Compute symmetry loss if needed
                sym_loss = model.symmetry_loss(X_tensor, lb_tensor, ub_tensor)

                # Reduce the weight on normalization constraint for higher μ values
                if mu > 0:
                    norm_weight = weights[1] / (1 + mu / 10)  # Decrease normalization importance as μ increases
                else:
                    norm_weight = weights[1]

                total_loss = (
                        weights[0] * pde_loss / (ub - lb) +  # Scale PDE loss
                        norm_weight * norm_loss +  # Relaxed normalization for higher μ
                        # weights[2] * sym_loss +  # Symmetry constraint
                        weights[2] * boundary_loss  # Boundary conditions
                )

                # Backpropagate
                total_loss.backward()

                # Clip gradients for stability (recommended for PINNs)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step(total_loss)

                if epoch % 500 == 0:
                    print(f"Epoch {epoch}, Loss: {total_loss.item():.6f}, γ: {gamma:.4f}")

            # Store the final gamma value for this mu
            gamma_logs.append((mu, gamma.item() if isinstance(gamma, torch.Tensor) else gamma))
            models_by_mu[mu] = model

            # For the next mu value, use the current model as a warm start if helpful
            if mu > 0.0:
                prev_prediction = make_prev_prediction(model, mode)

        gamma_table[mode] = gamma_logs
        models_by_mode[mode] = models_by_mu

    return models_by_mode, gamma_table


def plot_figure11_corrected(models_by_mode, X_test, modes=[0, 1, 2, 3], save_dir="plots"):
    """
    Plot Figure 11 from the paper, preserving the natural scaling with μ.
    """
    os.makedirs(save_dir, exist_ok=True)
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    linestyles = ['-', '--', '-.', ':', '-', '--']
    colors = ['k', 'b', 'r', 'g', 'm', 'c']

    for idx, mode in enumerate(modes):
        ax = axs[idx]
        mu_values = sorted(list(models_by_mode[mode].keys()))

        for i, mu in enumerate(mu_values):
            model = models_by_mode[mode][mu]
            model.eval()
            with torch.no_grad():
                u_pred = model.forward(X_tensor)
                full_u = model.get_complete_solution(X_tensor, u_pred, mode)

                # Don't normalize - keep the natural amplitude that increases with μ
                u_np = full_u.cpu().numpy().flatten()

                ax.plot(X_test.flatten(), u_np,
                        linestyle=linestyles[i % len(linestyles)],
                        color=colors[i % len(colors)],
                        label=f"μ={mu:.1f}")

        ax.set_title(f"Mode {mode}", fontsize=14)
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("ψ(x)", fontsize=12)
        ax.grid(True)
        ax.legend()
        ax.set_xlim(-10, 10)

    fig.suptitle("Wavefunctions for different modes and chemical potentials", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig(os.path.join(save_dir, "figure11_all_modes.png"), dpi=300)
    plt.show()


def plot_figure2(models_by_mode, X_test, gamma_values, mu_table, save_dir="plots"):
    """
    Plot Figure 2 from the paper - comparison with Thomas-Fermi approximation
    """
    os.makedirs(save_dir, exist_ok=True)
    X_tensor = torch.tensor(X_test.reshape(-1, 1), dtype=torch.float32).to(device)
    dx = X_test[1] - X_test[0]  # Spatial step size

    # Use mode 0 with the highest gamma value (strongest interaction)
    mode = 0
    gamma = max(gamma_values)  # Should be 100.0 to match the paper
    model = models_by_mode[mode][gamma]
    model.eval()

    # Get the mu value for this gamma
    mu = next(mu for g, mu in mu_table[mode] if g == gamma)

    # Get the predicted wavefunction
    with torch.no_grad():
        u_pred = model.forward(X_tensor)
        full_u = model.get_complete_solution(X_tensor, u_pred, mode)
        u_np = full_u.cpu().numpy().flatten()

    # Proper normalization
    u_np /= np.sqrt(np.sum(u_np ** 2) * dx)

    # Calculate Thomas-Fermi approximation
    tf_approx = model.compute_thomas_fermi_approx(mu, X_tensor, gamma)
    tf_np = tf_approx.cpu().numpy().flatten()

    # Calculate potential V(x)/μ
    V = 0.5 * X_test ** 2  # Harmonic potential
    V_scaled = V / mu

    # Plot with exact paper styling
    plt.figure(figsize=(10, 6))
    plt.plot(X_test, u_np ** 2, 'b-', linewidth=2, label="Numerical (PINN)")
    plt.plot(X_test, tf_np ** 2, 'r--', linewidth=2, label="Thomas-Fermi")
    plt.plot(X_test, V_scaled, 'g-.', linewidth=2, label="V(x)/μ")

    plt.title(f"Ground state density, γ={gamma:.1f}, μ={mu:.2f}", fontsize=14)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("|ψ(x)|²", fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.xlim(-5, 5)  # Match the paper's x-range

    # Add a tight layout
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, "figure2_comparison.png"), dpi=300)
    plt.show()


def plot_gamma_table(gamma_table, save_path="plots/table_gamma_vs_mu.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    for mode, gamma_logs in gamma_table.items():
        mus, gammas = zip(*gamma_logs)
        plt.plot(mus, gammas, marker='o', label=f"mode {mode}")

    plt.title("γ vs μ (per mode)", fontsize=16)
    plt.xlabel("μ")
    plt.ylabel("Estimated γ")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_mu_vs_gamma(gamma_table, save_path="plots/mu_vs_gamma.png"):
    """
    Plot the relationship between chemical potential (mu) and interaction strength (gamma)
    for each mode, as in Figure 10 of the paper.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 6))

    markers = ['o', 's', 'D', '^']
    colors = ['blue', 'red', 'green', 'purple']

    for i, (mode, gamma_logs) in enumerate(gamma_table.items()):
        mus, gammas = zip(*gamma_logs)
        plt.plot(mus, gammas, marker=markers[i % len(markers)], color=colors[i % len(colors)],
                 linestyle='-', linewidth=2, label=f"Mode {mode}")

    plt.title("Interaction strength (γ) vs. Chemical potential (μ)", fontsize=16)
    plt.xlabel("Chemical potential (μ)", fontsize=14)
    plt.ylabel("Interaction strength (γ)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


if __name__ == "__main__":
    lb, ub = -10, 10  # Domain boundaries
    N_f = 4000  # Number of collocation points
    N_u = 200  # Number of boundary points
    epochs = 5001  # More epochs for better convergence as in the paper
    layers = [1, 100, 100, 100, 1]  # Neural network architecture as in the paper

    X = np.linspace(lb, ub, N_f).reshape(-1, 1)
    X_test = np.linspace(lb, ub, 1000).reshape(-1, 1)  # Higher resolution for plotting

    # Use mu values directly from Figure 11 in the paper
    mu_values = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]  # Exact values from the paper
    modes = [0, 1, 2, 3]  # All modes shown in Figure 11

    # Adjusted weights for better convergence
    # [pde_loss, norm_loss, sym_loss, boundary_loss]
    # weights = [1.0, 10.0, 0.1, 10.0]
    weights = [1.0, 10.0, 10.0]

    # Train with fixed mu values (Algorithm 2)
    models_by_mode, gamma_table = train_with_mu_and_mode(
        mu_values, modes, X, N_u, N_f, lb, ub, layers, epochs, weights)

    # Plot Figure 11 with correct implementation
    plot_figure11_corrected(models_by_mode, X_test, modes=modes)

    # Plot mu vs gamma as in Figure 10 of the paper
    plot_mu_vs_gamma(gamma_table)

    # Also create a table of gamma values
    print("\nComputed γ values for each mode and μ:")
    for mode in modes:
        print(f"Mode {mode}:")
        for mu, gamma in gamma_table[mode]:
            print(f"  μ = {mu:.1f}, γ = {gamma:.4f}")
