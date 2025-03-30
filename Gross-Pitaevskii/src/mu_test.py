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

    def __init__(self, layers, hbar=1.0, m=1.0, mode=0, beta_init=1.0, alpha_init=1.0, decay_rate=0.001, gamma=1.0):
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
        u = self.get_complete_solution(inputs, predictions, mode)

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

    def pde_loss(self, inputs, predictions, gamma, potential_type, precomputed_potential=None, prev_prediction=None,
                 mode=0):
        """
        Compute the PDE loss for the Gross-Pitaevskii equation.
        """
        u = self.get_complete_solution(inputs, predictions, mode)

        # Compute first and second derivatives with respect to x
        u_x = grad(u, inputs, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = grad(u_x, inputs, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

        # Compute potential
        if precomputed_potential is not None:
            V = precomputed_potential
        else:
            V = self.compute_potential(inputs, potential_type)

        # Calculate chemical potential (μ) as per the paper
        lambda_pde = torch.mean(0.5 * u_x ** 2 + V * u ** 2 + gamma * u ** 4) / torch.mean(u ** 2)

        # Residual of the 1D Gross-Pitaevskii equation (with the 0.5 factor for kinetic term)
        pde_residual = -0.5 * u_xx + V * u + gamma * u ** 2 * u - lambda_pde * u

        # PDE loss (residual)
        pde_loss = torch.mean(pde_residual ** 2)

        return pde_loss, pde_residual, lambda_pde

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
        base_solution = self.weighted_hermite(x, mode)
        return base_solution + self.alpha * perturbation

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

    def total_loss(self, collocation_points, boundary_points, boundary_values, eta, lb, ub, weights, potential_type,
                   precomputed_potential=None, prev_prediction=None, mode=0):
        """
        Compute the total loss combining boundary loss, Riesz energy loss,
        PDE loss, L^2 norm regularization loss, and symmetry loss.

        Parameters
        ----------
        collocation_points : torch.Tensor
            Input tensor of spatial coordinates for the interior points.
        boundary_points : torch.Tensor
            Input tensor of boundary spatial points.
        boundary_values : torch.Tensor
            Tensor of boundary values (for Dirichlet conditions).
        eta : float
            Interaction strength.
        lb : torch.Tensor
            Lower bound of interval.
        ub : torch.Tensor
            Upper bound of interval.
        weights : list
            Weights for different loss terms.
        potential_type : str
            Type of potential function to use
        precomputed_potential : torch.Tensor
            Precomputed potential. Default is None.
        prev_prediction : GrossPitaevskiiPINN or None
            Previously trained model whose predictions are used as part of the training process.
            If None, the model starts training from scratch. Default is None.
        mode: int
            Mode of ground state solution to Gross-Pitavskii equation. Default is 0.

        Returns
        -------
        total_loss : torch.Tensor
            Total loss value.
        """

        # Use precomputed potential if provided
        if precomputed_potential is not None:
            V = precomputed_potential
        else:
            V = self.compute_potential(collocation_points, potential_type)

        # Compute individual loss components
        data_loss = self.boundary_loss(boundary_points, boundary_values, prev_prediction, mode)
        riesz_energy_loss = self.riesz_loss(self.forward(collocation_points), collocation_points, eta, potential_type,
                                            V, prev_prediction, mode)
        pde_loss, _, _ = self.pde_loss(collocation_points, self.forward(collocation_points), eta, potential_type,
                                            V, prev_prediction, mode)
        norm_loss = (torch.norm(self.forward(collocation_points), p=2) - 1) ** 2
        # full_u = self.get_complete_solution(collocation_points, self.forward(collocation_points), prev_prediction, mode)
        # norm_loss = (torch.norm(full_u, p=2) - 1) ** 2

        sym_loss = self.symmetry_loss(collocation_points, lb, ub)

        # Scaling factor for pde loss and riesz energy loss
        domain_length = ub - lb

        # Compute weighted losses and total loss
        losses = [data_loss, riesz_energy_loss  / domain_length, pde_loss / domain_length, norm_loss, sym_loss]
        weighted_losses = [weights[i] * loss for i, loss in enumerate(losses)]
        total_loss = sum(weighted_losses)

        # Add regularization term to encourage wider solutions for higher eta
        # width_penalty = -0.01 * eta * torch.mean(collocation_points ** 2 * full_u ** 2)

        # total_loss = data_loss + (pde_loss / domain_length) + (riesz_energy_loss  / domain_length) + width_penalty
        total_loss = data_loss + (pde_loss / domain_length) +  (riesz_energy_loss / domain_length) + norm_loss #+ width_penalty

        return total_loss, data_loss, riesz_energy_loss, pde_loss, norm_loss


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


def compute_gamma(u):
    return 1.0 / torch.mean(u ** 2).item()


def make_prev_prediction(model, mode):
    model.eval()
    def _predict(x):
        with torch.no_grad():
            u_pred = model.forward(x)
            return model.get_complete_solution(x, u_pred, mode)
    return _predict


def train_with_mu_and_mode(mu_values, modes, X_train, N_u, N_f, lb, ub, layers, epochs, weights, potential_type='harmonic'):
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
            model = GrossPitaevskiiPINN(layers, mode=mode).to(device)
            model.apply(initialize_weights)

            optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

            X_tensor = torch.tensor(X_train, dtype=torch.float32, requires_grad=True).to(device)
            lb_tensor = torch.tensor(lb, dtype=torch.float32).to(device)
            ub_tensor = torch.tensor(ub, dtype=torch.float32).to(device)

            for epoch in range(epochs):
                optimizer.zero_grad()
                u_pred = model.forward(X_tensor)
                full_u = model.get_complete_solution(X_tensor, u_pred, mode)
                gamma = compute_gamma(full_u)
                full_u = normalize_wavefunction(full_u)
                pde_loss, _ = model.fixed_mu_pde_loss(X_tensor, u_pred, mu, gamma, potential_type, None, prev_prediction, mode)
                boundary_loss = model.boundary_loss(boundary_points, boundary_values, prev_prediction, mode)
                norm_loss = (torch.norm(full_u, p=2) - 1)**2
                sym_loss = model.symmetry_loss(X_tensor, lb_tensor, ub_tensor)

                total_loss = (
                        weights[0] * pde_loss +
                        weights[1] * norm_loss +
                        weights[2] * sym_loss +
                        weights[3] * boundary_loss
                )
                total_loss.backward()
                optimizer.step()
                scheduler.step()

                if epoch % 1000 == 0:
                    print(f"Epoch {epoch}, Loss: {total_loss.item():.6f}, γ: {gamma:.4f}")

            gamma_logs.append((mu, gamma))
            models_by_mu[mu] = model
            prev_prediction = make_prev_prediction(model, mode)

        gamma_table[mode] = gamma_logs
        models_by_mode[mode] = models_by_mu

    # After training, calculate actual gamma values for each model
    gamma_values = {}
    for mode in modes:
        gamma_values[mode] = []
        for mu in mu_values:
            model = models_by_mode[mode][mu]
            X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
            u_pred = model.forward(X_tensor)
            full_u = model.get_complete_solution(X_tensor, u_pred, mode)
            gamma = compute_gamma(full_u)
            gamma_values[mode].append(gamma)

    return models_by_mode, gamma_table, gamma_values


def plot_figure11_corrected(models_by_mode, X_test, mode=0, save_dir="plots"):
    """Plot Figure 11 from the paper following their exact procedure"""
    os.makedirs(save_dir, exist_ok=True)
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    dx = X_test[1, 0] - X_test[0, 0]  # Spatial step size

    plt.figure(figsize=(10, 6))

    # Get the gammas for this mode from the model dictionary
    gammas = sorted(list(models_by_mode[mode].keys()))

    # Plot analytical solution for γ=0 first (if available in models)
    if 0.0 in models_by_mode[mode]:
        # Use the model trained with gamma=0
        model = models_by_mode[mode][0.0]
        model.eval()
        with torch.no_grad():
            u_pred = model.forward(X_tensor)
            full_u = model.get_complete_solution(X_tensor, u_pred, mode)
            wh_tensor = full_u.cpu().numpy().flatten()
            # Normalize properly with integration
            wh_tensor /= np.sqrt(np.sum(wh_tensor ** 2) * dx)
            plt.plot(X_test, wh_tensor ** 2, 'k-', label='γ=0')
    else:
        # Fallback to using theoretical Hermite function
        dummy_model = list(models_by_mode[mode].values())[0]
        wh_tensor = dummy_model.weighted_hermite(X_tensor, mode).detach().cpu().numpy().flatten()
        # Normalize properly with integration
        wh_tensor /= np.sqrt(np.sum(wh_tensor ** 2) * dx)
        plt.plot(X_test, wh_tensor ** 2, 'k-', label='γ=0')

    # Different line styles for different gamma values
    linestyles = ['-', '--', '-.', ':', '-', '--']
    colors = ['k', 'b', 'r', 'g', 'm', 'c']

    # Plot numerical solutions for γ > 0
    for i, gamma in enumerate(gammas):
        if np.isclose(gamma, 0.0):
            continue  # Skip, already plotted

        model = models_by_mode[mode][gamma]
        model.eval()
        with torch.no_grad():
            u_pred = model.forward(X_tensor)
            full_u = model.get_complete_solution(X_tensor, u_pred, mode)
            u_np = full_u.cpu().numpy().flatten()

            # Normalize properly with integration
            u_np /= np.sqrt(np.sum(u_np ** 2) * dx)

            plt.plot(X_test, u_np ** 2, linestyle=linestyles[(i - 1) % len(linestyles)],
                     color=colors[(i - 1) % len(colors)], label=f"γ={gamma:.1f}")

    plt.title(f"Mode {mode} solution densities", fontsize=14)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("|ψ(x)|²", fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.xlim(-10, 10)  # Match the paper's x-range

    # Add a tight layout to make it look nicer
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, f"figure11_mode_{mode}.png"), dpi=300)
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


if __name__ == "__main__":
    lb, ub = -10, 10  # Domain boundaries
    N_f = 4000  # Number of collocation points
    N_u = 200  # Number of boundary points
    epochs = 5001  # More epochs for better convergence
    layers = [1, 100, 100, 100, 1]  # Neural network architecture

    X = np.linspace(lb, ub, N_f).reshape(-1, 1)
    X_test = np.linspace(lb, ub, 1000).reshape(-1, 1)  # Higher resolution for plotting

    # Use gamma values directly from Figure 11 in the paper
    gamma_values = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]  # Exact values from the paper
    modes = [0, 1, 2, 3]

    # Adjusted weights for better convergence
    weights = [1.0, 5.0, 1.0, 10.0, 1.0]  # [data_loss, riesz_loss, pde_loss, norm_loss, sym_loss]

    # Train with fixed gamma values directly
    models_by_mode, mu_table = train_with_fixed_gamma(
        gamma_values, modes, X, N_u, N_f, lb, ub, layers, epochs, weights)

    # Plot Figure 11 (mode 0)
    plot_figure11_corrected(models_by_mode, X_test, mode=0)

    # Plot Figure 2 (Thomas-Fermi comparison with high gamma)
    plot_figure2(models_by_mode, X_test.flatten(), gamma_values, mu_table, save_dir="plots")

    # Plot mu vs gamma as in the paper
    plt.figure(figsize=(8, 6))
    for mode, logs in mu_table.items():
        gamma_list, mu_list = zip(*logs)
        plt.plot(gamma_list, mu_list, 'o-', label=f"Mode {mode}")

    plt.xlabel("γ")
    plt.ylabel("μ")
    plt.title("Chemical potential vs. interaction strength")
    plt.grid(True)
    plt.legend()
    plt.savefig("plots/mu_vs_gamma.png", dpi=300)
    plt.show()

