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

    def __init__(self, layers, hbar=1.0, m=1.0, mode=0, beta_init=1.0, alpha_init=1.0, decay_rate=0.001):
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
        u = self.get_complete_solution(inputs, predictions, prev_prediction, mode)

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

        residual = -u_xx + V * u + gamma * u ** 3 - mu * u
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

    def compute_thomas_fermi_approx(self, lambda_pde, potential, eta):
        """
        Calculate the Thomas–Fermi approximation for the given potential.

        Parameters
        ----------
        lambda_pde : float
            Eigenvalue from lowest energy ground state.
        potential : torch.Tensor
            Potential values corresponding to the spatial coordinates.
        eta : float
            Interaction strength.

        Returns
        -------
        torch.Tensor
            Thomas–Fermi approximation of the wave function.
        """
        # For eta = 0, return None (TF approximation not valid)
        if eta == 0:
            return None

        # Calculate TF approximation: ψ_TF(x) = sqrt(max(0, (λ - V(x))/η))
        # tf_approx = torch.sqrt(torch.relu((lambda_pde - potential) / eta))
        tf_approx = torch.sqrt((lambda_pde - potential) / eta)

        return tf_approx

    def get_complete_solution(self, x, perturbation, prev_prediction=None, mode=0):
        if prev_prediction is None:
            base_solution = self.weighted_hermite(x, mode)
        else:
            base_solution = prev_prediction(x)

        # return self.alpha * (base_solution + perturbation)
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
        u = self.get_complete_solution(boundary_points, u_pred, prev_prediction, mode)

        return torch.mean((u - boundary_values) ** 2)

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

        u = self.get_complete_solution(inputs, predictions, prev_prediction, mode)

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

    def pde_loss(self, inputs, predictions, eta, potential_type, precomputed_potential=None, prev_prediction=None, mode=0):
        """
        Compute the PDE loss for the Gross-Pitaevskii equation.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of spatial coordinates (collocation points).
        predictions : torch.Tensor
            Predicted solution from the network.
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
        tuple
            Tuple containing:
                - torch.Tensor: PDE loss value.
                - torch.Tensor: PDE residual.
                - torch.Tensor: Smallest eigenvalue (lambda).
        """
        u = self.get_complete_solution(inputs, predictions, prev_prediction, mode)

        # Compute first and second derivatives with respect to x
        u_x = grad(u, inputs, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = grad(u_x, inputs, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

        # Compute λ from the energy functional
        if precomputed_potential is not None:
            V = precomputed_potential
        else:
            V = self.compute_potential(inputs, potential_type)
        lambda_pde = torch.mean(u_x ** 2 + V * u ** 2 + eta * u ** 4) / torch.mean(u ** 2)

        # Residual of the 1D Gross-Pitaevskii equation
        if precomputed_potential is not None:
            V = precomputed_potential
        else:
            V = self.compute_potential(inputs, potential_type)
        pde_residual = -u_xx + V * u + eta * torch.abs(u ** 2) * u - lambda_pde * u

        # Regularization: See https://arxiv.org/abs/2010.05075

        # Term 1: L_f = 1 / (f(x, λ))^2, penalizes the network if the PDE residual is close to zero to avoid trivial eigenfunctions
        L_f = 1 / (torch.mean(u ** 2) + 1e-2)

        # Term 2: L_λ = 1 / λ^2, penalizes small eigenvalues λ, ensuring non-trivial eigenvalues
        L_lambda = 1 / (lambda_pde ** 2 + 1e-6)

        # Term 3: L_drive = e^(-λ + c), encourages λ to grow, preventing collapse to small values
        L_drive = torch.exp(-lambda_pde + 1.0)

        # PDE loss (residual plus regularization terms)
        pde_loss = torch.mean(pde_residual ** 2)  #+ L_lambda + L_f

        return pde_loss, pde_residual, lambda_pde

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
            return model.get_complete_solution(x, u_pred, None, mode)
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
                full_u = model.get_complete_solution(X_tensor, u_pred, prev_prediction, mode)
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

    return models_by_mode, gamma_table


def plot_figure11(models_by_mode, mu_values, X_test, modes, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    X_tensor = torch.tensor(X_test, dtype=torch.float32, requires_grad=True)

    for mode in modes:
        plt.figure(figsize=(10, 6))

        # Plot weighted Hermite approximation once
        dummy_model = list(models_by_mode[mode].values())[0]  # Use any model to access .weighted_hermite
        wh_tensor = dummy_model.weighted_hermite(X_tensor, mode).detach().cpu().numpy().flatten()
        wh_tensor /= np.sqrt(np.sum(wh_tensor ** 2) * (X_test[1] - X_test[0]))
        plt.plot(X_test, wh_tensor, linestyle='--', color='black', label='Weighted Hermite')

        for i, mu in enumerate(mu_values):
            model = models_by_mode[mode][mu]
            model.eval()
            u_pred = model.forward(X_tensor)
            full_u = model.get_complete_solution(X_tensor, u_pred, None, mode)
            u_np = full_u.detach().cpu().numpy().flatten()
            dx = X_test[1] - X_test[0]
            u_np = u_np / np.sqrt(np.sum(u_np ** 2) * dx)
            plt.plot(X_test, u_np, label=f"μ={mu:.2f}")

        plt.title(f"Mode {mode} solutions Ψ(x)", fontsize=14)
        plt.xlabel("x")
        plt.ylabel("Ψ(x)")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(save_dir, f"figure11_mode_{mode}.png"), dpi=300)
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
    lb, ub = -10, 10
    N_f = 4000
    N_u = 200
    epochs = 5000
    layers = [1, 100, 100, 100, 1]

    X = np.linspace(lb, ub, N_f).reshape(-1, 1)
    X_test = np.linspace(lb, ub, N_f).reshape(-1, 1)

    mu_values = [10, 20, 30, 40, 50]
    # modes = list(range(0, 8))  # Mode 0 to 7
    modes = [0]
    weights = [1.0, 50.0, 20.0, 200.0]

    models_by_mode, gamma_table = train_with_mu_and_mode(mu_values, modes, X, N_u, N_f, lb, ub, layers, epochs, weights)

    plot_gamma_table(gamma_table)
    plot_figure11(models_by_mode, mu_values, X_test, modes)


