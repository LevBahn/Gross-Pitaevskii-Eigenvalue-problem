import torch
import torch.nn as nn
import pickle
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import math
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.special import hermite
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['text.usetex'] = False  # Disable LaTeX rendering

plot_params = {
    "figure.dpi": "300",
    "axes.labelsize": 20,
    "axes.linewidth": 1.5,
    "axes.titlesize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.title_fontsize": 14,
    "legend.fontsize": 16,
    "xtick.major.size": 3.5,
    "xtick.major.width": 1.5,
    "xtick.minor.size": 2.5,
    "xtick.minor.width": 1.5,
    "ytick.major.size": 3.5,
    "ytick.major.width": 1.5,
    "ytick.minor.size": 2.5,
    "ytick.minor.width": 1.5,
}

plt.rcParams.update(plot_params)


class ShiftedTanh(nn.Module):
    """ShiftedTanh(x) = tanh(x) + 1 + eps"""

    def __init__(self, eps=np.finfo(float).eps):
        super(ShiftedTanh, self).__init__()
        self.eps = eps

    def forward(self, x):
        return torch.tanh(x) + 1.0 + self.eps


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)

    def forward(self, x):
        identity = x
        out = torch.tanh(self.lin1(x))
        out = self.lin2(out)
        return torch.tanh(out + identity)


class GrossPitaevskiiPINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) for solving the 1D Gross-Pitaevskii Equation.
    """

    def __init__(self, layers, hbar=1.0, m=1.0, mode=0, gamma=1.0, L=1.0, use_residual=True):
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
        gamma : float, optional
            Interaction strength parameter.
        L : float, optional
            Length of the box (default is 1.0).
        """
        super().__init__()
        self.layers = layers
        self.use_residual = use_residual
        self.network = self.build_network()
        self.hbar = hbar  # Planck's constant, fixed
        self.m = m  # Particle mass, fixed
        self.mode = mode  # Mode number (n)
        self.gamma = gamma  # Interaction strength parameter
        self.L = L  # Length of the box

        # Now build the network after all attributes are set
        self.network = self.build_network()

    def build_network(self):
        """
        Build the neural network with shifted tanh activation function and residual connections.
        """
        if not self.use_residual:
            # Original architecture without residual blocks
            layers = []
            for i in range(len(self.layers) - 1):
                layers.append(nn.Linear(self.layers[i], self.layers[i + 1]))
                if i < len(self.layers) - 2:
                    layers.append(ShiftedTanh())
            return nn.Sequential(*layers)
        else:
            # New architecture with residual blocks
            modules = []

            # Input layer
            input_dim = self.layers[0]
            hidden_dim = self.layers[1]
            modules.append(nn.Linear(input_dim, hidden_dim))
            modules.append(ShiftedTanh())

            # Residual blocks in the middle layers
            num_res_blocks = len(self.layers) - 3
            for _ in range(num_res_blocks):
                modules.append(ResidualBlock(hidden_dim))

            # Output layer
            modules.append(nn.Linear(hidden_dim, self.layers[-1]))

            return nn.Sequential(*modules)

    def box_eigenfunction(self, x, n):
        """
        Compute the analytic eigenfunction for a particle in a box.

        For the linear case (gamma = 0), the solution is:
        phi_n(x) = sqrt(2/L) * sin(n*pi*x/L)

        This corresponds to equation (22) in the paper.
        """
        # For mode 0, n=1 in the sine function per equation (22)
        n_actual = n + 1  # Convert mode number to quantum number (n=0 → first excited state with n_actual=1)

        # Normalization factor
        norm_factor = torch.sqrt(torch.tensor(2.0 / self.L))

        # Sine function with proper scaling
        phi_n = norm_factor * torch.sin(n_actual * torch.pi * x / self.L)

        return phi_n

    def forward(self, inputs):
        """
        Forward pass through the neural network.
        """
        return self.network(inputs)

    def get_complete_solution(self, x, perturbation, mode=None):
        """
        Get the complete solution by combining the base Hermite solution with the neural network perturbation.
        """
        if mode is None:
            mode = self.mode
        base_solution = self.box_eigenfunction(x, mode)
        return base_solution + perturbation

    def compute_potential(self, x, potential_type="gaussian", **kwargs):
        """
        Compute potential function for the 1D domain.
        """
        if potential_type == "gaussian":
            V = torch.exp(-(x - 0.5)**2)
        else:
            raise ValueError(f"Unknown potential type: {potential_type}")
        return V

    def pde_loss(self, inputs, predictions, gamma, p, potential_type="gaussian", precomputed_potential=None):
        """
        Compute the PDE loss for the Gross-Pitaevskii equation.
        μψ = - ∇²ψ + Vψ + γ|ψ|²ψ
        """
        # Get the complete solution (base + perturbation)
        u = self.get_complete_solution(inputs, predictions)

        # Compute derivatives with respect to x
        u_x = torch.autograd.grad(
            outputs=u,
            inputs=inputs,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]

        u_xx = torch.autograd.grad(
            outputs=u_x,
            inputs=inputs,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True
        )[0]

        # Compute potential
        if precomputed_potential is not None:
            V = precomputed_potential
        else:
            V = self.compute_potential(inputs, potential_type)

        # Calculate chemical potential
        kinetic = -u_xx
        potential = V * u
        #interaction = gamma * u ** 3
        interaction = gamma * u ** p

        numerator = torch.mean(u * (kinetic + potential + interaction))
        denominator = torch.mean(u ** 2)
        lambda_pde = numerator / denominator

        # Residual of the 1D Gross-Pitaevskii equation
        pde_residual = kinetic + potential + interaction - lambda_pde * u

        # PDE loss (mean squared residual)
        pde_loss = torch.mean(pde_residual ** 2)

        return pde_loss, lambda_pde

    def boundary_loss(self, boundary_points, boundary_values):
        """
        Compute the boundary loss for the boundary conditions.
        """
        u_pred = self.forward(boundary_points)
        full_u = self.get_complete_solution(boundary_points, u_pred)
        return torch.mean((full_u - boundary_values) ** 2)

    def normalization_loss(self, u, dx):
        """
        Compute normalization loss using proper numerical integration.
        """
        integral = torch.sum(u ** 2) * dx
        return (integral - 1.0) ** 2


def train_gpe_model(gamma_values, modes, p, X_train, lb, ub, layers,
                    epochs, tol, perturb_const,
                    potential_type='gaussian', lr=1e-5, verbose=True):
    """
    Train the GPE model for different modes and gamma values.

    Parameters:
    -----------
    gamma_values : list of float
        List of interaction strengths to train models for
    modes : list of int
        List of modes to train (0, 1, 2, 3, etc.)
    p : int
        Parameter for nonlinearity power
    X_train : numpy.ndarray
        Training points array
    lb, ub : float
        Lower and upper boundaries of the domain
    layers : list of int
        Network architecture
    epochs : int
        Number of training epochs
    potential_type : str
        Type of potential ('harmonic', 'box', 'gaussian', etc.)
    tol : float
        The desired tolerance for early stopping
    perturb_const : float
        A small constant representing the perturbation parameter
    lr : float
        Learning rate
    verbose : bool
        Whether to print training progress

    Returns:
    --------
    tuple: (models_by_mode, mu_table, training_history)
        Trained models organized by mode and gamma, chemical potential values, and training histories
    """
    # Convert training data to tensors
    dx = X_train[1, 0] - X_train[0, 0]
    X_tensor = torch.tensor(X_train, dtype=torch.float32, requires_grad=True).to(device)

    # Create boundary conditions
    L = ub
    boundary_points = torch.tensor([[lb], [ub]], dtype=torch.float32).to(device)
    boundary_values = torch.zeros((2, 1), dtype=torch.float32).to(device)

    # Track models, chemical potentials, training history, and epochs until stopping
    models_by_mode = {}
    mu_table = {}
    training_history = {}
    constant_history = {}
    epochs_history = {}

    # Sort gamma values
    gamma_values = sorted(gamma_values)

    print(f"Tolerance : {tol}, Perturbation constant : {perturb_const}")

    for mode in modes:
        if verbose:
            print(f"\n===== Training for mode {mode} =====")

        mu_logs = []
        models_by_gamma = {}
        history_by_gamma = {}
        epochs_by_gamma = {}

        prev_model = None

        for gamma in gamma_values:

            if verbose:
                print(f"\nTraining for γ = {gamma:.2f}, mode = {mode}, nonlinearity p = {p}")

            # Initialize model for this mode and gamma
            model = GrossPitaevskiiPINN(layers, mode=mode, gamma=gamma, L=L).to(device)
            # If this isn't the first gamma value, initialize with previous model's weights
            if prev_model is not None:
                model.load_state_dict(prev_model.state_dict())
            else:
                # Use the advanced initialization that considers mode number
                model.apply(lambda m: advanced_initialization(m, mode))

            # Adam optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            # Create scheduler to decrease learning rate during training
            if p < 20:
                scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=200, T_mult=2, eta_min=1e-6)
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8)

            # Track learning history
            lambda_history = []
            loss_history = []
            constraint_history = []

            # Early stopping variables
            best_loss = float('inf')
            best_model_state = None
            patience_counter = 0
            patience = 2000  # Number of epochs to wait without improvement
            final_epoch = epochs  # Track the actual final epoch

            for epoch in range(epochs):
                optimizer.zero_grad()

                # Forward pass
                u_pred = model.forward(X_tensor)
                if epoch == 0 and gamma == 0:
                    normal_const = torch.max(u_pred).detach().clone()
                    constant_history[mode] = normal_const
                    u_pred = u_pred / normal_const
                    u_pred = perturb_const * u_pred
                else:
                    u_pred = perturb_const * u_pred
                    u_pred = u_pred / normal_const

                # Calculate common constraint losses for all modes
                boundary_loss = model.boundary_loss(boundary_points, boundary_values)
                norm_loss = model.normalization_loss(model.get_complete_solution(X_tensor, u_pred), dx)

                # Combined constraint loss
                constraint_loss = 10.0 * boundary_loss + 20.0 * norm_loss

                # Use PDE residual for all modes
                pde_loss, lambda_value = model.pde_loss(X_tensor, u_pred, gamma, p, potential_type)
                physics_loss = pde_loss
                loss_type = "PDE residual"

                # Total loss for optimization
                total_loss = physics_loss + constraint_loss

                # Backpropagate
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
                optimizer.step()
                scheduler.step(total_loss)

                # Record current loss for early stopping
                current_loss = total_loss.item()

                # Check if this is the best model so far
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Record loss frequently
                if epoch % 10 == 0:
                    loss_history.append(current_loss)

                # Record history
                if epoch % 100 == 0:
                    lambda_history.append(lambda_value.item())
                    constraint_history.append(constraint_loss.item())

                    if verbose and epoch % 500 == 0:
                        print(f"Epoch {epoch}, μ: {lambda_value.item():.4f}")
                        print(f"Total Loss: {current_loss:.6f}, {loss_type}: {physics_loss.item():.6f}, "
                              f"Constraints: {constraint_loss.item():.6f}")

                # Early stopping conditions
                if current_loss <= tol:
                    if verbose:
                        print(f"Early stop: tolerance reached at epoch {epoch}, loss: {current_loss}")
                    final_epoch = epoch
                    break

                if patience_counter >= patience:
                    if verbose:
                        print(
                            f"Early stop: no improvement for {patience} epochs at epoch {epoch}, best loss: {best_loss}")
                    final_epoch = epoch
                    break

            # Load the best model found during training
            if best_model_state is not None:
                model.load_state_dict(best_model_state)

            # Record final chemical potential and save model
            final_mu = lambda_history[-1] if lambda_history else 0
            mu_logs.append((gamma, final_mu))
            models_by_gamma[gamma] = model

            # Save the training history
            history_by_gamma[gamma] = {
                'loss': loss_history,
                'constraint': constraint_history,
                'lambda': lambda_history
            }

            # Save the number of epochs until stopping
            epochs_by_gamma[gamma] = final_epoch

            # Update prev_model for next gamma value
            prev_model = model

        # Store results for this mode
        mu_table[mode] = mu_logs
        models_by_mode[mode] = models_by_gamma
        training_history[mode] = history_by_gamma
        epochs_history[mode] = epochs_by_gamma

    return models_by_mode, mu_table, training_history, constant_history, epochs_history


def plot_wavefunction(models_by_mode, X_test, gamma_values,
                      modes, p, constant_history, perturb_const, potential_type, lb, ub,
                      save_dir="Gross-Pitaevskii/src/final/refine/test"):
    """
    Plot wavefunctions for different modes and gamma values.
    """
    os.makedirs(save_dir, exist_ok=True)

    for mode in modes:
        if mode not in models_by_mode:
            continue

        plt.figure(figsize=(8, 6))

        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        dx = X_test[1, 0] - X_test[0, 0]

        linestyles = ['-', '--', '-.', ':', '-', '--']
        colors = ['k', 'b', 'r', 'g', 'm', 'c', 'slategray']

        for j, gamma in enumerate(gamma_values):
            if gamma not in models_by_mode[mode]:
                continue

            model = models_by_mode[mode][gamma]
            model.eval()
            const = constant_history[mode]

            with torch.no_grad():
                u_pred = model.forward(X_tensor)
                u_pred = u_pred * (perturb_const / const)
                full_u = model.get_complete_solution(X_tensor, u_pred)
                u_np = full_u.cpu().numpy().flatten()

                # Normalization
                u_np /= np.sqrt(np.sum(u_np ** 2) * dx)

                # For mode 0, ensure all wavefunctions are positive
                if mode == 0:
                    u_np = np.abs(u_np)

                # Plot wavefunctions
                if gamma % 20 == 0:
                    plt.plot(X_test.flatten(), u_np,
                             linestyle=linestyles[j % len(linestyles)],
                             color=colors[j % len(colors)],
                             label=f"η={gamma:.1f}")

        plt.title(f"Mode {mode} Wavefunction", fontsize=18)
        plt.xlabel("x", fontsize=18)
        plt.ylabel(r"$u(x)$", fontsize=18)
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.xlim(lb, ub)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"mode_{mode}_wavefunction_p{p}_{potential_type}.png"), dpi=300)
        plt.close()

    # Also create a combined grid figure to show all modes
    plot_combined_grid(models_by_mode, X_test, gamma_values, modes, p,
                       constant_history, perturb_const, potential_type, lb, ub, save_dir)


def plot_combined_grid(models_by_mode, X_test, gamma_values, modes, p,
                       constant_history, perturb_const, potential_type, lb, ub,
                       save_dir="Gross-Pitaevskii/src/final/refine/test"):
    """
    Create a grid of subplots showing all modes.
    """
    n_modes = len(modes)
    n_cols = min(4, n_modes)
    n_rows = (n_modes + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))

    if n_rows > 1 or n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    dx = X_test[1, 0] - X_test[0, 0]

    linestyles = ['-', '--', '-.', ':', '-', '--']
    colors = ['k', 'b', 'r', 'g', 'm', 'c', 'slategray']

    for i, mode in enumerate(modes):
        if i >= len(axes) or mode not in models_by_mode:
            continue

        ax = axes[i]

        for j, gamma in enumerate(gamma_values):
            if gamma not in models_by_mode[mode]:
                continue

            model = models_by_mode[mode][gamma]
            model.eval()
            const = constant_history[mode]

            with torch.no_grad():
                u_pred = model.forward(X_tensor)
                u_pred = u_pred * (perturb_const / const)
                full_u = model.get_complete_solution(X_tensor, u_pred)
                u_np = full_u.cpu().numpy().flatten()

                # Normalization
                u_np /= np.sqrt(np.sum(u_np ** 2) * dx)

                # For mode 0, ensure all wavefunctions are positive
                if mode == 0:
                    u_np = np.abs(u_np)

                # Plot the wavefunction
                if gamma % 20 == 0:
                    ax.plot(X_test.flatten(), u_np,
                            linestyle=linestyles[j % len(linestyles)],
                            color=colors[j % len(colors)],
                            label=f"γ={gamma:.1f}")

        ax.set_title(f"mode {mode}", fontsize=12)
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel(r"$u(x)$", fontsize=12)
        ax.grid(True)
        ax.legend(fontsize=6)
        ax.set_xlim(lb, ub)

    # Hide any unused subplots
    for i in range(len(modes), len(axes)):
        axes[i].axis('off')

    fig.suptitle(f"Wavefunctions for All Modes (p={p})", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(save_dir, f"all_modes_combined_wavefunctions_p{p}_{potential_type}.png"), dpi=300)
    plt.close(fig)


def plot_lambda_vs_eta(mu_table, modes, p, potential_type, save_dir="Gross-Pitaevskii/src/final/refine/test"):
    """
    Plot chemical potential vs. interaction strength for different modes.
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))

    markers = ['o', 's', '^', 'v', 'D', 'x', '*', '+']
    colors = ['k', 'b', 'r', 'g', 'm', 'c', 'orange', 'purple']

    # Plot μ vs γ for each mode
    for i, mode in enumerate(modes):
        if mode not in mu_table:
            continue

        gamma_list, mu_list = zip(*mu_table[mode])
        plt.plot(mu_list, gamma_list,
                 marker=markers[i % len(markers)],
                 color=colors[i % len(colors)],
                 linestyle='-',
                 label=f"Mode {mode}")

    plt.ylabel(r"$\eta$ (Interaction Strength)", fontsize=18)
    plt.xlabel(r"$\lambda$ (Chemical Potential)", fontsize=18)
    plt.title(f"Chemical Potential vs. Interaction Strength for All Modes (p={p})", fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"mu_vs_gamma_all_modes_p{p}_{potential_type}.png"), dpi=300)
    plt.close()


def advanced_initialization(m, mode):
    """Initialize network weights with consideration of the mode number"""
    if isinstance(m, nn.Linear):
        # Use Xavier initialization but scale based on mode
        gain = 1.0 / (1.0 + 0.2 * mode)  # Stronger scaling for higher modes
        nn.init.xavier_normal_(m.weight, gain=gain)  # Normal instead of uniform

        # More careful bias initialization for higher modes
        if mode > 3:
            m.bias.data.fill_(0.001)  # Smaller initial bias for higher modes
        else:
            m.bias.data.fill_(0.01)


def plot_all_modes_gamma_loss(training_history, modes, gamma_values, epochs, p, potential_type,
                              save_dir="Gross-Pitaevskii/src/final/refine/test"):
    """
    Plot the training loss history for all modes and all gamma values with each mode in its own subplot.

    Parameters:
    -----------
    training_history : dict
        Dictionary containing training history for all modes and gamma values
    modes : list
        List of modes to include in the plot
    gamma_values : list
        List of gamma values to include in the plot
    epochs : int
        Total number of training epochs
    save_dir : str
        Directory to save the plots
    """
    os.makedirs(save_dir, exist_ok=True)

    n_modes = len(modes)
    n_cols = min(4, n_modes)
    n_rows = (n_modes + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))

    if n_rows > 1 or n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    linestyles = ['-', '--', '-.', ':', '-', '--']
    colors = ['k', 'b', 'r', 'g', 'm', 'c', 'slategray']

    for i, mode in enumerate(modes):
        if i >= len(axes) or mode not in training_history:
            continue

        ax = axes[i]

        for j, gamma in enumerate(gamma_values):
            if gamma in training_history[mode]:
                loss_history = training_history[mode][gamma]['loss']

                epoch_nums = np.linspace(0, epochs, len(loss_history))

                # Use log scale
                if gamma % 20 == 0:
                    ax.semilogy(epoch_nums, loss_history,
                                color=colors[j % len(colors)],
                                linestyle=linestyles[j % len(linestyles)],
                                label=f"γ={gamma:.1f}")

        ax.set_title(f"mode {mode}", fontsize=12)
        ax.set_xlabel("Epochs", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.grid(True)
        ax.legend(fontsize=6)

    # Hide any unused subplots
    for i in range(len(modes), len(axes)):
        axes[i].axis('off')

    fig.suptitle("Training Loss for All Modes", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(save_dir, f"all_modes_gamma_loss_subplots_p{p}_{potential_type}.png"), dpi=300)
    plt.close(fig)


def plot_improved_loss_visualization(training_history, modes, gamma_values, epochs, p, potential_type,
                                     save_dir="Gross-Pitaevskii/src/final/refine/test"):
    """
    Creates informative and smoother visualizations of the training progress.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Plot overall training progress across modes
    plt.figure(figsize=(12, 8))
    max_mode = max(modes) if modes else 0
    colormap = plt.cm.viridis
    colors = [colormap(i / max_mode) for i in modes]

    for i, mode in enumerate(modes):
        for gamma in [0.0]:
            if mode in training_history and gamma in training_history[mode]:
                loss_history = training_history[mode][gamma]['loss']

                window_size = min(5, len(loss_history) // 5)
                if window_size > 1:
                    ultra_smooth_loss = moving_average(loss_history, window_size)
                    epoch_nums = np.linspace(0, epochs, len(ultra_smooth_loss))
                    plt.semilogy(epoch_nums, ultra_smooth_loss,
                                 color=colors[i],
                                 linewidth=2.0,
                                 label=f"Mode {mode}")
                else:
                    epoch_nums = np.linspace(0, epochs, len(loss_history))
                    plt.semilogy(epoch_nums, loss_history,
                                 color=colors[i],
                                 linewidth=2.0,
                                 label=f"Mode {mode}")

    plt.title("Training Progress for All Modes", fontsize=20)
    plt.xlabel("Epochs", fontsize=18)
    plt.ylabel("Loss", fontsize=18)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    # Adjust the line below as needed
    plt.savefig(os.path.join(save_dir, f"loss_history_training_progress_p{p}_{potential_type}.png"), dpi=300)
    plt.close()


def moving_average(values, window_size=10):
    """Apply moving average smoothing to a list of values"""
    if len(values) < window_size:
        return values
    weights = np.ones(window_size) / window_size
    return np.convolve(values, weights, mode='valid')


def save_models(models_by_mode, mu_table, training_history, constant_history, epochs_history,
                filename="gpe_models.pkl", save_dir="Gross-Pitaevskii/src/final/refine/test"):
    """Save all training results to a single file."""
    # Convert models to CPU and save state dicts
    models_state_dicts = {}
    for mode in models_by_mode:
        models_state_dicts[mode] = {}
        for gamma, model in models_by_mode[mode].items():
            models_state_dicts[mode][gamma] = {
                'state_dict': model.cpu().state_dict(),
                'layers': model.layers,
                'hbar': model.hbar,
                'm': model.m,
                'mode': model.mode,
                'gamma': model.gamma
            }

    data = {
        'models_state_dicts': models_state_dicts,
        'mu_table': mu_table,
        'training_history': training_history,
        'constant_history': constant_history,
        'epochs_history': epochs_history
    }

    filepath = os.path.join(save_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"Models saved to {filepath}")


def load_models(filename="gpe_models.pkl", save_dir="Gross-Pitaevskii/src/final/refine/test"):
    """Load all training results from file and reconstruct models properly."""
    filepath = os.path.join(save_dir, filename)
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    # Reconstruct models from state dicts
    models_by_mode = {}
    for mode in data['models_state_dicts']:
        models_by_mode[mode] = {}
        for gamma, model_data in data['models_state_dicts'][mode].items():
            # Recreate model
            model = GrossPitaevskiiPINN(
                layers=model_data['layers'],
                hbar=model_data['hbar'],
                m=model_data['m'],
                mode=model_data['mode'],
                gamma=model_data['gamma']
            ).to(device)

            # Load trained weights
            model.load_state_dict(model_data['state_dict'])
            model.eval()

            models_by_mode[mode][gamma] = model

    print(f"Models loaded from {filename}")
    return (models_by_mode, data['mu_table'], data['training_history'],
            data['constant_history'], data['epochs_history'])


if __name__ == "__main__":
    # Setup parameters
    lb, ub = 0, 1 # Domain boundaries
    N_f = 4000  # Number of collocation points
    epochs = 5001
    layers = [1, 64, 64, 64, 1]  # Neural network architecture

    # Create uniform grid for training and testing
    X = np.linspace(lb, ub, N_f).reshape(-1, 1)
    X_test = np.linspace(lb, ub, 1000).reshape(-1, 1)

    # Gamma values from the paper
    alpha = 0.5
    gamma_values = [k * alpha for k in range(201)]
    # gamma_values = [k * alpha for k in range(51)]
    #gamma_values = [k * alpha for k in range(21)]

    # Include modes 0 through 5
    modes = [0]

    # Set the perturbation constant
    perturb_const = 0.01  # q in paper

    # Set the tolerance
    tol = 0.00001

    # Nonlinearity powers
    nonlinearity_powers = [16]

    for p in nonlinearity_powers:

        # Specify potential type
        potential_type = "gaussian"

        # Train neural network or load existing models
        train_new = True  # Set to True to train, False to load
        filename = f"my_gpe_models_p{p}_{potential_type}_pert_const_1e-2_tol_{tol}_length_{ub}.pkl"

        # Create plotting and model saving directory
        p_save_dir = f"plots_p{p}_{potential_type}_paper_test_length_{ub}"
        os.makedirs(p_save_dir, exist_ok=True)

        if train_new:
            # Train models
            print("Starting training...")
            models_by_mode, mu_table, training_history, constant_history, epochs_history = train_gpe_model(
                gamma_values, modes, p, X, lb, ub, layers, epochs, tol, perturb_const,
                potential_type='gaussian', lr=1e-3, verbose=True)

            # Save results
            save_models(models_by_mode, mu_table, training_history, constant_history, epochs_history, filename, p_save_dir)
        else:
            # Load existing models
            print("Loading existing models...")
            models_by_mode, mu_table, training_history, constant_history, epochs_history = load_models(filename, p_save_dir)

        # Plot wavefunctions for individual modes
        print("Generating individual mode plots...")
        plot_wavefunction(models_by_mode, X_test, gamma_values, modes, p, constant_history, perturb_const,
                          potential_type, lb, ub, p_save_dir)

        # Plot μ vs γ for all modes
        print("Generating lambda vs. eta plot...")
        plot_lambda_vs_eta(mu_table, modes, p, potential_type, p_save_dir)

        # Plot combined loss history
        print("Generating combined loss plots...")
        # plot_improved_loss_visualization(training_history, modes, gamma_values, epochs, p, potential_type, p_save_dir)
        plot_all_modes_gamma_loss(training_history, modes, gamma_values, epochs, p, potential_type, p_save_dir)

        print(f"Results saved to: {p_save_dir}/")