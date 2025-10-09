import torch
import torch.nn as nn
import pickle
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import math
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.special import airy, ai_zeros, hermite
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
    """Custom activation: tanh(x) + 1 + eps"""

    def __init__(self, eps=np.finfo(float).eps):
        super(ShiftedTanh, self).__init__()
        self.eps = eps

    def forward(self, x):
        return torch.tanh(x) + 1.0 + self.eps


class GrossPitaevskiiPINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) for solving the 1D Gross-Pitaevskii Equation.
    """

    def __init__(self, layers, hbar=1.0, m=1.0, mode=0, gamma=1.0, use_residual=True, use_perturbation=True):
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
        use_perturbation: boolean, optional
            Get the complete solution by combining the base solution with the neural network perturbation. Default is
            True.
        """
        super().__init__()
        self.layers = layers
        self.use_residual = use_residual
        self.network = self.build_network()
        self.hbar = hbar  # Planck's constant, fixed
        self.m = m  # Particle mass, fixed
        self.mode = mode  # Mode number (n)
        self.gamma = gamma  # Interaction strength parameter
        self.use_perturbation = use_perturbation  # Combine the base solution with the neural network perturbation

    def build_network(self):
        """
        Build the neural network with tanh activation functions between layers.
        """
        layers = []
        for i in range(len(self.layers) - 1):
            layers.append(nn.Linear(self.layers[i], self.layers[i + 1]))
            if i < len(self.layers) - 2:
                layers.append(ShiftedTanh())
        return nn.Sequential(*layers)

    def airy_solution(self, x, n):
        """
        Linear eigenfunction: Ψ_n(x) = Ai(x + α_n) where α_n is the n-th Airy zero
        """

        # Convert to numpy
        x_np = x.detach().cpu().numpy()

        # Get n-th zero of Airy function
        alpha_n = ai_zeros(n + 1)[0][n]

        # Eigenfunction: Ψ_n(x) = Ai(x - α_n)
        psi = airy(x_np + alpha_n)[0]

        # # Normalize: ∫|ψ|² dx = 1
        # dx = x_np[1] - x_np[0] if len(x_np) > 1 else 0.01
        # norm_sq = np.sum(psi ** 2) * dx
        # psi = psi / np.sqrt(norm_sq)

        # Convert back to tensor
        solution = torch.tensor(psi, dtype=torch.float32).to(device)

        return solution

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
        base_solution = self.airy_solution(x, mode)
        return base_solution + perturbation

    def compute_potential(self, x, potential_type="gravity_well", **kwargs):
        """
        Compute potential function for the 1D domain.
        """
        if potential_type == "gravity_well":
            # The dimensionless form requires scaling factor
            # V(x) = (2^(1/3)) * x
            xi_scale = (2.0) ** (1.0 / 3.0)
            V = xi_scale * x.clone()

            return V
        else:
            raise ValueError(f"Unknown potential type: {potential_type}")

    def pde_loss(self, inputs, predictions, gamma, p, potential_type="gravity_well", precomputed_potential=None):
        """
        Compute the PDE loss for the Gross-Pitaevskii equation.
        μψ = - ∇²ψ + Vψ + γ|ψ|²ψ
        """
        # Get the complete solution (base + perturbation) for PL-PINN. Else, use the neural network predicton
        if self.use_perturbation:
            u = self.get_complete_solution(inputs, predictions)  # PL-PINN algorithm
        else:
            u = predictions  # Vanilla PINN / Curriculum learning

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
        #interaction = gamma * torch.abs(u) ** (p - 1) * u

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

        # Get the complete solution (base + perturbation) for PL-PINN. Else, use the neural network predicton
        if self.use_perturbation:
            full_u = self.get_complete_solution(boundary_points, u_pred)  # PL-PINN algorithm
        else:
            full_u = u_pred  # Vanilla PINN / Curriculum learning

        return torch.mean((full_u - boundary_values) ** 2)

    def normalization_loss(self, u, dx):
        """
        Compute normalization loss using proper numerical integration.
        """
        integral = torch.sum(u ** 2) * dx
        return (integral - 1.0) ** 2


def train_gpe_model(gamma_values, modes, p, X_train, lb, ub, layers,
                    epochs, tol, perturb_const,
                    potential_type='gravity_well', lr=1e-5, verbose=True):
    """
    Train the GPE model for different modes and gamma values for the PL-PINN implementation.

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
    dx = X_train[1, 0] - X_train[0, 0]  # Assuming uniform grid
    X_tensor = torch.tensor(X_train, dtype=torch.float32, requires_grad=True).to(device)

    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_train[0] = {X_train[0, 0]:.6f}, X_train[-1] = {X_train[-1, 0]:.6f}")
    print(f"  dx = {X_train[1, 0] - X_train[0, 0]:.6f}")

    # Create boundary conditions
    boundary_points = torch.tensor([[lb]], dtype=torch.float32).to(device)
    boundary_values = torch.zeros((1, 1), dtype=torch.float32).to(device)

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
            model = GrossPitaevskiiPINN(layers, mode=mode, gamma=gamma).to(device)

            # If this isn't the first gamma value, initialize with previous model's weights
            if prev_model is not None:
                model.load_state_dict(prev_model.state_dict())
            elif gamma == 0.0:
                # Pre-train on analytical solution
                model = pretrain_on_analytical_solution(model, mode, X_train,
                                                        epochs=2000, lr=1e-3, verbose=verbose)
                # Verify the eigenvalue is correct
                if verbose:
                    from scipy.special import ai_zeros
                    expected_mu = -ai_zeros(mode + 1)[0][mode]
                    print(f"  Expected μ for mode {mode}: {expected_mu:.4f}")

                    x_np = X_train.flatten()
                    alpha_n = ai_zeros(mode + 1)[0][mode]
                    psi = airy(x_np + alpha_n)[0]
                    print(f"  ψ(0) = {psi[0]:.6e}")
                    print(f"  ψ(35) = {psi[-1]:.6e}")
                    print(f"  ψ(35)/ψ(0) = {psi[-1] / psi[0]:.6e}")

            else:
                # Use advanced initialization for any other starting gamma not equal 0
                model.apply(lambda m: advanced_initialization(m, mode))

            # Adam optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            # Create scheduler to decrease learning rate during training
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=200, T_mult=2, eta_min=1e-6
            )

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
                      modes, p, constant_history, perturb_const, potential_type,
                      save_dir="Gross-Pitaevskii/src/final/refine/test",
                      normalization_type="unit"):  # Add this parameter
    """
    Plot wavefunctions for different modes and gamma values.

    normalization_type:
        - "unit": normalize to ∫|ψ|² = 1 (default)
        - "max": normalize to max|ψ| = 1 for γ=0, scale others accordingly
        - "particle": use γ to determine effective particle number
        - "scaling":
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

        # Store γ=0 normalization factor for reference
        gamma_0_norm = None

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

                # Choose normalization based on type
                if normalization_type == "unit":
                    # Standard: ∫|ψ|² = 1
                    #u_np /= np.sqrt(np.sum(u_np ** 2) * dx)

                    xi_scale = (2.0) ** (1.0 / 3.0)
                    u_np *= xi_scale

                    u_np /= np.sqrt(np.sum(u_np ** 2) * dx)

                elif normalization_type == "max":

                    xi_scale = (2.0) ** (1.0 / 3.0)
                    u_np /= xi_scale

                    # Normalize so max amplitude matches paper's scale
                    if gamma == 0:
                        # For γ=0, normalize to max = 1
                        gamma_0_norm = np.max(np.abs(u_np))
                        u_np /= gamma_0_norm
                    else:
                        # For γ>0, apply same normalization as γ=0
                        if gamma_0_norm is not None:
                            u_np /= gamma_0_norm
                        else:
                            u_np /= np.max(np.abs(u_np))

                elif normalization_type == "particle":
                    # Normalize using the relationship: γ ∝ N
                    # Smaller γ → larger amplitude (fewer particles compressed)
                    # This mimics the paper's approach
                    norm_factor = np.sqrt(np.sum(u_np ** 2) * dx)
                    if norm_factor > 0:
                        u_np /= norm_factor
                    # Apply γ-dependent scaling
                    scale = 1.0 / np.sqrt(1.0 + gamma / 10.0)  # Adjust denominator as needed
                    u_np *= scale

                elif normalization_type == "scaling":
                    if gamma == 0:
                        u_np /= np.sqrt(np.sum(u_np ** 2) * dx)
                        gamma_0_amplitude = np.max(np.abs(u_np))
                    else:
                        # Scale by same factor as γ=0
                        current_norm = np.sqrt(np.sum(u_np ** 2) * dx)
                        u_np /= current_norm
                        u_np *= gamma_0_amplitude / np.max(np.abs(u_np))

                # For mode 0, ensure positive values
                if mode == 0:
                    u_np = np.abs(u_np)

                # Plot
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
        plt.xlim(0, 20.0)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"mode_{mode}_wavefunction_p{p}_{potential_type}.png"), dpi=300)
        plt.close()

    # Also create a combined grid figure to show all modes
    plot_combined_grid(models_by_mode, X_test, gamma_values, modes, p,
                       constant_history, perturb_const, potential_type, save_dir)


def plot_combined_grid(models_by_mode, X_test, gamma_values, modes, p,
                       constant_history, perturb_const, potential_type,
                       save_dir="Gross-Pitaevskii/src/final/refine/test"):
    """
    Create a grid of subplots showing all modes.
    """
    # Determine grid dimensions
    n_modes = len(modes)
    n_cols = min(4, n_modes)  # Max 4 columns
    n_rows = (n_modes + n_cols - 1) // n_cols  # Ceiling division

    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))

    # Flatten axes if it's a 2D array
    if n_rows > 1 or n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]  # Make it iterable

    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    dx = X_test[1, 0] - X_test[0, 0]  # Spatial step size

    # Different line styles and colors
    linestyles = ['-', '--', '-.', ':', '-', '--']
    colors = ['k', 'b', 'r', 'g', 'm', 'c', 'slategray']

    # Plot each mode in its subplot
    for i, mode in enumerate(modes):
        if i >= len(axes) or mode not in models_by_mode:
            continue

        ax = axes[i]

        # Plot solutions for different gamma values
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

                # Proper normalization
                u_np /= np.sqrt(np.sum(u_np ** 2) * dx)

                # For mode 0, ensure all wavefunctions are positive
                if mode == 0:
                    # Take absolute value to ensure positive values
                    u_np = np.abs(u_np)

                # Plot the wavefunction (not density)
                if gamma % 4 == 0:
                    ax.plot(X_test.flatten(), u_np,
                            linestyle=linestyles[j % len(linestyles)],
                            color=colors[j % len(colors)],
                            label=f"γ={gamma:.1f}")

        # Configure the subplot
        ax.set_title(f"mode {mode}", fontsize=12)
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel(r"$\psi(x)$", fontsize=12)
        ax.grid(True)
        ax.legend(fontsize=6)
        ax.set_xlim(0, 15)

    # Hide any unused subplots
    for i in range(len(modes), len(axes)):
        axes[i].axis('off')

    # Finalize and save combined figure
    fig.suptitle(f"Wavefunctions for All Modes (p={p})", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(save_dir, f"all_modes_combined_wavefunctions_p{p}_{potential_type}.png"), dpi=300)
    plt.close(fig)


def plot_mu_vs_gamma(mu_table, modes, p, potential_type, save_dir="Gross-Pitaevskii/src/final/refine/test",
                     sample_interval=4):
    """
    Plot chemical potential vs. interaction strength for different modes.

    Parameters:
    -----------
    sample_interval : int
        Plot every nth data point (default: 4). Set to 1 for all points.
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))

    # Different markers for different modes
    markers = ['o', 's', '^', 'v', 'D', 'x', '*', '+']
    colors = ['k', 'b', 'r', 'g', 'm', 'c', 'orange', 'purple']

    # Plot μ vs γ for each mode
    for i, mode in enumerate(modes):
        if mode not in mu_table:
            continue

        gamma_list, mu_list = zip(*mu_table[mode])

        # Sample data points at specified interval
        if sample_interval > 1:
            # Take every sample_interval-th point
            sampled_indices = range(0, len(gamma_list), sample_interval)
            gamma_sampled = [gamma_list[idx] for idx in sampled_indices]
            mu_sampled = [mu_list[idx] for idx in sampled_indices]

            # Always include the last point if it's not already included
            if len(gamma_list) - 1 not in sampled_indices:
                gamma_sampled.append(gamma_list[-1])
                mu_sampled.append(mu_list[-1])
        else:
            gamma_sampled = gamma_list
            mu_sampled = mu_list

        plt.plot(mu_sampled, gamma_sampled,
                 marker=markers[i % len(markers)],
                 color=colors[i % len(colors)],
                 linestyle='-',
                 markersize=6,  # Slightly larger markers since fewer points
                 linewidth=1.5,
                 label=f"Mode {mode}")

    plt.ylabel(r"$\eta$ (Interaction Strength)", fontsize=18)
    plt.xlabel(r"$\lambda$ (Eigenvalue)", fontsize=18)
    plt.title(f"Chemical Potential vs. Interaction Strength for Modes 0-5", fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Update filename to indicate sampling if used
    if sample_interval > 1:
        filename = f"mu_vs_gamma_all_modes_p{p}_{potential_type}_sampled_{sample_interval}.png"
    else:
        filename = f"mu_vs_gamma_all_modes_p{p}_{potential_type}.png"

    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.close()


def advanced_initialization(m, mode):
    """Initialize network weights with consideration of the mode number"""
    if isinstance(m, nn.Linear):
        if mode >= 3:
            gain = 0.1 / (1.0 + 0.2 * mode)  # Smaller for high modes
            nn.init.xavier_normal_(m.weight, gain=gain)
            m.bias.data.fill_(0.0001)  # Small bias
        else:
            gain = 1.0 / (1.0 + 0.2 * mode)
            nn.init.xavier_normal_(m.weight, gain=gain)
            m.bias.data.fill_(0.01)


def pretrain_on_analytical_solution(model, mode, X_train, epochs=5000, lr=1e-3, verbose=True):
    """Pre-train network to output the analytical eigenfunction."""
    model.train()
    X_tensor = torch.tensor(X_train, dtype=torch.float32, requires_grad=True).to(device)

    with torch.no_grad():
        analytical_target = model.airy_solution(X_tensor.squeeze(), mode).unsqueeze(-1).detach()

    # Use Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        if epoch < epochs - 500:
            # Adam phase
            optimizer.zero_grad()
            prediction = model.forward(X_tensor)
            loss = torch.mean((prediction - analytical_target) ** 2)
            loss.backward()
            optimizer.step()
        else:
            # Switch to LBFGS for final refinement
            if epoch == epochs - 500:
                optimizer = torch.optim.LBFGS(model.parameters(), lr=lr * 0.1, max_iter=20)

                def closure():
                    optimizer.zero_grad()
                    pred = model.forward(X_tensor)
                    l = torch.mean((pred - analytical_target) ** 2)
                    l.backward()
                    return l

                if verbose:
                    print(f"  Switching to LBFGS at epoch {epoch}")

            # LBFGS phase
            loss = optimizer.step(closure)

        # Logging
        if verbose and epoch % 500 == 0:
            print(f"  Pre-training epoch {epoch}, loss: {loss.item():.2e}")

        # Early stopping
        if loss.item() < 1e-12:
            if verbose:
                print(f"  Pre-training converged at epoch {epoch}, loss: {loss.item():.2e}")
            break

    if verbose:
        print(f"  Pre-training finished, final loss: {loss.item():.2e}")

    if verbose:
        print(f"  Pre-training finished, final loss: {loss.item():.2e}")

        # DEBUG: Check eigenvalue after pre-training
        X_tensor = torch.tensor(X_train, dtype=torch.float32, requires_grad=True).to(device)
        pred = model.forward(X_tensor)

        # Compute eigenvalue manually
        pde_loss, lambda_val = model.pde_loss(X_tensor, pred, gamma=0.0, p=3, potential_type='gravity_well')
        print(f"  Eigenvalue after pre-training: {lambda_val.item():.4f}")

    return model


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

    # Determine grid dimensions
    n_modes = len(modes)
    n_cols = min(4, n_modes)  # Max 4 columns, matching plot_combined_grid
    n_rows = (n_modes + n_cols - 1) // n_cols  # Ceiling division

    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))

    # Flatten axes if it's a 2D array
    if n_rows > 1 or n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]  # Make it iterable

    # Different line styles and colors for different gamma values
    linestyles = ['-', '--', '-.', ':', '-', '--']
    colors = ['k', 'b', 'r', 'g', 'm', 'c', 'slategray']

    # Plot each mode in its subplot
    for i, mode in enumerate(modes):
        if i >= len(axes) or mode not in training_history:
            continue

        ax = axes[i]

        # Plot loss for each gamma value
        for j, gamma in enumerate(gamma_values):
            if gamma in training_history[mode]:
                # Get loss history for this mode and gamma
                loss_history = training_history[mode][gamma]['loss']

                # X-axis values (epoch numbers)
                epoch_nums = np.linspace(0, epochs, len(loss_history))

                # Plot loss on log scale
                if gamma % 4 == 0:
                    ax.semilogy(epoch_nums, loss_history,
                                color=colors[j % len(colors)],
                                linestyle=linestyles[j % len(linestyles)],
                                label=f"η={gamma:.1f}")

        # Configure the subplot
        ax.set_title(f"mode {mode}", fontsize=12)
        ax.set_xlabel("Epochs", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.grid(True)
        ax.legend(fontsize=6)

    # Hide any unused subplots
    for i in range(len(modes), len(axes)):
        axes[i].axis('off')

    # Finalize and save combined figure
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
        for gamma in [0.0]:  # Focus on γ=0 for clarity
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

                # Add a trend line (final 30% of training)
                if len(loss_history) > 10:
                    start_idx = int(len(loss_history) * 0.7)
                    end_loss = np.log(ultra_smooth_loss[-1] if window_size > 1 else loss_history[-1])
                    start_loss = np.log(ultra_smooth_loss[start_idx] if window_size > 1 else loss_history[start_idx])

                    # Only add trend if there's a decrease
                    if end_loss < start_loss:
                        trend_x = np.array([epoch_nums[start_idx], epoch_nums[-1]])
                        trend_y = np.exp(np.array([start_loss, end_loss]))
                        plt.semilogy(trend_x, trend_y, '--', color=colors[i], alpha=0.5)

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
    # Convert models to CPU and save state dicts to avoid device issues
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


def plot_epochs_until_stopping(epochs_history, modes, gamma_values, p, potential_type,
                               save_dir="Gross-Pitaevskii/src/final/refine/test"):
    """
    Plot the number of epochs until early stopping for different modes and gamma values.
    Creates three separate plots instead of subplots.

    Parameters:
    -----------
    epochs_history : dict
        Dictionary containing epochs until stopping for all modes and gamma values
    modes : list
        List of modes to include in the plot
    gamma_values : list
        List of gamma values to include in the plot
    save_dir : str
        Directory to save the plots
    """
    os.makedirs(save_dir, exist_ok=True)

    # Plot 1: Line plot showing epochs vs gamma for each mode
    plt.figure(figsize=(10, 6))
    colors = ['k', 'b', 'r', 'g', 'm', 'c', 'orange', 'purple']
    markers = ['o', 's', '^', 'v', 'D', 'x', '*', '+']

    for i, mode in enumerate(modes):
        if mode in epochs_history:
            gamma_list = []
            epochs_list = []

            for gamma in gamma_values:
                if gamma in epochs_history[mode]:
                    gamma_list.append(gamma)
                    epochs_list.append(epochs_history[mode][gamma])

            if gamma_list:  # Only plot if we have data
                plt.plot(gamma_list, epochs_list,
                         color=colors[i % len(colors)],
                         marker=markers[i % len(markers)],
                         linestyle='-',
                         markersize=4,
                         label=f"Mode {mode}")

    plt.xlabel(r"$\gamma$ (Interaction Strength)", fontsize=14)
    plt.ylabel("Epochs Until Early Stopping", fontsize=14)
    plt.title("Training Efficiency: Epochs Until Convergence", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"epochs_vs_gamma_p{p}_{potential_type}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Heatmap showing epochs for all mode-gamma combinations
    plt.figure(figsize=(12, 6))

    # Prepare data for heatmap
    heatmap_data = []
    gamma_labels = []
    mode_labels = [f"Mode {mode}" for mode in modes]

    # Sample gamma values for heatmap
    sampled_gammas = [g for g in gamma_values if g % 10 == 0]  # Every 10th gamma value

    for gamma in sampled_gammas:
        gamma_labels.append(f"γ={gamma}")
        row = []
        for mode in modes:
            if mode in epochs_history and gamma in epochs_history[mode]:
                row.append(epochs_history[mode][gamma])
            else:
                row.append(np.nan)
        heatmap_data.append(row)

    heatmap_data = np.array(heatmap_data).T

    # Create heatmap
    im = plt.imshow(heatmap_data, cmap='viridis_r', aspect='auto', interpolation='nearest')

    plt.xticks(range(len(gamma_labels)), gamma_labels, rotation=45, ha='right')
    plt.yticks(range(len(mode_labels)), mode_labels)
    plt.xlabel(r"$\gamma$ (Interaction Strength)", fontsize=14)
    plt.ylabel("Mode", fontsize=14)
    plt.title("Training Efficiency Heatmap", fontsize=16)
    cbar = plt.colorbar(im)
    cbar.set_label("Epochs Until Early Stopping", fontsize=12)

    for i in range(len(mode_labels)):
        for j in range(len(gamma_labels)):
            if not np.isnan(heatmap_data[i, j]):
                plt.text(j, i, int(heatmap_data[i, j]),
                         ha="center", va="center", color="white", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"epochs_heatmap_p{p}_{potential_type}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 3: Trend across modes for selected gamma values
    plt.figure(figsize=(10, 6))

    selected_gammas = [0, 20, 40, 60, 80, 100]

    for gamma in selected_gammas:
        epochs_for_gamma = []
        valid_modes = []

        for mode in modes:
            if mode in epochs_history and gamma in epochs_history[mode]:
                epochs_for_gamma.append(epochs_history[mode][gamma])
                valid_modes.append(mode)

        if epochs_for_gamma:
            plt.plot(valid_modes, epochs_for_gamma,
                     marker='o', linestyle='-', linewidth=2,
                     label=f"γ={gamma}")

    plt.xlabel("Mode Number", fontsize=14)
    plt.ylabel("Epochs Until Early Stopping", fontsize=14)
    plt.title("Training Efficiency Across Modes", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xticks(modes)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"epochs_by_mode_trend_p{p}_{potential_type}.png"), dpi=300)
    plt.close()

    print("\n=== Early Stopping Statistics ===")
    for mode in modes:
        if mode in epochs_history:
            epochs_list = list(epochs_history[mode].values())
            print(f"Mode {mode}:")
            print(f"  Average epochs: {np.mean(epochs_list):.1f}")
            print(f"  Min epochs: {np.min(epochs_list)}")
            print(f"  Max epochs: {np.max(epochs_list)}")
            print(f"  Std epochs: {np.std(epochs_list):.1f}")


# ====================== COMPARISON FUNCTIONS ========================

def train_regular_pinn(mode, gamma, p, X_train, lb, ub, layers, epochs, lr=1e-3, tol=1e-5, perturb_const=0.01,
                       verbose=False):
    """
    Train a regular PINN for a single mode and gamma value.
    This is the baseline comparison method - trains from scratch for every gamma.
    """
    dx = X_train[1, 0] - X_train[0, 0]
    X_tensor = torch.tensor(X_train, dtype=torch.float32, requires_grad=True).to(device)

    # Create boundary conditions
    boundary_points = torch.tensor([[lb]], dtype=torch.float32).to(device)
    boundary_values = torch.zeros((1, 1), dtype=torch.float32).to(device)

    # Initialize model
    model = GrossPitaevskiiPINN(layers, mode=mode, gamma=gamma, use_perturbation=False).to(device)

    # Key change: Pre-train on analytical solution for gamma=0
    if gamma == 0.0:
        if verbose:
            print(f"    Pre-training Vanilla PINN on analytical solution for mode {mode}")
        model = pretrain_on_analytical_solution(model, mode, X_train, epochs=2000, lr=1e-3, verbose=verbose)
    else:
        model.apply(lambda m: advanced_initialization(m, mode))

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=200, T_mult=2, eta_min=1e-6)

    # Training loop
    best_loss = float('inf')
    patience_counter = 0
    patience = 2000
    normal_const = None

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        u_pred = model.forward(X_tensor)

        # Calculate losses
        boundary_loss = model.boundary_loss(boundary_points, boundary_values)
        norm_loss = model.normalization_loss(u_pred, dx)
        constraint_loss = 1.0 * boundary_loss + 20.0 * norm_loss

        pde_loss, lambda_value = model.pde_loss(X_tensor, u_pred, gamma, p, "gravity_well")
        total_loss = pde_loss + constraint_loss

        # Backpropagate
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(total_loss)

        # Early stopping
        current_loss = total_loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if current_loss <= tol or patience_counter >= patience:
            if verbose:
                print(f"Vanilla PINN - Mode {mode}, γ={gamma}, stopped at epoch {epoch}, loss: {current_loss:.6f}")
            break

        if verbose and epoch % 500 == 0:
            print(f"Vanilla PINN - Mode {mode}, γ={gamma}, Epoch {epoch}, Loss: {current_loss:.6f}")

    return model, normal_const


def train_curriculum_pinn(modes, gamma_values, p, X_train, lb, ub, layers, epochs, lr=1e-3, tol=1e-5,
                          perturb_const=0.01, verbose=False):
    """
    Curriculum learning with analytical initialization at gamma=0.
    """
    models_by_mode = {}
    constants_by_mode = {}

    # Sort gamma values for curriculum (start with smaller gamma)
    sorted_gammas = sorted(gamma_values)

    for mode in modes:
        if verbose:
            print(f"Training curriculum for mode {mode}")

        models_by_gamma = {}
        prev_model = None

        for i, gamma in enumerate(sorted_gammas):
            if verbose:
                print(f"  Training γ={gamma}")

            # Initialize model
            model = GrossPitaevskiiPINN(layers, mode=mode, gamma=gamma, use_perturbation=False).to(device)

            if i == 0 and gamma == 0.0:
                # Key change: Start with analytical initialization for gamma=0
                if verbose:
                    print(f"    Pre-training Curriculum PINN on analytical solution for mode {mode}")
                model = pretrain_on_analytical_solution(model, mode, X_train, epochs=2000, lr=1e-3, verbose=verbose)
            elif prev_model is not None and i > 0:
                # Transfer Learning from previous model
                prev_state = prev_model.state_dict()
                model_state = model.state_dict()

                # Full network transfer with adaptive strength
                transfer_strength = 0.95

                # Transfer all layers
                for layer_name in model_state.keys():
                    if layer_name in prev_state:
                        model_state[layer_name] = (transfer_strength * prev_state[layer_name] +
                                                   (1 - transfer_strength) * model_state[layer_name])

                model.load_state_dict(model_state)
                initial_lr = lr * 0.1  # Start with lower LR for transferred model
            else:
                # Use normal initialization
                model.apply(lambda m: advanced_initialization(m, mode))
                initial_lr = lr

            # Adaptive epoch allocation
            if i == 0:
                epochs_for_this_gamma = epochs // 2
            else:
                remaining_epochs = epochs - (epochs // 2)
                remaining_gammas = len(sorted_gammas) - 1
                epochs_for_this_gamma = int(remaining_epochs / remaining_gammas * (1.0 + 0.2 * i))

            epochs_for_this_gamma = min(epochs_for_this_gamma, epochs)

            if verbose:
                print(f"    Allocated {epochs_for_this_gamma} epochs")

            model, const = train_single_model(
                model, gamma, p, X_train, lb, ub,
                epochs_for_this_gamma, initial_lr if i > 0 else lr, lr, tol, perturb_const,
                verbose=False, use_warmstart=(prev_model is not None and i > 0)
            )

            models_by_gamma[gamma] = model
            constants_by_mode[mode] = const
            prev_model = model

        models_by_mode[mode] = models_by_gamma

    return models_by_mode, constants_by_mode


def train_single_model(model, gamma, p, X_train, lb, ub, epochs, initial_lr, final_lr, tol,
                       perturb_const, verbose=False, use_warmstart=False):
    """
    Improved single model training with adaptive learning rate and better early stopping.
    """
    dx = X_train[1, 0] - X_train[0, 0]
    X_tensor = torch.tensor(X_train, dtype=torch.float32, requires_grad=True).to(device)

    # Create boundary conditions
    boundary_points = torch.tensor([[lb]], dtype=torch.float32).to(device)
    boundary_values = torch.zeros((1, 1), dtype=torch.float32).to(device)

    if use_warmstart:
        # For transferred models, use lower initial LR with gradual increase
        optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)

        # Custom scheduler that increases LR first, then decreases
        def lr_lambda(epoch):
            warmup_epochs = min(100, epochs // 10)
            if epoch < warmup_epochs:
                # Gradually increase LR from initial to final
                return 1.0 + (final_lr / initial_lr - 1.0) * (epoch / warmup_epochs)
            else:
                # Then use cosine annealing
                remaining_epochs = epochs - warmup_epochs
                progress = (epoch - warmup_epochs) / remaining_epochs
                return (final_lr / initial_lr) * (0.5 * (1 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        # For fresh models, use standard approach
        optimizer = torch.optim.Adam(model.parameters(), lr=final_lr)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=200, T_mult=2, eta_min=1e-6)

    # Early Stopping
    best_loss = float('inf')
    patience_counter = 0
    patience = min(2000, epochs // 2)  # Adaptive patience, but not too low
    normal_const = None

    # Track improvement rate for dynamic stopping
    loss_history = []
    improvement_window = 50

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        u_pred = model.forward(X_tensor)

        # Calculate losses
        boundary_loss = model.boundary_loss(boundary_points, boundary_values)
        norm_loss = model.normalization_loss(u_pred, dx)
        constraint_loss = 10.0 * boundary_loss + 20.0 * norm_loss

        pde_loss, lambda_value = model.pde_loss(X_tensor, u_pred, gamma, p, "gravity_well")
        total_loss = pde_loss + constraint_loss

        # Backpropagate
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Enhanced early stopping
        current_loss = total_loss.item()
        loss_history.append(current_loss)

        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1

        # Check convergence based on improvement rate
        if len(loss_history) >= improvement_window:
            recent_improvement = (loss_history[-improvement_window] - loss_history[-1]) / loss_history[
                -improvement_window]
            if recent_improvement < 0.001:  # Less than 0.1% improvement
                patience_counter += 2  # Accelerate stopping for slow improvement

        # Early stopping conditions
        if current_loss <= tol or patience_counter >= patience:
            if verbose:
                print(f"    Converged at epoch {epoch}, loss: {current_loss:.6f}")
            break

    return model, normal_const


def compute_analytical_solution(X_test, mode, potential_type="gravity_well", gamma=0, L=1.0):
    """
    Compute analytical solution for comparison.
    For gamma=0, this provides the linear solution for different potential types.
    For gamma>0, returns None if no analytical solution is available.
    """
    x = X_test.flatten()
    if len(X_test) <= 1:
        raise ValueError("X_test must have at least 2 points to compute dx")
    dx = X_test[1, 0] - X_test[0, 0]

    if gamma == 0:  # Linear case - analytical solutions available

        if potential_type == "box":
            # Box eigenfunction solution
            n_actual = mode + 1  # Convert mode number to quantum number
            norm_factor = np.sqrt(2.0 / L)
            psi_analytical = norm_factor * np.sin(n_actual * np.pi * x / L)

        elif potential_type == "gravity_well":
            # Gravity well eigenfunction using Airy functions
            # Airy function zeros (negative values where Ai(z) = 0)
            airy_zeros = [
                -2.33810741, -4.08794944, -5.52055983, -6.78670809, -7.94413359,
                -9.02265085, -10.0401743, -11.0085243, -11.9360157, -12.8287767
            ]

            # Get the nth zero
            if mode < len(airy_zeros):
                x_n = airy_zeros[mode]
            else:
                # Asymptotic approximation for higher modes
                x_n = -(1.5 * np.pi * (mode + 0.75)) ** (2 / 3)

            # Coordinate transformation for gravity well
            xi_scale = (2.0) ** (1 / 3)

            # Initialize solution array
            psi_analytical = np.zeros_like(x)

            # Calculate Airy function for x >= 0
            for i, x_val in enumerate(x):
                if x_val >= 0:
                    z = xi_scale * x_val + x_n
                    psi_analytical[i] = airy(z)[0]
                else:
                    psi_analytical[i] = 0.0

        elif potential_type == "harmonic":
            # Harmonic oscillator eigenfunctions (Hermite polynomials)
            from scipy.special import hermite
            import math

            # Hermite polynomial of order n
            H_n = hermite(mode)(x)

            # Normalization factor
            norm_factor = (2 ** mode * math.factorial(mode) * np.sqrt(np.pi)) ** (-0.5)

            # Complete wavefunction
            psi_analytical = norm_factor * np.exp(-x ** 2 / 2) * H_n

        else:
            raise ValueError(f"Unknown potential type for analytical solution: {potential_type}")

        # Normalize the solution
        norm_integral = np.sum(psi_analytical ** 2) * dx
        if norm_integral > 1e-12:
            psi_analytical /= np.sqrt(norm_integral)

        # For mode 0 in non-gravity potentials, ensure positive values
        if mode == 0 and potential_type != "gravity_well":
            psi_analytical = np.abs(psi_analytical)

        return psi_analytical

    else:
        # For gamma > 0, analytical solutions are generally not available
        # Exception: box potential has Jacobi elliptic function solution
        if potential_type == "box" and gamma > 0:
            # You could implement the Jacobi elliptic function solution here
            # For now, return None
            return None
        else:
            return None


def compare_with_analytical(model, X_test, mode, potential_type="gravity_well", gamma=0, L=1.0):
    """
    Compare PINN solution with analytical solution and compute error metrics.
    """
    # Get PINN solution
    model.eval()
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        u_pred = model.forward(X_tensor)
        psi_pinn = model.get_complete_solution(X_tensor, u_pred).cpu().numpy().flatten()

    # Normalize PINN solution
    dx = X_test[1, 0] - X_test[0, 0] if len(X_test) > 1 else 0.01

    if potential_type == "gravity_well":
        # Only normalize over x >= 0 domain
        x_vals = X_test.flatten()
        pos_mask = x_vals >= 0
        if np.any(pos_mask):
            psi_pos = psi_pinn[pos_mask]
            norm_factor = np.sqrt(np.sum(psi_pos ** 2) * dx)
            if norm_factor > 1e-12:
                psi_pinn = psi_pinn / norm_factor
    else:
        # Standard normalization
        norm_factor = np.sqrt(np.sum(psi_pinn ** 2) * dx)
        if norm_factor > 1e-12:
            psi_pinn = psi_pinn / norm_factor

    # Get analytical solution
    psi_analytical = compute_analytical_solution(X_test, mode, potential_type, gamma, L)

    if psi_analytical is not None:
        # Ensure same sign convention (align the solutions)
        if np.dot(psi_pinn, psi_analytical) < 0:
            psi_analytical = -psi_analytical

        # Compute error metrics
        l2_error = np.sqrt(np.sum((psi_pinn - psi_analytical) ** 2) * dx)
        max_error = np.max(np.abs(psi_pinn - psi_analytical))

        # Compute relative errors
        analytical_norm = np.sqrt(np.sum(psi_analytical ** 2) * dx)
        if analytical_norm > 1e-12:
            relative_l2_error = l2_error / analytical_norm
        else:
            relative_l2_error = float('inf')

        print(f"Comparison with analytical solution:")
        print(f"  L2 error: {l2_error:.6e}")
        print(f"  Max error: {max_error:.6e}")
        print(f"  Relative L2 error: {relative_l2_error:.6e}")

        return {
            'psi_pinn': psi_pinn,
            'psi_analytical': psi_analytical,
            'l2_error': l2_error,
            'max_error': max_error,
            'relative_l2_error': relative_l2_error
        }
    else:
        print(f"No analytical solution available for {potential_type} with gamma={gamma}")
        return {'psi_pinn': psi_pinn, 'psi_analytical': None}


def compute_solution_error(model, constant, mode, gamma, p, X_test, method_name, perturb_const=0.01):
    """
    Compute the absolute and relative error for a trained model.
    """
    model.eval()

    X_tensor = torch.tensor(X_test, dtype=torch.float32, requires_grad=True).to(device)
    dx = X_test[1, 0] - X_test[0, 0]

    with torch.no_grad():
        if method_name == "PL-PINN":
            # Perturbative approach
            u_pred = model.forward(X_tensor)
            u_pred = u_pred * (perturb_const / constant)
            full_u = model.get_complete_solution(X_tensor, u_pred)
        else:
            # Use the network output as the solution
            full_u = model.forward(X_tensor)

        u_np = full_u.cpu().numpy().flatten()

        # Normalize the solution
        u_np /= np.sqrt(np.sum(u_np ** 2) * dx)

        # For mode 0, ensure positive values
        if mode == 0:
            u_np = np.abs(u_np)

    # Try to get analytical solution for comparison
    analytical = compute_analytical_solution(X_test, mode, "gravity_well", gamma)

    if analytical is not None and gamma == 0:
        # Compute error against analytical solution
        abs_error = np.mean(np.abs(u_np - analytical))
        rel_error = np.linalg.norm(u_np - analytical) / np.linalg.norm(analytical) * 100
    else:
        # For cases without analytical solution, compute PDE residual as quality metric
        X_tensor_eval = torch.tensor(X_test, dtype=torch.float32, requires_grad=True).to(device)

        if method_name == "PL-PINN":
            u_pred_eval = model.forward(X_tensor_eval)
            u_pred_eval = u_pred_eval * (perturb_const / constant)
            u_eval = model.get_complete_solution(X_tensor_eval, u_pred_eval)
        else:
            u_eval = model.forward(X_tensor_eval)

        # Compute PDE residual
        u_x = torch.autograd.grad(u_eval, X_tensor_eval, torch.ones_like(u_eval),
                                  create_graph=True, retain_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, X_tensor_eval, torch.ones_like(u_x),
                                   create_graph=True, retain_graph=True)[0]

        V = model.compute_potential(X_tensor_eval, "gravity_well")
        F_u = -u_xx + V * u_eval + gamma * torch.abs(u_eval) ** (p - 1) * u_eval

        # Chemical potential
        mu = torch.mean(u_eval * F_u) / torch.mean(u_eval * u_eval)

        # PDE residual
        residual = F_u - mu * u_eval
        pde_loss = torch.mean(residual ** 2)

        abs_error = pde_loss.item()
        rel_error = abs_error * 100

    return abs_error, rel_error


def save_regular_pinn_models(regular_models, regular_constants, modes, gamma_values, p, potential_type,
                             save_dir="comparison_results"):
    """
    Save only the Vanilla PINN models.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Convert models to CPU and save state dicts
    regular_state_dicts = {}
    for mode in modes:
        if mode in regular_models:
            regular_state_dicts[mode] = {}
            for gamma in gamma_values:
                if gamma in regular_models[mode]:
                    model = regular_models[mode][gamma]
                    regular_state_dicts[mode][gamma] = {
                        'state_dict': model.cpu().state_dict(),
                        'layers': model.layers,
                        'hbar': model.hbar,
                        'm': model.m,
                        'mode': model.mode,
                        'gamma': model.gamma
                    }

    data = {
        'regular_state_dicts': regular_state_dicts,
        'regular_constants': regular_constants,
        'modes': modes,
        'gamma_values': gamma_values,
        'p': p,
        'potential_type': potential_type
    }

    filename = f"regular_pinn_models_p{p}_{potential_type}.pkl"
    filepath = os.path.join(save_dir, filename)

    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

    print(f"Vanilla PINN models saved to {filepath}")
    return filename


def load_regular_pinn_models(filename, save_dir="comparison_results"):
    """
    Load only the Vanilla PINN models.
    """
    filepath = os.path.join(save_dir, filename)

    if not os.path.exists(filepath):
        print(f"No saved Vanilla PINN models found at {filepath}")
        return None, None, None, None, None, None

    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    # Reconstruct regular models
    regular_models = {}
    for mode in data['regular_state_dicts']:
        regular_models[mode] = {}
        for gamma, model_data in data['regular_state_dicts'][mode].items():
            model = GrossPitaevskiiPINN(
                layers=model_data['layers'],
                hbar=model_data['hbar'],
                m=model_data['m'],
                mode=model_data['mode'],
                gamma=model_data['gamma'],
                use_perturbation=False
            ).to(device)
            model.load_state_dict(model_data['state_dict'])
            model.eval()
            regular_models[mode][gamma] = model

    print(f"Vanilla PINN models loaded from {filepath}")
    return (regular_models, data['regular_constants'],
            data['modes'], data['gamma_values'], data['p'], data['potential_type'])


def save_curriculum_pinn_models(curriculum_models, curriculum_constants, modes, gamma_values, p, potential_type,
                                save_dir="comparison_results"):
    """
    Save only the Curriculum PINN models.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Convert models to CPU and save state dicts
    curriculum_state_dicts = {}
    for mode in modes:
        if mode in curriculum_models:
            curriculum_state_dicts[mode] = {}
            for gamma in gamma_values:
                if gamma in curriculum_models[mode]:
                    model = curriculum_models[mode][gamma]
                    curriculum_state_dicts[mode][gamma] = {
                        'state_dict': model.cpu().state_dict(),
                        'layers': model.layers,
                        'hbar': model.hbar,
                        'm': model.m,
                        'mode': model.mode,
                        'gamma': model.gamma
                    }

    data = {
        'curriculum_state_dicts': curriculum_state_dicts,
        'curriculum_constants': curriculum_constants,
        'modes': modes,
        'gamma_values': gamma_values,
        'p': p,
        'potential_type': potential_type
    }

    filename = f"curriculum_pinn_models_p{p}_{potential_type}.pkl"
    filepath = os.path.join(save_dir, filename)

    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

    print(f"Curriculum PINN models saved to {filepath}")
    return filename


def load_curriculum_pinn_models(filename, save_dir="comparison_results"):
    """
    Load only the Curriculum PINN models.
    """
    filepath = os.path.join(save_dir, filename)

    if not os.path.exists(filepath):
        print(f"No saved Curriculum PINN models found at {filepath}")
        return None, None, None, None, None, None

    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    # Reconstruct curriculum models
    curriculum_models = {}
    for mode in data['curriculum_state_dicts']:
        curriculum_models[mode] = {}
        for gamma, model_data in data['curriculum_state_dicts'][mode].items():
            model = GrossPitaevskiiPINN(
                layers=model_data['layers'],
                hbar=model_data['hbar'],
                m=model_data['m'],
                mode=model_data['mode'],
                gamma=model_data['gamma'],
                use_perturbation=False
            ).to(device)
            model.load_state_dict(model_data['state_dict'])
            model.eval()
            curriculum_models[mode][gamma] = model

    print(f"Curriculum PINN models loaded from {filepath}")
    return (curriculum_models, data['curriculum_constants'],
            data['modes'], data['gamma_values'], data['p'], data['potential_type'])


def train_or_load_regular_pinns(modes, gamma_values, p, X_train, lb, ub, layers, epochs,
                                save_dir="comparison_results", tol=1e-5, perturb_const=0.01, potential_type="gravity_well",
                                force_retrain=False):
    """
    Train Regular PINNs or load from cache if available.
    """
    filename = f"regular_pinn_models_p{p}_{potential_type}.pkl"

    if not force_retrain:
        print("Attempting to load cached Vanilla PINN models...")
        (regular_models, regular_constants, cached_modes, cached_gammas,
         cached_p, cached_potential) = load_regular_pinn_models(filename, save_dir)

        # Check if cached data matches current parameters
        if (regular_models is not None and
                set(cached_modes) == set(modes) and
                set(cached_gammas) == set(gamma_values) and
                cached_p == p and cached_potential == potential_type):
            print("✓ Found matching cached Vanilla PINN models!")
            return regular_models, regular_constants
        else:
            print("✗ Cached Vanilla PINN models don't match current parameters.")

    # Train Regular PINNs from scratch
    print("Training Regular PINNs from scratch...")
    print("-" * 40)

    regular_models = {}
    regular_constants = {}

    for mode in modes:
        regular_models[mode] = {}
        for gamma in gamma_values:
            print(f"Training Vanilla PINN - Mode {mode}, γ={gamma}")
            model, const = train_regular_pinn(mode, gamma, p, X_train, lb, ub, layers, epochs, tol=tol,
                                              perturb_const=perturb_const, verbose=True)
            regular_models[mode][gamma] = model
            regular_constants[mode] = const

    # Save the trained models
    print("Saving Vanilla PINN models...")
    save_regular_pinn_models(regular_models, regular_constants, modes, gamma_values, p, potential_type, save_dir)

    return regular_models, regular_constants


def train_or_load_curriculum_pinns(modes, gamma_values, p, X_train, lb, ub, layers, epochs,
                                   save_dir="comparison_results", tol=1e-5, perturb_const=0.01, potential_type="gravity_well",
                                   force_retrain=False):
    """
    Train Curriculum PINNs or load from cache if available.
    """
    filename = f"curriculum_pinn_models_p{p}_{potential_type}.pkl"

    if not force_retrain:
        print("Attempting to load cached Curriculum PINN models...")
        (curriculum_models, curriculum_constants, cached_modes, cached_gammas,
         cached_p, cached_potential) = load_curriculum_pinn_models(filename, save_dir)

        # Check if cached data matches current parameters
        if (curriculum_models is not None and
                set(cached_modes) == set(modes) and
                set(cached_gammas) == set(gamma_values) and
                cached_p == p and cached_potential == potential_type):
            print("✓ Found matching cached Curriculum PINN models!")
            return curriculum_models, curriculum_constants
        else:
            print("✗ Cached Curriculum PINN models don't match current parameters.")

    # Train Curriculum PINNs from scratch
    print("Training Curriculum PINNs from scratch...")
    print("-" * 40)

    curriculum_models, curriculum_constants = train_curriculum_pinn(
        modes, gamma_values, p, X_train, lb, ub, layers, epochs, tol=tol, perturb_const=perturb_const, verbose=True)

    # Save the trained models
    print("Saving Curriculum PINN models...")
    save_curriculum_pinn_models(curriculum_models, curriculum_constants, modes, gamma_values, p, potential_type,
                                save_dir)

    return curriculum_models, curriculum_constants


def create_comparison_table_individual_caching(modes, gamma_values, p, X_train, X_test, lb, ub, layers, epochs,
                                               save_dir="comparison_results", tol=1e-5, perturb_const=0.01,
                                               force_retrain_regular=False, force_retrain_curriculum=False,
                                               potential_type="gravity_well"):
    """
    Create comparison table with individual caching for each method.

    Parameters:
    -----------
    force_retrain_regular : bool
        If True, retrain Regular PINNs even if cached
    force_retrain_curriculum : bool
        If True, retrain Curriculum PINNs even if cached
    """
    os.makedirs(save_dir, exist_ok=True)

    # Select gamma values for comparison
    comparison_gammas = [0, 20, 40, 60, 80, 100]
    comparison_gammas = [g for g in comparison_gammas if g in gamma_values]

    results = []

    print("=" * 80)
    print("PINN METHODS COMPARISON WITH INDIVIDUAL CACHING")
    print("=" * 80)
    print(f"Modes: {modes}")
    print(f"Gamma values: {comparison_gammas}")
    print(f"Epochs per method: {epochs}")
    print(f"Tolerance: {tol}")
    print("=" * 80)

    # 1. Train or Load Regular PINNs
    print("\n1. Regular PINNs...")
    regular_models, regular_constants = train_or_load_regular_pinns(
        modes, comparison_gammas, p, X_train, lb, ub, layers, epochs,
        save_dir, tol, perturb_const, potential_type, force_retrain_regular
    )

    # 2. Train or Load Curriculum PINNs
    print("\n2. Curriculum PINNs...")
    curriculum_models, curriculum_constants = train_or_load_curriculum_pinns(
        modes, comparison_gammas, p, X_train, lb, ub, layers, epochs,
        save_dir, tol, perturb_const, potential_type, force_retrain_curriculum
    )

    # 3. Load PL-PINN models
    print("\n3. Loading Pre-trained PL-PINN models...")
    print("-" * 40)

    if 'models_by_mode' in globals() and 'constant_history' in globals():
        print("Using PL-PINN models from current session...")
        pl_pinn_models = models_by_mode
        pl_constant_history = constant_history
    else:
        try:
            pl_filename = f"my_gpe_models_p{p}_{potential_type}_pert_const_1e-2_tol_{tol}.pkl"
            pl_save_dir = f"plots_p{p}_{potential_type}_paper_test"
            pl_pinn_models, _, _, pl_constant_history, _ = load_models(pl_filename, pl_save_dir)
            print(f"Loaded PL-PINN models from {pl_filename}")
        except FileNotFoundError:
            print(f"ERROR: Could not find PL-PINN models. Please train PL-PINN models first.")
            return None, None, None

    # 4. Evaluate all methods
    print("\n4. Evaluating all methods...")
    print("-" * 40)

    # Evaluate Regular PINNs
    print("Evaluating Regular PINNs...")
    for mode in modes:
        for gamma in comparison_gammas:
            if mode in regular_models and gamma in regular_models[mode]:
                model = regular_models[mode][gamma]
                const = regular_constants[mode]
                abs_err, rel_err = compute_solution_error(model, const, mode, gamma, p, X_test, "Vanilla PINN",
                                                          perturb_const)
                results.append({
                    'Method': 'Vanilla PINN',
                    'Mode': mode,
                    'Gamma': gamma,
                    'Abs Error': abs_err,
                    'Rel Error': rel_err
                })
                print(f"Vanilla PINN - Mode {mode}, γ={gamma} -> Abs Error: {abs_err:.2e}, Rel Error: {rel_err:.3f}%")

    # Evaluate Curriculum PINNs
    print("Evaluating Curriculum PINNs...")
    for mode in modes:
        for gamma in comparison_gammas:
            if mode in curriculum_models and gamma in curriculum_models[mode]:
                model = curriculum_models[mode][gamma]
                const = curriculum_constants[mode]
                abs_err, rel_err = compute_solution_error(model, const, mode, gamma, p, X_test, "Curriculum Training",
                                                          perturb_const)
                results.append({
                    'Method': 'Curriculum Training',
                    'Mode': mode,
                    'Gamma': gamma,
                    'Abs Error': abs_err,
                    'Rel Error': rel_err
                })
                print(f"Curriculum - Mode {mode}, γ={gamma} -> Abs Error: {abs_err:.2e}, Rel Error: {rel_err:.3f}%")

    # Evaluate PL-PINNs
    print("Evaluating PL-PINNs...")
    for mode in modes:
        for gamma in comparison_gammas:
            if mode in pl_pinn_models and gamma in pl_pinn_models[mode]:
                model = pl_pinn_models[mode][gamma]
                const = pl_constant_history[mode]
                abs_err, rel_err = compute_solution_error(model, const, mode, gamma, p, X_test, "PL-PINN",
                                                          perturb_const)
                results.append({
                    'Method': 'PL-PINN',
                    'Mode': mode,
                    'Gamma': gamma,
                    'Abs Error': abs_err,
                    'Rel Error': rel_err
                })
                print(f"PL-PINN - Mode {mode}, γ={gamma} -> Abs Error: {abs_err:.2e}, Rel Error: {rel_err:.3f}%")

    # Create DataFrame and continue with analysis
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(save_dir, "raw_comparison_results.csv"), index=False)

    # Save raw results
    df.to_csv(os.path.join(save_dir, "raw_comparison_results.csv"), index=False)

    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    # Create paper-style comparison table
    print("\nPAPER-STYLE COMPARISON TABLE")
    print("-" * 60)

    methods = ['Vanilla PINN', 'Curriculum Training', 'PL-PINN']

    # Print header
    print(f"{'Method':<20} {'abs. err':<12} {'rel. err':<10}")
    print("-" * 42)

    # Create summary for each mode
    for mode in modes:
        print(f"\nMode {mode}")
        mode_data = df[df['Mode'] == mode]

        for method in methods:
            method_data = mode_data[mode_data['Method'] == method]
            if not method_data.empty:
                avg_abs = method_data['Abs Error'].mean()
                avg_rel = method_data['Rel Error'].mean()
                print(f"  {method:<18} {avg_abs:<12.2e} {avg_rel:<10.3f}%")

    # Create detailed comparison tables
    print("\n" + "=" * 80)
    print("DETAILED COMPARISON BY GAMMA VALUES")
    print("=" * 80)

    for error_type in ['Abs Error', 'Rel Error']:
        print(f"\n{error_type} Comparison:")
        print("-" * 50)

        # Create pivot table
        pivot_df = df.pivot_table(
            index=['Mode', 'Gamma'],
            columns='Method',
            values=error_type,
            aggfunc='first'
        )

        # Format the table nicely
        if error_type == 'Abs Error':
            formatted_table = pivot_df.applymap(lambda x: f"{x:.2e}" if pd.notna(x) else "N/A")
        else:  # Relative Error
            formatted_table = pivot_df.applymap(lambda x: f"{x:.3f}%" if pd.notna(x) else "N/A")

        print(formatted_table)

        # Save to CSV
        pivot_df.to_csv(os.path.join(save_dir, f"{error_type.lower().replace(' ', '_')}_comparison.csv"))

    # Find best performing method for each mode
    print("\n" + "=" * 80)
    print("BEST PERFORMING METHOD BY MODE")
    print("=" * 80)

    best_results = []
    for mode in modes:
        mode_data = df[df['Mode'] == mode]

        # Find best method by average absolute error
        best_abs = mode_data.groupby('Method')['Abs Error'].mean().idxmin()
        best_abs_value = mode_data.groupby('Method')['Abs Error'].mean().min()

        # Find best method by average relative error
        best_rel = mode_data.groupby('Method')['Rel Error'].mean().idxmin()
        best_rel_value = mode_data.groupby('Method')['Rel Error'].mean().min()

        print(f"Mode {mode}:")
        print(f"  Best Abs Error: {best_abs} ({best_abs_value:.2e})")
        print(f"  Best Rel Error: {best_rel} ({best_rel_value:.3f}%)")

        best_results.append({
            'Mode': mode,
            'Best_Abs_Method': best_abs,
            'Best_Abs_Value': best_abs_value,
            'Best_Rel_Method': best_rel,
            'Best_Rel_Value': best_rel_value
        })

    # Save best results
    best_df = pd.DataFrame(best_results)
    best_df.to_csv(os.path.join(save_dir, "best_methods_by_mode.csv"), index=False)

    # Create final paper-style table with bold formatting indicators
    print("\n" + "=" * 80)
    print("FINAL PAPER-STYLE TABLE")
    print("=" * 80)

    paper_style_data = []
    for mode in modes:
        mode_data = df[df['Mode'] == mode]

        # Calculate average performance for each method
        method_performance = {}
        for method in methods:
            method_data = mode_data[mode_data['Method'] == method]
            if not method_data.empty:
                avg_abs = method_data['Abs Error'].mean()
                avg_rel = method_data['Rel Error'].mean()
                method_performance[method] = {'abs': avg_abs, 'rel': avg_rel}

        # Find best performance
        best_abs_method = min(method_performance.keys(), key=lambda x: method_performance[x]['abs'])
        best_rel_method = min(method_performance.keys(), key=lambda x: method_performance[x]['rel'])

        # Format results with bold indicators
        for method in methods:
            if method in method_performance:
                abs_val = method_performance[method]['abs']
                rel_val = method_performance[method]['rel']

                # Mark best performers
                abs_str = f"**{abs_val:.2e}**" if method == best_abs_method else f"{abs_val:.2e}"
                rel_str = f"**{rel_val:.3f}%**" if method == best_rel_method else f"{rel_val:.3f}%"

                paper_style_data.append({
                    'Mode': f'Mode {mode}',
                    'Method': method,
                    'abs_err': abs_str,
                    'rel_err': rel_str
                })

    # Print final table
    print(f"{'Mode':<8} {'Method':<20} {'abs. err':<15} {'rel. err':<10}")
    print("-" * 55)

    current_mode = None
    for row in paper_style_data:
        if row['Mode'] != current_mode:
            if current_mode is not None:
                print()  # Add spacing between modes
            current_mode = row['Mode']
            print(f"{row['Mode']}")

        print(f"{'':8} {row['Method']:<20} {row['abs_err']:<15} {row['rel_err']:<10}")

    # Save paper style results
    paper_df = pd.DataFrame(paper_style_data)
    paper_df.to_csv(os.path.join(save_dir, "paper_style_results.csv"), index=False)

    print(f"\n** indicates best performance for that metric")
    print(f"\nAll results saved to: {save_dir}/")
    print("Files created:")
    print("  - raw_comparison_results.csv")
    print("  - abs_error_comparison.csv")
    print("  - rel_error_comparison.csv")
    print("  - best_methods_by_mode.csv")
    print("  - paper_style_results.csv")

    return df, paper_style_data, best_results


def plot_comparison_results(results_df, save_dir="comparison_results"):
    """
    Create visualization plots comparing the different methods.
    """
    os.makedirs(save_dir, exist_ok=True)

    methods = results_df['Method'].unique()
    modes = sorted(results_df['Mode'].unique())

    # Set up colors for methods
    colors = {'Vanilla PINN': 'red', 'Curriculum Training': 'blue', 'PL-PINN': 'green'}

    # 1. Plot absolute error comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Absolute Error
    ax1 = axes[0]
    for method in methods:
        method_data = results_df[results_df['Method'] == method]
        mode_avg = method_data.groupby('Mode')['Abs Error'].mean()
        ax1.semilogy(mode_avg.index, mode_avg.values, 'o-',
                     label=method, color=colors.get(method, 'black'), linewidth=2, markersize=6)

    ax1.set_xlabel('Mode', fontsize=14)
    ax1.set_ylabel('Absolute Error', fontsize=14)
    ax1.set_title('Absolute Error Comparison', fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(modes)

    # Relative Error
    ax2 = axes[1]
    for method in methods:
        method_data = results_df[results_df['Method'] == method]
        mode_avg = method_data.groupby('Mode')['Rel Error'].mean()
        ax2.semilogy(mode_avg.index, mode_avg.values, 'o-',
                     label=method, color=colors.get(method, 'black'), linewidth=2, markersize=6)

    ax2.set_xlabel('Mode', fontsize=14)
    ax2.set_ylabel('Relative Error (%)', fontsize=14)
    ax2.set_title('Relative Error Comparison', fontsize=16)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(modes)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "error_comparison_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Plot performance by gamma values
    gamma_values = sorted(results_df['Gamma'].unique())

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, mode in enumerate(modes[:6]):  # Plot first 6 modes
        if i < len(axes):
            ax = axes[i]

            for method in methods:
                method_data = results_df[(results_df['Method'] == method) & (results_df['Mode'] == mode)]
                if not method_data.empty:
                    gamma_avg = method_data.groupby('Gamma')['Abs Error'].mean()
                    ax.semilogy(gamma_avg.index, gamma_avg.values, 'o-',
                                label=method, color=colors.get(method, 'black'), linewidth=2)

            ax.set_xlabel(r"$\eta$ (Interaction Strength)", fontsize=12)
            ax.set_ylabel('Absolute Error', fontsize=12)
            ax.set_title(f'Mode {mode}', fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(len(modes), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "performance_by_interaction_strength.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plots saved to {save_dir}/")


def create_latex_table(results_df, save_path=None):
    """
    Create a LaTeX table from the results for publication.
    """
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Comparison of PINN Methods for Gross-Pitaevskii Equation}")
    latex_lines.append("\\label{tab:pinn_comparison}")
    latex_lines.append("\\begin{tabular}{llcc}")
    latex_lines.append("\\toprule")
    latex_lines.append("Mode & Method & Abs. Error & Rel. Error \\\\")
    latex_lines.append("\\midrule")

    modes = sorted(results_df['Mode'].unique())
    methods = ['Vanilla PINN', 'Curriculum Training', 'PL-PINN']

    for mode in modes:
        mode_data = results_df[results_df['Mode'] == mode]

        # Find best performance for highlighting
        best_abs_method = mode_data.groupby('Method')['Abs Error'].mean().idxmin()
        best_rel_method = mode_data.groupby('Method')['Rel Error'].mean().idxmin()

        for i, method in enumerate(methods):
            method_data = mode_data[mode_data['Method'] == method]
            if not method_data.empty:
                avg_abs = method_data['Abs Error'].mean()
                avg_rel = method_data['Rel Error'].mean()

                # Format with bold for best performance
                abs_str = f"\\textbf{{{avg_abs:.2e}}}" if method == best_abs_method else f"{avg_abs:.2e}"
                rel_str = f"\\textbf{{{avg_rel:.3f}\\%}}" if method == best_rel_method else f"{avg_rel:.3f}\\%"

                mode_str = f"Mode {mode}" if i == 0 else ""
                latex_lines.append(f"{mode_str} & {method} & {abs_str} & {rel_str} \\\\")

        if mode < max(modes):
            latex_lines.append("\\midrule")

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    latex_table = "\n".join(latex_lines)

    if save_path:
        with open(save_path, 'w') as f:
            f.write(latex_table)
        print(f"LaTeX table saved to: {save_path}")

    print("\nLaTeX Table:")
    print(latex_table)

    return latex_table


if __name__ == "__main__":
    # Setup parameters
    lb, ub = 0, 35 # Domain boundaries
    N_f = 4000  # Number of collocation points
    epochs = 5001
    layers = [1, 64, 64, 64, 1]  # Neural network architecture

    # Create uniform grid for training and testing
    X = np.linspace(lb, ub, N_f).reshape(-1, 1)
    X_test = np.linspace(lb, ub, 1000).reshape(-1, 1)

    # Gamma values from the paper
    alpha = 0.5
    gamma_values = [k * alpha for k in range(201)]
    #gamma_values = [k * alpha for k in range(51)]

    # Include modes 0 through 5
    modes = [0, 1, 2, 3, 4, 5]

    # Set the perturbation constant
    perturb_const = 0.01  # q in paper

    # Set the tolerance
    tol = 0.00001

    # Nonlinearity powers
    nonlinearity_powers = [3]

    for p in nonlinearity_powers:

        # Specify potential type
        potential_type = "gravity_well"

        # Train neural network or load existing models
        train_new = False  # Set to True to train, False to load
        filename = f"tmp_mean.pkl"

        # Create plotting and model saving directory
        p_save_dir = f"tmp_mean"
        os.makedirs(p_save_dir, exist_ok=True)

        if train_new:
            # Train models
            print("Starting training...")
            models_by_mode, mu_table, training_history, constant_history, epochs_history = train_gpe_model(
                gamma_values, modes, p, X, lb, ub, layers, epochs, tol, perturb_const,
                potential_type='gravity_well', lr=1e-3, verbose=True)

            # Save results
            save_models(models_by_mode, mu_table, training_history, constant_history, epochs_history, filename, p_save_dir)
        else:
            # Load existing models
            print("Loading existing models...")
            models_by_mode, mu_table, training_history, constant_history, epochs_history = load_models(filename, p_save_dir)

        # Plot wavefunctions for individual modes
        print("Generating individual mode plots...")
        plot_wavefunction(models_by_mode, X_test, gamma_values, modes, p, constant_history, perturb_const,
                          potential_type, p_save_dir, normalization_type="unit")

        # Plot μ vs γ for all modes
        print("Generating chemical potential vs. gamma plot...")
        plot_mu_vs_gamma(mu_table, modes, p, potential_type, p_save_dir)
        #
        # # Plot combined loss history
        # print("Generating combined loss plots...")
        # plot_improved_loss_visualization(training_history, modes, gamma_values, epochs, p, potential_type, p_save_dir)
        # plot_all_modes_gamma_loss(training_history, modes, gamma_values, epochs, p, potential_type, p_save_dir)
        #
        # print("Generating early stopping analysis plots...")
        # plot_epochs_until_stopping(epochs_history, modes, gamma_values, p, potential_type, p_save_dir)

        # Decide if we want to run the comparison
        run_comparison = False

        if run_comparison:
            # Comparison parameters
            comparison_modes = modes
            comparison_gammas = gamma_values
            comparison_epochs = epochs

            print(f"Running comparison with:")
            print(f"  Modes: {comparison_modes}")
            print(f"  Gamma values: {comparison_gammas}")
            print(f"  Epochs per method: {comparison_epochs}")

            # Create comparison save directory
            comparison_save_dir = f"comparison_tmp_mean"

            # Run comparison with individual caching
            results_df, paper_results, best_results = create_comparison_table_individual_caching(
                comparison_modes, comparison_gammas, p, X, X_test, lb, ub, layers,
                comparison_epochs, save_dir=comparison_save_dir, tol=tol,
                perturb_const=perturb_const,
                force_retrain_regular=False,  # Set to True to retrain Regular PINNs
                force_retrain_curriculum=False,  # Set to True to retrain Curriculum PINNs
                potential_type=potential_type
            )

            # Create plots
            print("\nGenerating comparison plots...")
            plot_comparison_results(results_df, comparison_save_dir)

            # Create LaTeX table
            latex_file = os.path.join(comparison_save_dir, "comparison_table.tex")
            create_latex_table(results_df, latex_file)

            print("\n" + "=" * 80)
            print("COMPARISON STUDY COMPLETE!")
            print("=" * 80)
            print(f"Check '{comparison_save_dir}/' for:")
            print("  - Detailed CSV files with all results")
            print("  - Paper-style formatted results")
            print("  - Comparison plots")
            print("  - LaTeX table for publication")
            print("  - Best method analysis")
        else:
            print("Skipping comparison study.")

        print(f"Results saved to: {p_save_dir}/")