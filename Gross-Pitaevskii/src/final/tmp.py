import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import math
import numpy as np
import os
from scipy.special import hermite
from distributed_shampoo import AdamGraftingConfig, DistributedShampoo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import matplotlib.pyplot as plt
import scienceplots
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


class GrossPitaevskiiPINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) for solving the 1D Gross-Pitaevskii Equation.
    Fixed to match Algorithm 2 from the paper.
    """

    def __init__(self, layers, hbar=1.0, m=1.0, mode=0, gamma=1.0):
        super().__init__()
        self.layers = layers
        self.network = self.build_network()
        self.hbar = hbar
        self.m = m
        self.mode = mode
        self.gamma = gamma

    def build_network(self):
        """Build the neural network with tanh activation functions."""
        layers = []
        for i in range(len(self.layers) - 1):
            layers.append(nn.Linear(self.layers[i], self.layers[i + 1]))
            if i < len(self.layers) - 2:
                layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def weighted_hermite(self, x, n):
        """Compute the weighted Hermite polynomial solution for the linear case (gamma = 0)."""
        x_np = x.cpu().detach().numpy()
        H_n = hermite(n)(x_np)
        norm_factor = (2 ** n * math.factorial(n) * np.sqrt(np.pi)) ** (-0.5)
        weighted_hermite = norm_factor * torch.exp(-x ** 2 / 2) * torch.tensor(H_n, dtype=torch.float32).to(device)
        return weighted_hermite

    def forward(self, inputs):
        return self.network(inputs)

    def get_complete_solution(self, x, perturbation, mode=None):
        """Get the complete solution by combining the base Hermite solution with the neural network perturbation."""
        if mode is None:
            mode = self.mode
        base_solution = self.weighted_hermite(x, mode)
        return base_solution + perturbation

    def compute_potential(self, x, potential_type="harmonic", **kwargs):
        """Compute potential function for the 1D domain."""
        if potential_type == "harmonic":
            omega = kwargs.get('omega', 1.0)
            V = 0.5 * omega ** 2 * x ** 2
        elif potential_type == "gaussian":
            a = kwargs.get('a', 0.0)
            V = torch.exp(-(x - a) ** 2)
        elif potential_type == "periodic":
            V0 = kwargs.get('V0', 1.0)
            k = kwargs.get('k', 2 * np.pi / 5.0)
            V = V0 * torch.cos(k * x) ** 2
        else:
            raise ValueError(f"Unknown potential type: {potential_type}")
        return V

    def pde_loss(self, inputs, predictions, gamma, potential_type="harmonic", precomputed_potential=None):
        """
        Compute the PDE loss for the Gross-Pitaevskii equation with CORRECT nonlinearity.
        μψ = -1/2 ∇²ψ + Vψ + γ|ψ|²ψ  (NOT γ|ψ|^p)
        """
        u = self.get_complete_solution(inputs, predictions)

        # Compute derivatives
        u_x = torch.autograd.grad(
            outputs=u, inputs=inputs, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]

        u_xx = torch.autograd.grad(
            outputs=u_x, inputs=inputs, grad_outputs=torch.ones_like(u_x),
            create_graph=True, retain_graph=True
        )[0]

        # Compute potential
        if precomputed_potential is not None:
            V = precomputed_potential
        else:
            V = self.compute_potential(inputs, potential_type)

        # CORRECTED: Use standard GPE nonlinearity γ|ψ|²ψ
        kinetic = -0.5 * u_xx
        potential = V * u
        interaction = gamma * (u.abs() ** 2) * u  # This is |ψ|²ψ

        # Calculate chemical potential using variational principle
        numerator = torch.mean(u * (kinetic + potential + interaction))
        denominator = torch.mean(u ** 2)
        lambda_pde = numerator / denominator

        # PDE residual
        pde_residual = kinetic + potential + interaction - lambda_pde * u
        pde_loss = torch.mean(pde_residual ** 2)

        # Add the base energy for the mode
        lambda_pde = lambda_pde + self.mode + 0.5

        return pde_loss, pde_residual, lambda_pde, u

    def riesz_loss(self, inputs, predictions, gamma, potential_type="harmonic", precomputed_potential=None):
        """
        Compute the CORRECT Riesz energy loss for the Gross-Pitaevskii equation.
        E[ψ] = ∫[|∇ψ|²/2 + V|ψ|² + γ|ψ|⁴/2]dx  (NOT generalized p-power)
        """
        u = self.get_complete_solution(inputs, predictions)

        if not inputs.requires_grad:
            inputs = inputs.clone().detach().requires_grad_(True)

        u_x = torch.autograd.grad(
            outputs=u, inputs=inputs, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]

        # Calculate normalization factor
        dx = inputs[1] - inputs[0]
        norm_factor = torch.sum(u ** 2) * dx

        # CORRECTED: Use standard GPE energy functional
        kinetic_term = 0.5 * torch.sum(u_x ** 2) * dx / norm_factor

        if precomputed_potential is not None:
            V = precomputed_potential
        else:
            V = self.compute_potential(inputs, potential_type)
        potential_term = torch.sum(V * u ** 2) * dx / norm_factor

        # CORRECTED: Use γ|ψ|⁴/2 (standard GPE)
        interaction_term = 0.5 * gamma * torch.sum(u ** 4) * dx / norm_factor

        riesz_energy = kinetic_term + potential_term + interaction_term

        # Chemical potential includes base energy
        lambda_riesz = riesz_energy + self.mode + 0.5

        return riesz_energy, lambda_riesz, u

    def boundary_loss(self, boundary_points, boundary_values):
        """Compute the boundary loss for the boundary conditions."""
        u_pred = self.forward(boundary_points)
        full_u = self.get_complete_solution(boundary_points, u_pred)
        return torch.mean((full_u - boundary_values) ** 2)

    def symmetry_loss(self, collocation_points, lb, ub):
        """Compute the symmetry loss to enforce proper parity."""
        x_reflected = -collocation_points

        u_pred_original = self.forward(collocation_points)
        u_full_original = self.get_complete_solution(collocation_points, u_pred_original)

        u_pred_reflected = self.forward(x_reflected)
        u_full_reflected = self.get_complete_solution(x_reflected, u_pred_reflected)

        # Apply correct symmetry condition based on mode parity
        if self.mode % 2 == 1:
            # Odd modes: u(x) = -u(-x)
            sym_loss = torch.mean((u_full_original + u_full_reflected) ** 2)
        else:
            # Even modes: u(x) = u(-x)
            sym_loss = torch.mean((u_full_original - u_full_reflected) ** 2)

        return sym_loss

    def normalization_loss(self, u, dx):
        """Compute normalization loss using proper numerical integration."""
        integral = torch.sum(u ** 2) * dx
        return (integral - 1.0) ** 2

    def orthogonality_loss(self, u, dx, mode):
        """
        Enforce orthogonality with lower modes (especially important for higher modes).
        """
        if mode == 0:
            return torch.tensor(0.0, device=device)

        ortho_loss = torch.tensor(0.0, device=device)

        # Check orthogonality with lower modes
        for lower_mode in range(mode):
            lower_hermite = self.weighted_hermite(
                torch.linspace(-10, 10, len(u), device=device).reshape(-1, 1),
                lower_mode
            )
            # Normalize lower_hermite
            lower_norm = torch.sqrt(torch.sum(lower_hermite ** 2) * dx)
            lower_hermite = lower_hermite / lower_norm

            # Compute overlap
            overlap = torch.sum(u * lower_hermite) * dx
            ortho_loss += overlap ** 2

        return ortho_loss


def advanced_initialization(m, mode):
    """Enhanced initialization considering mode number and network depth."""
    if isinstance(m, nn.Linear):
        # Use He initialization with mode-dependent scaling
        fan_in = m.weight.size(1)
        std = math.sqrt(2.0 / fan_in)

        # Scale initialization based on mode
        if mode == 0:
            scale = 1.0
        elif mode <= 2:
            scale = 0.5
        else:
            scale = 0.1 / (1 + 0.1 * mode)  # Smaller for higher modes

        m.weight.data.normal_(0, std * scale)

        # Bias initialization
        if mode > 2:
            m.bias.data.fill_(0.001)
        else:
            m.bias.data.fill_(0.01)


def train_gpe_model(gamma_values, modes, X_train, lb, ub, layers, epochs,
                    potential_type='harmonic', lr=1e-3, verbose=True,
                    min_epochs=3000, patience=500, min_delta=1e-6):
    """
    Enhanced training function with improved loss weighting and convergence strategies.
    """
    dx = X_train[1, 0] - X_train[0, 0]
    X_tensor = torch.tensor(X_train, dtype=torch.float32, requires_grad=True).to(device)

    boundary_points = torch.tensor([[lb], [ub]], dtype=torch.float32).to(device)
    boundary_values = torch.zeros((2, 1), dtype=torch.float32).to(device)

    models_by_mode = {}
    mu_table = {}
    training_history = {}

    gamma_values = sorted(gamma_values)

    # Precompute potential
    temp_model = GrossPitaevskiiPINN(layers).to(device)
    precomputed_potential = temp_model.compute_potential(X_tensor, potential_type).detach()

    for mode in modes:
        if verbose:
            print(f"\n===== Training for mode {mode} =====")

        mu_logs = []
        models_by_gamma = {}
        history_by_gamma = {}
        prev_model = None

        for gamma in gamma_values:
            if verbose:
                print(f"\nTraining for γ = {gamma:.2f}, mode = {mode}, potential = {potential_type}")

            model = GrossPitaevskiiPINN(layers, mode=mode, gamma=gamma).to(device)

            # Initialize with previous model if available
            if prev_model is not None:
                model.load_state_dict(prev_model.state_dict())
            else:
                model.apply(lambda m: advanced_initialization(m, mode))

            # Use different optimizers for different modes
            if mode <= 1:
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            else:
                # Lower learning rate for higher modes
                optimizer = torch.optim.Adam(model.parameters(), lr=lr * 0.5)

            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=200, T_mult=2, eta_min=1e-7
            )

            lambda_history = []
            loss_history = []
            constraint_history = []

            best_loss = float('inf')
            epochs_without_improvement = 0
            best_model_state = None

            for epoch in range(epochs):
                optimizer.zero_grad()

                u_pred = model.forward(X_tensor)

                # Enhanced loss weighting based on mode
                if mode == 0:
                    boundary_weight = 10.0
                    norm_weight = 20.0
                    sym_weight = 5.0
                    ortho_weight = 0.0
                else:
                    # Higher weights for constraints in higher modes
                    boundary_weight = 20.0 * (1 + 0.5 * mode)
                    norm_weight = 30.0 * (1 + 0.3 * mode)
                    sym_weight = 10.0 * (1 + 0.2 * mode)
                    ortho_weight = 50.0  # Strong orthogonality enforcement

                boundary_loss = model.boundary_loss(boundary_points, boundary_values)
                full_u = model.get_complete_solution(X_tensor, u_pred)
                norm_loss = model.normalization_loss(full_u, dx)
                sym_loss = model.symmetry_loss(X_tensor, lb, ub)
                ortho_loss = model.orthogonality_loss(full_u, dx, mode)

                constraint_loss = (boundary_weight * boundary_loss +
                                   norm_weight * norm_loss +
                                   sym_weight * sym_loss +
                                   ortho_weight * ortho_loss)

                # Choose loss type based on mode
                if mode == 0:
                    riesz_energy, lambda_value, _ = model.riesz_loss(
                        X_tensor, u_pred, gamma, potential_type, precomputed_potential
                    )
                    physics_loss = riesz_energy
                    monitoring_loss = constraint_loss.item()
                else:
                    pde_loss, _, lambda_value, _ = model.pde_loss(
                        X_tensor, u_pred, gamma, potential_type, precomputed_potential
                    )
                    physics_loss = pde_loss
                    monitoring_loss = pde_loss.item()

                total_loss = physics_loss + constraint_loss

                total_loss.backward()

                # Gradient clipping (more aggressive for higher modes)
                clip_value = 1.0 if mode <= 2 else 0.5
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

                optimizer.step()
                scheduler.step(total_loss)

                # Record history
                if epoch % 100 == 0:
                    lambda_history.append(lambda_value.item())
                    loss_history.append(total_loss.item())
                    constraint_history.append(monitoring_loss)

                    if verbose and epoch % 500 == 0:
                        print(f"Epoch {epoch}, Physics Loss: {physics_loss.item():.6f}, "
                              f"Constraint: {constraint_loss.item():.6f}, μ: {lambda_value.item():.4f}")

                # Early stopping logic
                if epoch >= min_epochs:
                    current_loss = total_loss.item()
                    if current_loss < best_loss - min_delta:
                        best_loss = current_loss
                        epochs_without_improvement = 0
                        best_model_state = model.state_dict().copy()
                    else:
                        epochs_without_improvement += 1

                    if epochs_without_improvement >= patience:
                        if verbose:
                            print(f"    Early stopping at epoch {epoch}")
                        break
                else:
                    current_loss = total_loss.item()
                    if current_loss < best_loss:
                        best_loss = current_loss
                        best_model_state = model.state_dict().copy()

            # Load best model state
            if best_model_state is not None:
                model.load_state_dict(best_model_state)

            final_mu = lambda_history[-1] if lambda_history else 0
            mu_logs.append((gamma, final_mu))
            models_by_gamma[gamma] = model

            history_by_gamma[gamma] = {
                'loss': loss_history,
                'constraint': constraint_history,
                'lambda': lambda_history,
                'final_epoch': epoch + 1,
                'best_loss': best_loss
            }

            prev_model = model

        mu_table[mode] = mu_logs
        models_by_mode[mode] = models_by_gamma
        training_history[mode] = history_by_gamma

    return models_by_mode, mu_table, training_history

def plot_wavefunction(models_by_mode, X_test, gamma_values, modes, p, lb, ub, potential_type = "box", save_dir="harmonic_plots"):
    """
    Plot wavefunctions (not densities) for different modes and gamma values.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Generate individual figures for each mode
    for mode in modes:
        if mode not in models_by_mode:
            continue

        # Create individual figure
        plt.figure(figsize=(8, 6))

        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        dx = X_test[1, 0] - X_test[0, 0]  # Spatial step size

        # Different line styles and colors
        linestyles = ['-', '--', '-.', ':', '-', '--']
        colors = ['k', 'b', 'r', 'g', 'm', 'c', 'slategray']

        # Plot solutions for different gamma values
        for j, gamma in enumerate(gamma_values):
            if gamma not in models_by_mode[mode]:
                continue

            model = models_by_mode[mode][gamma]
            model.eval()

            with torch.no_grad():
                u_pred = model.forward(X_tensor)
                full_u = model.get_complete_solution(X_tensor, u_pred)
                u_np = full_u.cpu().numpy().flatten()

                # Normalization
                u_np /= np.sqrt(np.sum(u_np ** 2) * dx)

                # For mode 0, ensure all wavefunctions are positive
                if mode == 0:
                    # Take absolute value to ensure positive values
                    # This is valid for ground state (mode 0) which should be nodeless
                    u_np = np.abs(u_np)

                # Plot wavefunction (not density)
                plt.plot(X_test.flatten(), u_np,
                         linestyle=linestyles[j % len(linestyles)],
                         color=colors[j % len(colors)],
                         label=f"γ={gamma:.1f}")

        # Configure individual figure
        plt.title(f"Mode {mode} Wavefunction", fontsize=18)
        plt.xlabel("x", fontsize=18)
        plt.ylabel(r"$\psi(x)$", fontsize=18)
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.xlim(lb, ub)  # Set x limits to match domain
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"mode_{mode}_wavefunction_p{p}_{potential_type}.png"), dpi=300)
        plt.close()

    # Also create a combined grid figure to show all modes
    plot_combined_grid(models_by_mode, X_test, gamma_values, modes, p, lb, ub, potential_type, save_dir)


def plot_combined_grid(models_by_mode, X_test, gamma_values, modes, p, lb, ub, potential_type="box", save_dir="harmonic_plots"):
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

            with torch.no_grad():
                u_pred = model.forward(X_tensor)
                full_u = model.get_complete_solution(X_tensor, u_pred)
                u_np = full_u.cpu().numpy().flatten()

                # Proper normalization
                u_np /= np.sqrt(np.sum(u_np ** 2) * dx)

                # For mode 0, ensure all wavefunctions are positive
                if mode == 0:
                    # Take absolute value to ensure positive values
                    u_np = np.abs(u_np)

                # Plot the wavefunction (not density)
                ax.plot(X_test.flatten(), u_np,
                        linestyle=linestyles[j % len(linestyles)],
                        color=colors[j % len(colors)],
                        label=f"γ={gamma:.1f}")

        # Configure the subplot
        ax.set_title(f"mode {mode}", fontsize=12)
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel(r"$\psi(x)$", fontsize=12)
        ax.grid(True)
        ax.legend(fontsize=8)
        ax.set_xlim(lb, ub)

    # Hide any unused subplots
    for i in range(len(modes), len(axes)):
        axes[i].axis('off')

    # Finalize and save combined figure
    fig.suptitle(f"Wavefunctions for All Modes (p={p})", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.legend(fontsize=8)
    fig.savefig(os.path.join(save_dir, f"all_modes_combined_wavefunctions_p{p}_{potential_type}.png"), dpi=300)
    plt.close(fig)


def plot_mu_vs_gamma(mu_table, modes, p, potential_type="box", save_dir="harmonic_plots"):
    """
    Plot chemical potential vs. interaction strength for different modes.
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
        plt.plot(gamma_list, mu_list,
                 marker=markers[i % len(markers)],
                 color=colors[i % len(colors)],
                 linestyle='-',
                 label=f"Mode {mode}")

    plt.xlabel(r"$\gamma$ (Interaction Strength)", fontsize=18)
    plt.ylabel(r"$\mu$ (Chemical Potential)", fontsize=18)
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


def plot_all_modes_gamma_loss(training_history, modes, gamma_values, epochs,  p, potential_type, save_dir="box_plots"):
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
                ax.semilogy(epoch_nums, loss_history,
                            color=colors[j % len(colors)],
                            linestyle=linestyles[j % len(linestyles)],
                            label=f"γ={gamma:.1f}")

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


def moving_average(values, window_size=10):
    """Apply moving average smoothing to a list of values"""
    if len(values) < window_size:
        return values
    weights = np.ones(window_size) / window_size
    return np.convolve(values, weights, mode='valid')


def plot_improved_loss_visualization(training_history, modes, gamma_values, epochs, p, potential_type,
                                     save_dir="box_plots"):
    """
    Creates informative and smoother visualizations of the training progress.
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. Separate plots by loss type with smoothing
    plt.figure(figsize=(12, 6))

    # Plot for Mode 0 (energy minimization)
    plt.subplot(1, 2, 1)
    for gamma in gamma_values:
        if 0 in training_history and gamma in training_history[0]:
            # Get loss history for mode 0
            loss_history = training_history[0][gamma]['loss']

            # Apply smoothing to the loss data
            window_size = min(30, len(loss_history) // 10)  # Adaptive window size
            if window_size > 1:
                smooth_loss = moving_average(loss_history, window_size)
                # Adjust epoch numbers to match the smoothed array length
                epoch_nums = np.linspace(0, epochs, len(smooth_loss))
                plt.semilogy(epoch_nums, smooth_loss, label=f"γ={gamma:.1f}")
            else:
                epoch_nums = np.linspace(0, epochs, len(loss_history))
                plt.semilogy(epoch_nums, loss_history, label=f"γ={gamma:.1f}")

    plt.title("Mode 0: Energy Functional Minimization", fontsize=18)
    plt.xlabel("Epochs", fontsize=18)
    plt.ylabel("Energy Functional", fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=12)

    # Plot for Modes 1+ (PDE residual minimization)
    plt.subplot(1, 2, 2)
    for mode in modes:
        if mode == 0:
            continue  # Skip mode 0 for this plot

        for gamma in [0.0]:  # Focus on γ=0 for clarity
            if mode in training_history and gamma in training_history[mode]:
                loss_history = training_history[mode][gamma]['loss']

                # Apply smoothing to the loss data
                window_size = min(30, len(loss_history) // 10)  # Adaptive window size
                if window_size > 1:
                    smooth_loss = moving_average(loss_history, window_size)
                    epoch_nums = np.linspace(0, epochs, len(smooth_loss))
                    plt.semilogy(epoch_nums, smooth_loss, label=f"Mode {mode}")
                else:
                    epoch_nums = np.linspace(0, epochs, len(loss_history))
                    plt.semilogy(epoch_nums, loss_history, label=f"Mode {mode}")

    plt.title(r"Modes 1-5: PDE Residual Minimization", fontsize=18)
    plt.xlabel("Epochs", fontsize=18)
    plt.ylabel("PDE Residual", fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"separated_loss_types_p{p}_{potential_type}.png"), dpi=300)
    plt.close()

    # 2. Plot chemical potential convergence with smoothing
    plt.figure(figsize=(10, 6))
    for mode in modes:
        for gamma in [0.0]:  # Focus on γ=0 for clarity
            if mode in training_history and gamma in training_history[mode]:
                lambda_history = training_history[mode][gamma]['lambda']

                # Apply smoothing to the chemical potential data
                window_size = min(20, len(lambda_history) // 10)  # Smaller window for μ
                if window_size > 1:
                    smooth_lambda = moving_average(lambda_history, window_size)

                    # For γ=0, the theoretical value should be mode + 0.5
                    theoretical_value = mode + 0.5

                    # Calculate relative error using smoothed values
                    relative_error = [abs(l - theoretical_value) / theoretical_value for l in smooth_lambda]
                    epoch_nums = np.linspace(0, epochs, len(smooth_lambda))
                    plt.semilogy(epoch_nums, relative_error, label=f"Mode {mode}")
                else:
                    theoretical_value = mode + 0.5
                    relative_error = [abs(l - theoretical_value) / theoretical_value for l in lambda_history]
                    epoch_nums = np.linspace(0, epochs, len(lambda_history))
                    plt.semilogy(epoch_nums, relative_error, label=f"Mode {mode}")

    plt.title(r"Chemical Potential Convergence", fontsize=18)
    plt.xlabel("Epochs", fontsize=18)
    plt.ylabel(r"Relative Error in $\mu$", fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"chemical_potential_convergence_p{p}_{potential_type}.png"), dpi=300)
    plt.close()

    # 3. Plot normalized loss (as percentage of initial loss) with smoothing
    plt.figure(figsize=(10, 6))
    for mode in modes:
        for gamma in [0.0]:  # Focus on γ=0 for clarity
            if mode in training_history and gamma in training_history[mode]:
                loss_history = training_history[mode][gamma]['loss']
                initial_loss = loss_history[0]
                normalized_loss = [l / initial_loss for l in loss_history]

                # Apply smoothing to the normalized loss
                window_size = min(30, len(normalized_loss) // 10)
                if window_size > 1:
                    smooth_norm_loss = moving_average(normalized_loss, window_size)
                    epoch_nums = np.linspace(0, epochs, len(smooth_norm_loss))
                    plt.semilogy(epoch_nums, smooth_norm_loss, label=f"Mode {mode}")
                else:
                    epoch_nums = np.linspace(0, epochs, len(normalized_loss))
                    plt.semilogy(epoch_nums, normalized_loss, label=f"Mode {mode}")

    plt.title(r"Normalized Loss Convergence", fontsize=18)
    plt.xlabel("Epochs", fontsize=18)
    plt.ylabel("Loss / Initial Loss", fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"normalized_loss_p{p}_{potential_type}.png"), dpi=300)
    plt.close()

    # 4. Add a new visualization: Smoothed overall training progress across modes
    plt.figure(figsize=(12, 8))
    max_mode = max(modes) if modes else 0
    colormap = plt.cm.viridis
    colors = [colormap(i / max_mode) for i in modes]

    for i, mode in enumerate(modes):
        for gamma in [0.0]:  # Focus on γ=0 for clarity
            if mode in training_history and gamma in training_history[mode]:
                loss_history = training_history[mode][gamma]['loss']

                # For ultra-smooth visualization, use a larger window
                window_size = min(50, len(loss_history) // 5)
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
    plt.savefig(os.path.join(save_dir, f"ultra_smooth_training_progress_p{p}_{potential_type}.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    # Setup parameters
    lb, ub = -10, 10
    N_f = 4000
    epochs = 6000
    layers = [1, 64, 64, 64, 64, 1]  # Deeper network for higher modes
    # Create uniform grid for training and testing
    X = np.linspace(lb, ub, N_f).reshape(-1, 1)
    X_test = np.linspace(lb, ub, 1000).reshape(-1, 1)  # Higher resolution for plotting

    # Gamma values from the paper
    gamma_values = [0.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0]

    # Include modes 0 through 5
    modes = [0, 1, 2, 3, 4, 5]

    # Nonlinearity powers
    nonlinearity_powers = [3]

    for p in nonlinearity_powers:

        # all_potentials = ['box', 'harmonic']
        all_potentials = ['harmonic']

        for potential_type in all_potentials:

            # Create a specific directory for this p value and potential type
            p_save_dir = f"plots_p{p}_{potential_type}_tmp"
            os.makedirs(p_save_dir, exist_ok=True)

            # Train models
            print(f"\nStarting training for {potential_type} potential for all modes and gamma values with p={p}...")
            models_by_mode, mu_table, training_history = train_gpe_model(
                gamma_values, modes, X, lb, ub, layers, epochs,
                potential_type='harmonic', lr=1e-3, verbose=True
            )
            print("Training completed!")

            # Plot wavefunctions for individual modes
            print("Generating individual mode plots...")
            plot_wavefunction(models_by_mode, X_test, gamma_values, modes, p, lb, ub, potential_type, p_save_dir)

            # Plot μ vs γ for all modes
            print("Generating chemical potential vs. gamma plot...")
            plot_mu_vs_gamma(mu_table, modes, p, potential_type, p_save_dir)

            # Plot loss history
            print("Generating loss plots...")
            plot_improved_loss_visualization(training_history, modes, gamma_values, epochs, p, potential_type,
                                             p_save_dir)
            plot_all_modes_gamma_loss(training_history, modes, gamma_values, epochs, p, potential_type, p_save_dir)

            print(f"Completed all calculations for {potential_type} potential\n")

            print("\nChemical potentials at γ=0 (should be n+0.5):")
            for mode in modes:
                mu_at_gamma_0 = mu_table[mode][0][1]  # First gamma (0.0), mu value
                expected = mode + 0.5
                print(f"Mode {mode}: μ = {mu_at_gamma_0:.4f} (expected {expected:.1f})")