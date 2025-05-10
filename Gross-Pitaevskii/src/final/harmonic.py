import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import math
import numpy as np
import os
import matplotlib.pyplot as plt
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
    """

    def __init__(self, layers, hbar=1.0, m=1.0, mode=0, gamma=1.0):
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
        """
        super().__init__()
        self.layers = layers
        self.network = self.build_network()
        self.hbar = hbar  # Planck's constant, fixed
        self.m = m  # Particle mass, fixed
        self.mode = mode  # Mode number (n)
        self.gamma = gamma  # Interaction strength parameter

    def build_network(self):
        """
        Build the neural network with tanh activation functions between layers.
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
        """
        H_n = hermite(n)(x.cpu().detach().numpy())  # Hermite polynomial evaluated at x
        norm_factor = (2 ** n * math.factorial(n) * np.sqrt(np.pi)) ** (-0.5)
        weighted_hermite = norm_factor * torch.exp(-x ** 2 / 2) * torch.tensor(H_n, dtype=torch.float32).to(device)
        return weighted_hermite

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
        base_solution = self.weighted_hermite(x, mode)
        return base_solution + perturbation

    def compute_potential(self, x, potential_type="harmonic", **kwargs):
        """
        Compute potential function for the 1D domain.
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

    def pde_loss(self, inputs, predictions, gamma, potential_type="harmonic", precomputed_potential=None):
        """
        Compute the PDE loss for the Gross-Pitaevskii equation.
        μψ = -1/2 ∇²ψ + Vψ + γ|ψ|²ψ
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
        kinetic = -0.5 * u_xx
        potential = V * u
        interaction = gamma * u ** 3

        numerator = torch.mean(u * (kinetic + potential + interaction))
        denominator = torch.mean(u ** 2)
        lambda_pde = numerator / denominator

        # Residual of the 1D Gross-Pitaevskii equation
        pde_residual = kinetic + potential + interaction - lambda_pde * u

        # PDE loss (mean squared residual)
        pde_loss = torch.mean(pde_residual ** 2)

        lambda_pde = lambda_pde + self.mode

        return pde_loss, pde_residual, lambda_pde, u

    def riesz_loss(self, inputs, predictions, gamma, potential_type="harmonic", precomputed_potential=None):
        """
        Compute the Riesz energy loss for the Gross-Pitaevskii equation.
        E[ψ] = ∫[|∇ψ|²/2 + V|ψ|² + γ|ψ|⁴/2]dx

        This corresponds to Algorithm 2 in the paper at https://arxiv.org/pdf/1208.2123
        """
        # Get the complete solution (base + perturbation)
        u = self.get_complete_solution(inputs, predictions)

        # Ensure inputs require gradients for autograd
        if not inputs.requires_grad:
            inputs = inputs.clone().detach().requires_grad_(True)

        # Compute derivative with respect to x
        u_x = torch.autograd.grad(
            outputs=u,
            inputs=inputs,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]

        # Calculate normalization factor for proper numerical integration
        dx = inputs[1] - inputs[0]  # Grid spacing
        norm_factor = torch.sum(u ** 2) * dx

        # Compute each term in the energy functional with proper normalization

        # Kinetic energy term: |∇ψ|²/2 with proper normalization
        kinetic_term = 0.5 * torch.sum(u_x ** 2) * dx / norm_factor

        # Potential term: V|ψ|² with proper normalization
        if precomputed_potential is not None:
            V = precomputed_potential
        else:
            V = self.compute_potential(inputs, potential_type)
        potential_term = torch.sum(V * u ** 2) * dx / norm_factor

        # Interaction term: γ|ψ|⁴/2 with proper normalization
        interaction_term = 0.5 * gamma * torch.sum(u ** 4) * dx / norm_factor

        # Total Riesz energy functional
        riesz_energy = kinetic_term + potential_term + interaction_term

        # Calculate chemical potential using variational approach
        # For the harmonic oscillator ground state (mode 0), μ should be 0.5 when γ = 0
        lambda_riesz = riesz_energy + self.mode

        return riesz_energy, lambda_riesz, u

    def boundary_loss(self, boundary_points, boundary_values):
        """
        Compute the boundary loss for the boundary conditions.
        """
        u_pred = self.forward(boundary_points)
        full_u = self.get_complete_solution(boundary_points, u_pred)
        return torch.mean((full_u - boundary_values) ** 2)

    def symmetry_loss(self, collocation_points, lb, ub):
        """
        Compute the symmetry loss to enforce u(x) = u(-x) for even modes
        and u(x) = -u(-x) for odd modes.
        """
        # For symmetric potential around x=0, we reflect around 0
        x_reflected = -collocation_points

        # Evaluate u(x) and u(-x) for the FULL solution
        u_pred_original = self.forward(collocation_points)
        u_full_original = self.get_complete_solution(collocation_points, u_pred_original)

        u_pred_reflected = self.forward(x_reflected)
        u_full_reflected = self.get_complete_solution(x_reflected, u_pred_reflected)

        # For odd modes, apply anti-symmetry condition
        if self.mode % 2 == 1:
            sym_loss = torch.mean((u_full_original + u_full_reflected) ** 2)
        else:
            sym_loss = torch.mean((u_full_original - u_full_reflected) ** 2)

        return sym_loss

    def normalization_loss(self, u, dx):
        """
        Compute normalization loss using proper numerical integration.
        """
        integral = torch.sum(u ** 2) * dx
        return (integral - 1.0) ** 2


def initialize_weights(m):
    """
    Initialize the weights using Xavier uniform initialization.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def train_gpe_model(gamma_values, modes, X_train, lb, ub, layers, epochs,
                    potential_type='harmonic', lr=1e-3, verbose=True):
    """
    Train the GPE model for different modes and gamma values.

    Parameters:
    -----------
    gamma_values : list of float
        List of interaction strengths to train models for
    modes : list of int
        List of modes to train (0, 1, 2, 3, etc.)
    X_train : numpy.ndarray
        Training points array
    lb, ub : float
        Lower and upper boundaries of the domain
    layers : list of int
        Network architecture
    epochs : int
        Number of training epochs
    potential_type : str
        Type of potential ('harmonic', 'gaussian', etc.)
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

    # Create boundary conditions
    boundary_points = torch.tensor([[lb], [ub]], dtype=torch.float32).to(device)
    boundary_values = torch.zeros((2, 1), dtype=torch.float32).to(device)

    # Track models, chemical potentials, and training history
    models_by_mode = {}
    mu_table = {}
    training_history = {}

    # Sort gamma values
    gamma_values = sorted(gamma_values)

    for mode in modes:
        if verbose:
            print(f"\n===== Training for mode {mode} =====")

        mu_logs = []
        models_by_gamma = {}
        history_by_gamma = {}
        prev_model = None

        for gamma in gamma_values:
            if verbose:
                print(f"\nTraining for γ = {gamma:.2f}, mode = {mode}")

            # Initialize model for this mode and gamma
            model = GrossPitaevskiiPINN(layers, mode=mode, gamma=gamma).to(device)

            # If this isn't the first gamma value, initialize with previous model's weights
            if prev_model is not None:
                model.load_state_dict(prev_model.state_dict())
            else:
                # Use the advanced initialization that considers mode number
                model.apply(lambda m: advanced_initialization(m, mode))

            # Adam optimizer
            #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            optimizer = DistributedShampoo(model.parameters(),
                                           lr=lr,
                                           betas=(0.9, 0.999),
                                           epsilon=1e-12,
                                           weight_decay=1e-05,
                                           max_preconditioner_dim=8192,
                                           start_preconditioning_step=100,
                                           precondition_frequency=100,
                                           use_decoupled_weight_decay=False,
                                           grafting_config=AdamGraftingConfig(beta2=0.999, epsilon=1e-08),
                                           )

            # Create scheduler to decrease learning rate during training
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=200, T_mult=2, eta_min=1e-6
            )

            # Track learning history
            lambda_history = []
            loss_history = []
            constraint_history = []

            for epoch in range(epochs):
                optimizer.zero_grad()

                # Forward pass
                u_pred = model.forward(X_tensor)

                # Calculate common constraint losses for all modes
                boundary_loss = model.boundary_loss(boundary_points, boundary_values)
                norm_loss = model.normalization_loss(model.get_complete_solution(X_tensor, u_pred), dx)
                sym_loss = model.symmetry_loss(X_tensor, lb, ub)

                # Combined constraint loss
                constraint_loss = 10.0 * boundary_loss + 20.0 * norm_loss + 5.0 * sym_loss

                # Decide which loss to use based on mode
                if mode == 0:
                    # Use Riesz energy functional for mode 0 as specified in the paper (Algorithm 2)
                    riesz_energy, lambda_value, full_u = model.riesz_loss(
                        X_tensor, u_pred, gamma, potential_type
                    )

                    pde_loss, _, _, _ = model.pde_loss(
                        X_tensor, u_pred, gamma, potential_type
                    )

                    # For mode 0, we want to minimize the energy while satisfying constraints
                    # Note: We don't expect the energy to go to zero, it should converge to the ground state energy
                    physics_loss = riesz_energy + pde_loss
                    loss_type = "Riesz energy"

                    # Track the constraints separately for monitoring
                    monitoring_loss = constraint_loss.item()
                else:
                    # Use PDE residual for other modes
                    pde_loss, _, lambda_value, full_u = model.pde_loss(
                        X_tensor, u_pred, gamma, potential_type
                    )
                    physics_loss = pde_loss
                    loss_type = "PDE residual"

                    # For PDE residual, we do expect the residual to approach zero
                    monitoring_loss = pde_loss.item()

                # Total loss for optimization
                total_loss = physics_loss + constraint_loss

                # Backpropagate
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
                optimizer.step()
                scheduler.step(total_loss)

                # Record history
                if epoch % 100 == 0:
                    lambda_history.append(lambda_value.item())
                    loss_history.append(total_loss.item())
                    constraint_history.append(monitoring_loss)

                    if verbose and epoch % 500 == 0:
                        if mode == 0:
                            print(f"Epoch {epoch}, {loss_type}: {physics_loss.item():.6f}, "
                                  f"Constraints: {monitoring_loss:.6f}, μ: {lambda_value.item():.4f}")
                        else:
                            print(
                                f"Epoch {epoch}, {loss_type}: {physics_loss.item():.6f}, μ: {lambda_value.item():.4f}")

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

            # Update prev_model for next gamma value
            prev_model = model

        # Store results for this mode
        mu_table[mode] = mu_logs
        models_by_mode[mode] = models_by_gamma
        training_history[mode] = history_by_gamma

    return models_by_mode, mu_table, training_history


def plot_wavefunction(models_by_mode, X_test, gamma_values, modes, save_dir="tmp"):
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
        plt.xlim(-10, 10)  # Match paper's range
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"mode_{mode}_wavefunction.png"), dpi=300)
        plt.close()

    # Also create a combined grid figure to show all modes
    plot_combined_grid(models_by_mode, X_test, gamma_values, modes, save_dir)


def plot_combined_grid(models_by_mode, X_test, gamma_values, modes, save_dir="tmp"):
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
        ax.legend(fontsize=6)
        ax.set_xlim(-10, 10)

    # Hide any unused subplots
    for i in range(len(modes), len(axes)):
        axes[i].axis('off')

    # Finalize and save combined figure
    fig.suptitle("Wavefunctions for All Modes", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(save_dir, "all_modes_combined_wavefunctions.png"), dpi=300)
    plt.close(fig)


def plot_mu_vs_gamma(mu_table, modes, save_dir="tmp"):
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
    plt.title("Chemical Potential vs. Interaction Strength for All Modes", fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "mu_vs_gamma_all_modes.png"), dpi=300)
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


def plot_combined_loss_history(training_history, modes, gamma_values, epochs, save_dir="tmp"):
    """
    Plot the training loss history for all modes on a single log-scale plot.

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

    # Create a separate plot for each gamma value
    for gamma in gamma_values:
        plt.figure(figsize=(8, 6))

        # Different line styles and colors for different modes
        linestyles = ['-', '--', '-.', ':', '-', '--']
        colors = ['k', 'b', 'r', 'g', 'm', 'c', 'slategray']

        # Plot loss for each mode
        for i, mode in enumerate(modes):
            if mode in training_history and gamma in training_history[mode]:
                # Get loss history for this mode and gamma
                loss_history = training_history[mode][gamma]['loss']

                # X-axis values (epoch numbers)
                epoch_nums = np.linspace(0, epochs, len(loss_history))

                # Plot loss on log scale
                plt.semilogy(epoch_nums, loss_history,
                             color=colors[i % len(colors)],
                             linestyle=linestyles[i % len(linestyles)],
                             linewidth=1.5,
                             label=f'Mode {mode}')

        # Configure plot
        plt.xlabel("Epochs", fontsize=18)
        plt.ylabel("Loss", fontsize=18)
        plt.title(f"Training Loss (γ={gamma})", fontsize=18)

        # Add grid for better readability
        plt.grid(True)

        # Add legend
        plt.legend(fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"combined_loss_gamma{gamma}.png"), dpi=300)
        plt.close()


def plot_all_modes_gamma_loss(training_history, modes, gamma_values, epochs, save_dir="tmp"):
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
    fig.savefig(os.path.join(save_dir, "all_modes_gamma_loss_subplots.png"), dpi=300)
    plt.close(fig)


def plot_improved_loss_visualization(training_history, modes, gamma_values, epochs, save_dir="tmp"):
    """
    Creates informative visualizations of the training progress.
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. Separate plots by loss type
    plt.figure(figsize=(12, 6))

    # Plot for Mode 0 (energy minimization)
    plt.subplot(1, 2, 1)
    for gamma in gamma_values:
        if 0 in training_history and gamma in training_history[0]:
            # Get loss history for mode 0
            loss_history = training_history[0][gamma]['loss']
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
                epoch_nums = np.linspace(0, epochs, len(loss_history))
                plt.semilogy(epoch_nums, loss_history, label=f"Mode {mode}")

    plt.title(r"Modes 1-5: PDE Residual Minimization", fontsize=18)
    plt.xlabel("Epochs", fontsize=18)
    plt.ylabel("PDE Residual", fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "separated_loss_types.png"), dpi=300)
    plt.close()

    # 2. Plot chemical potential convergence
    plt.figure(figsize=(10, 6))
    for mode in modes:
        for gamma in [0.0]:  # Focus on γ=0 for clarity
            if mode in training_history and gamma in training_history[mode]:
                lambda_history = training_history[mode][gamma]['lambda']
                epoch_nums = np.linspace(0, epochs, len(lambda_history))

                # For γ=0, the theoretical value should be mode + 0.5
                theoretical_value = mode + 0.5

                # Plot relative error to theoretical value
                relative_error = [abs(l - theoretical_value) / theoretical_value for l in lambda_history]
                plt.semilogy(epoch_nums, relative_error, label=f"Mode {mode}")

    plt.title(r"Chemical Potential Convergence", fontsize=18)
    plt.xlabel("Epochs", fontsize=18)
    plt.ylabel(r"Relative Error in $\mu$", fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "chemical_potential_convergence.png"), dpi=300)
    plt.close()

    # 3. Plot normalized loss (as percentage of initial loss)
    plt.figure(figsize=(10, 6))
    for mode in modes:
        for gamma in [0.0]:  # Focus on γ=0 for clarity
            if mode in training_history and gamma in training_history[mode]:
                loss_history = training_history[mode][gamma]['loss']
                initial_loss = loss_history[0]
                normalized_loss = [l / initial_loss for l in loss_history]

                epoch_nums = np.linspace(0, epochs, len(loss_history))
                plt.semilogy(epoch_nums, normalized_loss, label=f"Mode {mode}")

    plt.title(r"Normalized Loss Convergence", fontsize=18)
    plt.xlabel("Epochs", fontsize=18)
    plt.ylabel("Loss / Initial Loss", fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "normalized_loss.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    # Setup parameters
    lb, ub = -10, 10  # Domain boundaries
    N_f = 4000  # Number of collocation points
    epochs = 5001  # Increased epochs for better convergence
    layers = [1, 64, 64, 64, 1]  # Neural network architecture

    # Create uniform grid for training and testing
    X = np.linspace(lb, ub, N_f).reshape(-1, 1)
    X_test = np.linspace(lb, ub, 1000).reshape(-1, 1)  # Higher resolution for plotting

    # Gamma values from the paper
    gamma_values = [0.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0]

    # Include modes 0 through 5
    modes = [0, 1, 2, 3, 4, 5]

    # Train models
    print("Starting training for all modes and gamma values...")
    models_by_mode, mu_table, training_history = train_gpe_model(
        gamma_values, modes, X, lb, ub, layers, epochs,
        potential_type='harmonic', lr=1e-3, verbose=True
    )
    print("Training completed!")

    # Plot wavefunctions (not densities) for individual modes
    print("Generating individual mode plots...")
    plot_wavefunction(models_by_mode, X_test, gamma_values, modes)

    # Plot μ vs γ for all modes
    print("Generating chemical potential vs. gamma plot...")
    plot_mu_vs_gamma(mu_table, modes)

    # Plot combined loss history
    print("Generating combined loss plots...")
    #plot_combined_loss_history(training_history, modes, gamma_values, epochs)
    plot_improved_loss_visualization(training_history, modes, gamma_values, epochs)
    plot_all_modes_gamma_loss(training_history, modes, gamma_values, epochs)