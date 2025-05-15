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

    def __init__(self, layers, hbar=1.0, m=1.0, mode=0, potential_type = "harmonic", gamma=1.0, L=1.0, use_residual=True):
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
        potential_type: str, optional
            Type of potential (default is harmonic)
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
        self.potential_type = potential_type # Type of potential
        self.gamma = gamma  # Interaction strength parameter
        self.L = L  # Length of the box

        # Now build the network after all attributes are set
        self.network = self.build_network()

    def build_network(self):
        """
        Build the neural network with tanh activation functions and residual connections.
        """
        if not self.use_residual:
            # Original architecture without residual blocks
            layers = []
            for i in range(len(self.layers) - 1):
                layers.append(nn.Linear(self.layers[i], self.layers[i + 1]))
                if i < len(self.layers) - 2:
                    layers.append(nn.Tanh())
            return nn.Sequential(*layers)
        else:
            # New architecture with residual blocks
            modules = []

            # Input layer
            input_dim = self.layers[0]
            hidden_dim = self.layers[1]
            modules.append(nn.Linear(input_dim, hidden_dim))
            modules.append(nn.Tanh())

            # Residual blocks in the middle layers
            num_res_blocks = len(self.layers) - 3  # Subtract input, first hidden, and output
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

    def energy_eigenvalue(self, n):
        """
        Compute the energy eigenvalue for mode n in a box potential.

        E_n = (n²π²ħ²)/(2mL²)

        With ħ=1 and m=1, this simplifies to:
        E_n = (n²π²)/(2L²)
        """
        return (n ** 2 * np.pi ** 2) / (2 * self.L ** 2)

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

    def get_complete_solution(self, x, perturbation, mode=None, potential_type=None):
        """
        Get the complete solution by combining the base Hermite solution with the neural network perturbation.
        """
        if mode is None:
            mode = self.mode

        if potential_type is None:
            potential_type = self.potential_type

        if potential_type == "harmonic":
            base_solution = self.weighted_hermite(x, mode)
        else:
            base_solution = self.box_eigenfunction(x, mode)
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
        elif potential_type == "box":
            # Infinite square well / box potential is zero inside the box
            V = torch.zeros_like(x)
        else:
            raise ValueError(f"Unknown potential type: {potential_type}")
        return V

    def pde_loss(self, inputs, predictions, gamma, p, potential_type="harmonic", precomputed_potential=None):
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
        #interaction = gamma * u ** 3
        #interaction = gamma * torch.abs(u) ** (p - 1) * u
        interaction = gamma * u ** p

        numerator = torch.mean(u * (kinetic + potential + interaction))
        denominator = torch.mean(u ** 2)
        lambda_pde = numerator / denominator

        # Residual of the 1D Gross-Pitaevskii equation
        pde_residual = kinetic + potential + interaction - lambda_pde * u

        # PDE loss (mean squared residual)
        pde_loss = torch.mean(pde_residual ** 2)

        if potential_type == "harmonic":
            lambda_pde = lambda_pde + self.mode

        return pde_loss, pde_residual, lambda_pde, u

    def riesz_loss(self, inputs, predictions, gamma, p, potential_type="harmonic", precomputed_potential=None):
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
        #interaction_term = 0.5 * gamma * torch.sum(u ** 4) * dx / norm_factor
        interaction_term = (2.0 * gamma / (p + 1)) * torch.sum(u ** (p + 1)) * dx / norm_factor

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

    def symmetry_loss(self, collocation_points, potential_type=None):
        """
        Compute the symmetry loss to enforce the correct symmetry based on domain and mode.

        For box potential [0, L]:
        - Even modes: ψ(x) = ψ(L-x)  (symmetric around L/2)
        - Odd modes: ψ(x) = -ψ(L-x)  (antisymmetric around L/2)

        For harmonic potential [-∞, ∞]:
        - Even modes: ψ(x) = ψ(-x)   (symmetric around 0)
        - Odd modes: ψ(x) = -ψ(-x)   (antisymmetric around 0)
        """

        if potential_type is None:
            potential_type = self.potential_type

        if potential_type == "box":
            # For box potential, reflection is around L/2
            L = self.L
            midpoint = L / 2
            x_reflected = L - collocation_points  # Reflect around L/2
        else:
            # For harmonic potential, reflection is around 0
            x_reflected = -collocation_points  # Reflect around 0

        # Evaluate original and reflected points for the FULL solution
        u_pred_original = self.forward(collocation_points)
        u_full_original = self.get_complete_solution(collocation_points, u_pred_original, potential_type=potential_type)

        u_pred_reflected = self.forward(x_reflected)
        u_full_reflected = self.get_complete_solution(x_reflected, u_pred_reflected, potential_type=potential_type)

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


def train_gpe_model(gamma_values, modes, p, X_train, lb, ub, layers, epochs,
                    potential_type='harmonic', lr=1e-3, verbose=True):
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
    L = ub
    boundary_points = torch.tensor([[lb], [ub]], dtype=torch.float32).to(device)
    boundary_values = torch.zeros((2, 1), dtype=torch.float32).to(device)

    # Track models, chemical potentials, and training history
    models_by_mode = {}
    mu_table = {}
    training_history = {}

    # Sort gamma values
    gamma_values = sorted(gamma_values)

    # Precompute potential for the entire grid
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
                print(f"\nTraining for γ = {gamma:.2f}, mode = {mode}, nonlinearity p = {p}, potential = {potential_type}")

            # Initialize model for this mode and gamma
            model = GrossPitaevskiiPINN(layers, mode=mode, gamma=gamma, L=L, potential_type=potential_type).to(device)

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
                sym_loss = model.symmetry_loss(X_tensor)

                # Combined constraint loss
                constraint_loss = 10.0 * boundary_loss + 20.0 * norm_loss + 5.0 * sym_loss

                # Decide which loss to use based on mode
                if mode == 0:
                    # Use Riesz energy functional for mode 0 as specified in the paper (Algorithm 2)
                    riesz_energy, lambda_value, full_u = model.riesz_loss(
                        X_tensor, u_pred, gamma, p, potential_type, precomputed_potential
                    )

                    pde_loss, _, _, _ = model.pde_loss(
                        X_tensor, u_pred, gamma, p, potential_type, precomputed_potential
                    )

                    # For mode 0, we want to minimize the energy while satisfying constraints
                    # Note: We don't expect the energy to go to zero, it should converge to the ground state energy
                    #physics_loss = pde_loss + riesz_energy
                    physics_loss = pde_loss
                    loss_type = "Riesz energy"

                    # Track the constraints separately for monitoring
                    monitoring_loss = constraint_loss.item()
                else:
                    # Use PDE residual for other modes
                    pde_loss, _, lambda_value, full_u = model.pde_loss(
                        X_tensor, u_pred, gamma, p, potential_type, precomputed_potential
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


def plot_wavefunction(models_by_mode, X_test, gamma_values, modes, p, lb, ub, potential_type = "box", save_dir="box_plots"):
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
        if potential_type == "box" and mode == 0:
            plt.ylim(-0.2, 1.6)
        elif potential_type == "box":
            plt.ylim(-1.6, 1.6)  # Set y limits to match Figure 5 in paper
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"mode_{mode}_wavefunction_p{p}_{potential_type}.png"), dpi=300)
        plt.close()

    # Also create a combined grid figure to show all modes
    plot_combined_grid(models_by_mode, X_test, gamma_values, modes, p, lb, ub, potential_type, save_dir)


def plot_combined_grid(models_by_mode, X_test, gamma_values, modes, p, lb, ub, potential_type="box", save_dir="box_plots"):
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
        if potential_type == "box" and mode == 0:
            plt.ylim(-0.2, 1.6)
        elif potential_type == "box":
            plt.ylim(-1.6, 1.6)  # Set y limits to match Figure 5 in paper

    # Hide any unused subplots
    for i in range(len(modes), len(axes)):
        axes[i].axis('off')

    # Finalize and save combined figure
    fig.suptitle(f"Wavefunctions for All Modes (p={p})", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.legend(fontsize=8)
    fig.savefig(os.path.join(save_dir, f"all_modes_combined_wavefunctions_p{p}_{potential_type}.png"), dpi=300)
    plt.close(fig)


def plot_mu_vs_gamma(mu_table, modes, p, potential_type="box", save_dir="plots"):
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

    plt.title("Ultra-Smooth Training Progress - All Modes", fontsize=20)
    plt.xlabel("Epochs", fontsize=18)
    plt.ylabel("Loss (Log Scale)", fontsize=18)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"ultra_smooth_training_progress_p{p}_{potential_type}.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    # Setup parameters
    N_f = 4000  # Number of collocation points
    epochs = 1001  # Increased epochs for better convergence
    layers = [1, 64, 64, 64, 1]  # Neural network architecture

    # Gamma values from the paper
    gamma_values = [0.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0]

    # Include modes 0 through 5
    modes = [0, 1, 2, 3, 4, 5]

    # Nonlinearity powers
    nonlinearity_powers = [2]

    for p in nonlinearity_powers:

        #all_potentials = ['box', 'harmonic']
        all_potentials = ['box']

        for potential_type in all_potentials:

            # Adjust lower and upper bound based on potential type
            if potential_type == 'box':
                # The length of the box is self.L, which is 1
                lb, ub = 0, 1
            else:
                lb, ub = -10, 10

            # Create uniform grid for training and testing
            X = np.linspace(lb, ub, N_f).reshape(-1, 1)
            X_test = np.linspace(lb, ub, 1000).reshape(-1, 1)

            # Create a specific directory for this p value and potential type
            p_save_dir = f"plots_p{p}_{potential_type}"
            os.makedirs(p_save_dir, exist_ok=True)

            # Train models
            print(f"\nStarting training for {potential_type} potential for all modes and gamma values with p={p}...")
            models_by_mode, mu_table, training_history = train_gpe_model(
                gamma_values, modes, p, X, lb, ub, layers, epochs,
                potential_type, lr=1e-3, verbose=True
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
            plot_improved_loss_visualization(training_history, modes, gamma_values, epochs, p, potential_type, p_save_dir)
            plot_all_modes_gamma_loss(training_history, modes, gamma_values, epochs, p, potential_type, p_save_dir)

            print(f"Completed all calculations for {potential_type} potential\n")