import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import math
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.special import hermite

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

    def __init__(self, layers, hbar=1.0, m=1.0, mode=0, gamma=1.0, L=1.0):
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
        self.network = self.build_network()
        self.hbar = hbar  # Planck's constant, fixed
        self.m = m  # Particle mass, fixed
        self.mode = mode  # Mode number (n)
        self.gamma = gamma  # Interaction strength parameter
        self.L = L  # Length of the box

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

    def box_eigenfunction(self, x, n):
        """
        Compute the analytic eigenfunction for a particle in a box.

        For the linear case (gamma = 0), the solution is:
        phi_n(x) = sqrt(2/L) * sin((n+1)*pi*x/L)

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

        E_n = ((n+1)²π²ħ²)/(2mL²)  # Adjusting for 0-indexed modes

        With ħ=1 and m=1, this simplifies to:
        E_n = ((n+1)²π²)/(2L²)
        """
        n_actual = n + 1  # Convert mode to quantum number
        return (n_actual ** 2 * np.pi ** 2) / (2 * self.L ** 2)

    def forward(self, inputs):
        """
        Forward pass through the neural network.
        """
        return self.network(inputs)

    def get_complete_solution(self, x, perturbation, mode=None, potential_type="box"):
        """
        Get the complete solution by combining the base sine solution with the neural network perturbation.
        """
        if mode is None:
            mode = self.mode

        if potential_type == "box":
            base_solution = self.box_eigenfunction(x, mode)
        elif potential_type == "harmonic":
            base_solution = self.weighted_hermite(x, mode)
        else:
            raise ValueError(f"Unknown potential type: {potential_type}")
        return base_solution + perturbation

    def compute_potential(self, x, potential_type="box"):
        """
        Compute potential function for the 1D domain.

        For the box potential:
        V(x) = 0 for 0 < x < L
        V(x) = ∞ for x <= 0 or x >= L (enforced via boundary conditions)
        """
        if potential_type == "box":
            # Infinite square well / box potential is zero inside the box
            V = torch.zeros_like(x)
        elif potential_type == "harmonic":
            omega = 1.0  # Frequency for harmonic potential
            V = 0.5 * omega ** 2 * x ** 2
        else:
            raise ValueError(f"Unknown potential type: {potential_type}")
        return V

    def pde_loss(self, inputs, predictions, gamma, p, potential_type="box", precomputed_potential=None):
        """
        Compute the PDE loss for the Gross-Pitaevskii equation.
        μψ = -1/2 ∇²ψ + Vψ + γ|ψ|²ψ
        """
        # Get the complete solution (base + perturbation)
        u = self.get_complete_solution(inputs, predictions, potential_type=potential_type)

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
        # interaction = gamma * u ** 3
        interaction = gamma * torch.abs(u) ** (p - 1) * u

        numerator = torch.mean(u * (kinetic + potential + interaction))
        denominator = torch.mean(u ** 2)
        lambda_pde = numerator / denominator

        # Residual of the 1D Gross-Pitaevskii equation
        pde_residual = kinetic + potential + interaction - lambda_pde * u

        # PDE loss (mean squared residual)
        pde_loss = torch.mean(pde_residual ** 2)

        return pde_loss, pde_residual, lambda_pde, u

    def riesz_loss(self, inputs, predictions, gamma, p, potential_type="box", precomputed_potential=None):
        """
        Compute the Riesz energy loss for the Gross-Pitaevskii equation.
        E[ψ] = ∫[|∇ψ|²/2 + V|ψ|² + γ|ψ|⁴/2]dx

        This corresponds to Algorithm 2 in the paper at https://arxiv.org/pdf/1208.2123
        """
        # Get the complete solution (base + perturbation)
        u = self.get_complete_solution(inputs, predictions, potential_type=potential_type)

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
        lambda_riesz = riesz_energy

        return riesz_energy, lambda_riesz, u

    def boundary_loss(self, boundary_points, boundary_values, potential_type="box"):
        """
        Compute the boundary loss for the boundary conditions.
        """
        u_pred = self.forward(boundary_points)
        full_u = self.get_complete_solution(boundary_points, u_pred, potential_type=potential_type)
        return torch.mean((full_u - boundary_values) ** 2)

    def symmetry_loss(self, collocation_points, potential_type="box"):
        """
        Compute the symmetry loss to enforce u(x) = u(-x) for even modes
        and u(x) = -u(-x) for odd modes.
        """
        # For symmetric potential around x=0, we reflect around 0
        x_reflected = -collocation_points

        # Evaluate u(x) and u(-x) for the FULL solution
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

    def mode2_shape_loss(self, x, u):
        """
        Simple shape regularization for mode 2.
        Encourages the expected "two-node" pattern of mode 2.
        """
        if self.mode != 2:
            return torch.tensor(0.0, device=x.device)

        # Mode 2 should have 2 internal nodes (at approximately x = L/3 and x = 2L/3)
        L = self.L
        third_L = L / 3
        two_thirds_L = 2 * L / 3

        # Find points closest to L/3 and 2L/3
        idx_L3 = torch.argmin(torch.abs(x - third_L))
        idx_2L3 = torch.argmin(torch.abs(x - two_thirds_L))

        # The wavefunction should be close to zero at these points
        node_penalty = u[idx_L3] ** 2 + u[idx_2L3] ** 2

        return node_penalty

    def normalization_loss(self, u, dx):
        """
        Compute normalization loss using proper numerical integration.
        """
        integral = torch.sum(u ** 2) * dx
        return (integral - 1.0) ** 2


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


def mode2_initialization(m):
    """Special initialization for mode 2"""
    if isinstance(m, nn.Linear):
        # Use orthogonal initialization for better feature extraction
        nn.init.orthogonal_(m.weight, gain=1.2)
        # Small bias initialization
        m.bias.data.fill_(0.005)


def train_gpe_model(gamma_values, modes, p, X_train, lb, ub, layers, epochs,
                    potential_type='box', lr=1e-3, verbose=True,
                    early_stopping_patience=500, min_epochs=2000):
    """
    Train the GPE model for different modes and gamma values with early stopping.

    Parameters:
    -----------
    gamma_values : list of float
        List of interaction strengths to train models for
    modes : list of int
        List of modes to train (0, 1, 2, 3, etc.)
    p : int
        Parameter for nonlinearity power
    lb, ub : float
        Lower and upper boundaries of the domain
    X_train : numpy.ndarray
        Training points array
    layers : list of int
        Network architecture
    epochs : int
        Number of training epochs
    potential_type : str
        Type of potential ('box', 'harmonic', etc.)
    lr : float
        Learning rate
    verbose : bool
        Whether to print training progress
    early_stopping_patience : int
        Number of epochs to wait after no improvement in loss before stopping
    min_epochs : int
        Minimum number of epochs to train regardless of early stopping

    Returns:
    --------
    tuple: (models_by_mode, mu_table)
        Trained models organized by mode and gamma, and chemical potential values
    """
    # Convert training data to tensors
    dx = X_train[1, 0] - X_train[0, 0]  # Assuming uniform grid
    X_tensor = torch.tensor(X_train, dtype=torch.float32, requires_grad=True).to(device)

    # Create boundary conditions
    boundary_points = torch.tensor([[lb], [ub]], dtype=torch.float32).to(device)
    boundary_values = torch.zeros((2, 1), dtype=torch.float32).to(device)

    # Track models and chemical potentials
    models_by_mode = {}
    mu_table = {}

    # Sort gamma values
    gamma_values = sorted(gamma_values)

    for mode in modes:
        if verbose:
            print(f"\n===== Training for mode {mode} =====")

        # Specialized training for mode 2 and box potential
        if potential_type == "box" and mode == 2:
            # Use a larger network
            layers = [1, 128, 128, 128, 1]
            lr = 5e-4  # Smaller learning rate for mode 2
            epochs = min(epochs * 2, 5000)

        mu_logs = []
        models_by_gamma = {}
        prev_model = None

        for gamma in gamma_values:
            if verbose:
                print(
                    f"\nTraining for γ = {gamma:.2f}, mode = {mode}, nonlinearity p = {p}, potential = {potential_type}")

            # Initialize model for this mode and gamma
            model = GrossPitaevskiiPINN(layers, mode=mode, gamma=gamma).to(device)

            # If this isn't the first gamma value, initialize with previous model's weights
            if prev_model is not None:
                model.load_state_dict(prev_model.state_dict())
            else:
                # Use specialized initialization for mode 2
                if mode == 2:
                    model.apply(mode2_initialization)
                else:
                    # Use the advanced initialization that considers mode number
                    model.apply(lambda m: advanced_initialization(m, mode))

            # Adam optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            # Create scheduler to decrease learning rate during training
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=100, min_lr=1e-5, verbose=verbose
            )

            # Track learning history
            lambda_history = []
            loss_history = []

            # Early stopping variables
            best_loss = float('inf')
            best_model_state = None
            best_lambda = None
            patience_counter = 0

            for epoch in range(epochs):
                optimizer.zero_grad()

                # Forward pass
                u_pred = model.forward(X_tensor)

                # Calculate losses
                pde_loss, _, lambda_pde, full_u = model.pde_loss(X_tensor, u_pred, gamma, p, potential_type)
                boundary_loss = model.boundary_loss(boundary_points, boundary_values, potential_type)
                norm_loss = model.normalization_loss(full_u, dx)
                sym_loss = model.symmetry_loss(X_tensor, potential_type)

                # Additional shape loss for mode 2
                shape_loss = 0.0
                if mode == 2:
                    shape_loss = model.mode2_shape_loss(X_tensor, full_u)

                # For γ = 0.0 and the box potential, we know the exact energy should be π²/(2L²)
                if potential_type == 'box' and gamma == 0.0:
                    # Get the analytical energy value
                    exact_energy = model.energy_eigenvalue(mode)

                    # Add loss term to push towards the correct energy
                    energy_error = (lambda_pde - exact_energy) ** 2
                    # Increase energy matching weight for mode 2
                    energy_weight = 10.0 if mode == 2 else 5.0
                    pde_loss = pde_loss + energy_weight * energy_error

                # Total loss - balance different components
                if mode == 2:
                    # Adjust weights for mode 2
                    total_loss = 2.0 * pde_loss + 15.0 * boundary_loss + 25.0 * norm_loss + 2.0 * sym_loss + 5.0 * shape_loss
                else:
                    # Standard weights for other modes
                    total_loss = pde_loss + 10.0 * boundary_loss + 20.0 * norm_loss + 5.0 * sym_loss

                # Backpropagate
                total_loss.backward()
                # Stronger gradient clipping for mode 2
                clip_norm = 0.5 if mode == 2 else 1.0
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
                scheduler.step(total_loss)

                # Record history
                if epoch % 100 == 0:
                    lambda_history.append(lambda_pde.item())
                    loss_history.append(total_loss.item())

                    if verbose and epoch % 500 == 0:
                        print(f"Epoch {epoch}, Loss: {total_loss.item():.6f}, μ: {lambda_pde.item():.4f}")

                # Early stopping check (only after min_epochs)
                if epoch >= min_epochs:
                    current_loss = total_loss.item()

                    # If we found a better model
                    if current_loss < best_loss:
                        best_loss = current_loss
                        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                        best_lambda = lambda_pde.item()  # Store the lambda value from the best model
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    # If we've waited long enough with no improvement
                    if patience_counter >= early_stopping_patience:
                        if verbose:
                            print(f"Early stopping triggered at epoch {epoch}, best loss: {best_loss:.6f}")
                        break

            # Load the best model state if we have one
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                final_mu = best_lambda  # Use the lambda from the best model
            else:
                # If early stopping wasn't triggered, use the last recorded lambda
                final_mu = lambda_history[-1] if lambda_history else 0.0

            if verbose:
                print(f"Final μ for mode {mode}, γ = {gamma:.2f}: {final_mu:.4f}")

            mu_logs.append((gamma, final_mu))
            models_by_gamma[gamma] = model

            # Update prev_model for next gamma value
            prev_model = model

        # Store results for this mode
        mu_table[mode] = mu_logs
        models_by_mode[mode] = models_by_gamma

    return models_by_mode, mu_table


def plot_wavefunction(models_by_mode, X_test, gamma_values, modes, p, lb, ub, potential_type="box", save_dir="plots"):
    """
    Plot wavefunction densities for different modes and gamma values.
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
        colors = ['k', 'b', 'r', 'g', 'm', 'c', 'y', 'orange']

        # Plot solutions for different gamma values
        for j, gamma in enumerate(gamma_values):
            if gamma not in models_by_mode[mode]:
                continue

            model = models_by_mode[mode][gamma]
            model.eval()

            with torch.no_grad():
                u_pred = model.forward(X_tensor)
                full_u = model.get_complete_solution(X_tensor, u_pred, mode, potential_type)
                u_np = full_u.cpu().numpy().flatten()

                # Normalization
                u_np /= np.sqrt(np.sum(u_np ** 2) * dx)

                # Plot wavefunction
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
        if potential_type=="box" and mode == 0:
            plt.ylim(-0.2, 1.6)
        elif potential_type=="box":
            plt.ylim(-1.6, 1.6)  # Set y limits to match Figure 5 in paper
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"mode_{mode}_wavefunction_p{p}_{potential_type}.png"), dpi=300)
        plt.close()

    # Also create a combined grid figure to show all modes
    plot_combined_grid(models_by_mode, X_test, gamma_values, modes, p, lb, ub, potential_type, save_dir)


def plot_combined_grid(models_by_mode, X_test, gamma_values, modes, p, lb, ub, potential_type="box", save_dir="plots"):
    """
    Create a grid of subplots showing all modes.
    """
    # Determine grid dimensions
    n_modes = len(modes)
    n_cols = min(3, n_modes)  # Max 3 columns (to match paper's format)
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
    colors = ['k', 'b', 'r', 'g', 'm', 'c', 'y', 'orange']

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
                full_u = model.get_complete_solution(X_tensor, u_pred, mode, potential_type)
                u_np = full_u.cpu().numpy().flatten()

                # Proper normalization
                u_np /= np.sqrt(np.sum(u_np ** 2) * dx)

                # Plot the wavefunction
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

    Parameters:
    -----------
    p : int
        Nonlinearity power used in the models
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


if __name__ == "__main__":
    # Setup parameters
    N_f = 4000  # Number of collocation points
    epochs = 4001  # Increased epochs for better convergence
    layers = [1, 64, 64, 64, 1]  # Neural network architecture

    # Gamma values from the paper
    gamma_values = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]

    # Include modes 0 through 5 to match Figure 5
    modes = [0, 1, 2, 3, 4, 5]

    # Nonlinearity powers
    nonlinearity_powers = [3]

    for p in nonlinearity_powers:

        # all_potentials = ['box', 'harmonic']
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

            if potential_type == 'box':
                # Print expected analytical energies for reference
                print("Expected analytical energies for γ = 0:")
                for mode in modes:
                    energy = ((mode + 1) ** 2 * np.pi ** 2) / (2 * ub ** 2)  # Box is assumed to be from lower bound to upper bound 1
                    print(f"Mode {mode}: Energy = {energy:.6f}")

            # Train models
            print(f"\nStarting training for {potential_type} potential for all modes and gamma values with p={p}...")
            models_by_mode, mu_table = train_gpe_model(
                gamma_values, modes, p, X, lb, ub, layers, epochs,
                potential_type, lr=1e-3, verbose=True, early_stopping_patience=500, min_epochs=2000
            )
            print("Training completed.")

            # Plot wavefunctions for individual modes
            print("Generating individual mode plots...")
            plot_wavefunction(models_by_mode, X_test, gamma_values, modes, p, lb, ub, potential_type, p_save_dir)

            # Plot μ vs γ for all modes
            print("Generating chemical potential vs. gamma plot...")
            plot_mu_vs_gamma(mu_table, modes, p, potential_type, p_save_dir)

            print(f"Completed all calculations for {potential_type} potential")