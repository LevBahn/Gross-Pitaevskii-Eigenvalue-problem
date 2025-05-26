import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import math
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.special import hermite

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GrossPitaevskiiPINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) for solving the 1D Gross-Pitaevskii Equation.
    With support for arbitrary nonlinearity power p.
    """

    def __init__(self, layers, hbar=1.0, m=1.0, mode=0, gamma=1.0, p=2):
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
        p : int, optional
            Nonlinearity power (default is 2, corresponding to cubic nonlinearity).
        """
        super().__init__()
        self.layers = layers
        self.network = self.build_network()
        self.hbar = hbar  # Planck's constant, fixed
        self.m = m  # Particle mass, fixed
        self.mode = mode  # Mode number (n)
        self.gamma = gamma  # Interaction strength parameter
        self.p = p  # Nonlinearity power

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

    def compute_energy_terms(self, u, u_x, potential, gamma, p, dx):
        """
        Compute the individual energy terms in the Riesz energy functional.
        Returns kinetic, potential, and interaction energy terms.
        """
        # Calculate normalization factor for proper numerical integration
        norm_factor = torch.sum(u ** 2) * dx

        # Kinetic energy term: |∇ψ|²/2
        kinetic_term = 0.5 * torch.sum(u_x ** 2) * dx / norm_factor

        # Potential energy term: V|ψ|²
        potential_term = torch.sum(potential * u ** 2) * dx / norm_factor

        # Interaction energy term: γ|ψ|^(p+1)/(p+1)
        interaction_term = (gamma / (p + 1)) * torch.sum(torch.abs(u) ** (p + 1)) * dx / norm_factor

        return kinetic_term, potential_term, interaction_term

    def pde_residual(self, u, u_xx, potential, gamma, p, chemical_potential):
        """
        Compute the residual of the Gross-Pitaevskii equation.
        μψ = -1/2 ∇²ψ + Vψ + γ|ψ|^(p-1)ψ
        """
        kinetic = -0.5 * u_xx
        potential_term = potential * u
        interaction = gamma * torch.abs(u) ** (p - 1) * u

        # Return the residual
        return kinetic + potential_term + interaction - chemical_potential * u

    def riesz_loss(self, inputs, predictions, gamma=None, p=None, potential_type="harmonic",
                   precomputed_potential=None, dx=None):
        """
        Compute the Riesz energy loss for the generalized Gross-Pitaevskii equation.
        E[ψ] = ∫[|∇ψ|²/2 + V|ψ|² + γ|ψ|^(p+1)/(p+1)]dx

        This corresponds to Algorithm 2 in the paper at https://arxiv.org/pdf/1208.2123

        Parameters:
        -----------
        dx : float
            Grid spacing for proper numerical integration
        p : int, optional
            Nonlinearity power (p=2 for cubic nonlinearity)
        """
        # Use class parameters if not provided
        if gamma is None:
            gamma = self.gamma
        if p is None:
            p = self.p

        # Create a fresh copy of inputs with requires_grad=True for derivative computation
        x = inputs.clone().detach().requires_grad_(True)

        # Get the neural network output for this fresh input
        pred_fresh = self.forward(x)

        # Get the complete solution with fresh inputs
        u = self.get_complete_solution(x, pred_fresh)

        # Compute first derivative with respect to x using autograd
        u_x = torch.autograd.grad(
            outputs=u,
            inputs=x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]

        # Compute second derivative with respect to x using autograd
        u_xx = torch.autograd.grad(
            outputs=u_x,
            inputs=x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True
        )[0]

        # Compute potential
        if precomputed_potential is not None:
            V = precomputed_potential
        else:
            V = self.compute_potential(x, potential_type)

        # Calculate each energy term
        kinetic_term, potential_term, interaction_term = self.compute_energy_terms(
            u, u_x, V, gamma, p, dx
        )

        # Total Riesz energy functional (this is what we want to minimize)
        riesz_energy = kinetic_term + potential_term + interaction_term

        # Chemical potential using variational approach
        lambda_riesz = kinetic_term + potential_term + (p / (p + 1)) * interaction_term

        # Also compute the PDE residual for monitoring
        pde_residual = self.pde_residual(u, u_xx, V, gamma, p, lambda_riesz)

        # Return the original solution (without derivatives) for other functions
        full_u = self.get_complete_solution(inputs, predictions)

        return riesz_energy, lambda_riesz, torch.mean(pde_residual ** 2), full_u

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


def adaptive_initialization(m, mode):
    """
    Initialize network weights with consideration of the mode number.
    This is critical for higher-order modes.
    """
    if isinstance(m, nn.Linear):
        # Use Xavier initialization but scale based on mode
        gain = 1.0 / (1.0 + 0.1 * mode)  # Gentler scaling for higher modes
        nn.init.xavier_normal_(m.weight, gain=gain)  # Normal instead of uniform

        # Mode-specific bias initialization (important for convergence)
        if mode > 3:
            # For higher modes, initialize with smaller values to avoid instability
            m.bias.data.fill_(0.0005)
        else:
            m.bias.data.fill_(0.005)


def train_gpe_model(gamma_values, modes, p, X_train, lb, ub, layers, epochs,
                    potential_type='harmonic', lr=1e-3, verbose=True,
                    early_stop_patience=100, early_stop_after=2000, max_epochs_per_mode=10000):
    """
    Train the GPE model for different modes and gamma values.
    Improved version that follows Algorithm 2 in the referenced paper.

    Parameters:
    -----------
    gamma_values : list of float
        List of interaction strengths to train models for
    modes : list of int
        List of modes to train (0, 1, 2, 3, etc.)
    """
    # Convert training data to tensors
    dx = X_train[1, 0] - X_train[0, 0]  # Grid spacing for accurate integration
    X_tensor = torch.tensor(X_train, dtype=torch.float32, requires_grad=True).to(device)

    # Create boundary conditions
    boundary_points = torch.tensor([[lb], [ub]], dtype=torch.float32).to(device)
    boundary_values = torch.zeros((2, 1), dtype=torch.float32).to(device)

    # Track models and chemical potentials
    models_by_mode = {}
    mu_table = {}

    # Sort gamma values to enable transfer learning from lower to higher gamma
    gamma_values = sorted(gamma_values)

    for mode in modes:
        if verbose:
            print(f"\n===== Training for mode {mode} =====")

        mu_logs = []
        models_by_gamma = {}
        prev_model = None

        for gamma in gamma_values:
            if verbose:
                print(f"\nTraining for γ = {gamma:.2f}, mode = {mode}, nonlinearity p = {p}")

            # Initialize model for this mode, gamma, and p
            model = GrossPitaevskiiPINN(layers, mode=mode, gamma=gamma, p=p).to(device)

            # Apply adaptive initialization based on mode number
            model.apply(lambda m: adaptive_initialization(m, mode))

            # If this isn't the first gamma value, initialize with previous model's weights
            if prev_model is not None:
                model.load_state_dict(prev_model.state_dict())

            # Adam optimizer with appropriate learning rate
            # For higher modes, we might need a lower learning rate
            effective_lr = lr / (1 + 0.1 * max(0, mode - 2))
            optimizer = torch.optim.Adam(model.parameters(), lr=effective_lr)

            # Create scheduler with mode-appropriate parameters
            T_0 = 100 if mode < 3 else 200  # Longer cycle for higher modes
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=T_0, T_mult=2, eta_min=1e-6
            )

            # Track learning history
            lambda_history = []
            energy_history = []
            pde_loss_history = []
            constraint_history = []

            # Early stopping variables
            best_loss = float('inf')
            best_model_state = None
            patience_counter = 0

            # Precompute the potential for efficiency (since it doesn't change during training)
            potential = model.compute_potential(X_tensor, potential_type)

            for epoch in range(max_epochs_per_mode):
                optimizer.zero_grad()

                # Forward pass - we need a clean forward pass for each epoch
                u_pred = model.forward(X_tensor)

                # Calculate common constraint losses for all modes
                boundary_loss = model.boundary_loss(boundary_points, boundary_values)
                norm_loss = model.normalization_loss(model.get_complete_solution(X_tensor, u_pred), dx)
                sym_loss = model.symmetry_loss(X_tensor, lb, ub)

                # Combined constraint loss with appropriate weighting
                # Increase the weight for normalization loss as it's crucial for proper solutions
                constraint_loss = 10.0 * boundary_loss + 50.0 * norm_loss + 10.0 * sym_loss

                # Following Algorithm 2 in the paper, we use the Riesz energy functional
                # Make sure to create a fresh computational graph for each call to riesz_loss
                try:
                    riesz_energy, lambda_value, pde_loss, full_u = model.riesz_loss(
                        X_tensor, u_pred, gamma, p, potential_type, potential, dx
                    )
                except RuntimeError as e:
                    if "Trying to backward through the graph a second time" in str(e):
                        if verbose:
                            print("Caught graph reuse error. Creating fresh tensors...")

                        # Create fresh tensors and recompute
                        fresh_X = X_tensor.clone().detach().requires_grad_(True)
                        fresh_u_pred = model.forward(fresh_X)

                        # Try again with fresh tensors
                        riesz_energy, lambda_value, pde_loss, full_u = model.riesz_loss(
                            fresh_X, fresh_u_pred, gamma, p, potential_type,
                            model.compute_potential(fresh_X, potential_type), dx
                        )
                    else:
                        # If it's not the error we expected, re-raise it
                        raise

                # For mode 0, we focus more on minimizing the energy functional
                # For higher modes, we balance energy minimization with PDE satisfaction
                if mode == 0:
                    # Ground state focuses on energy minimization as per Algorithm 2
                    physics_loss = riesz_energy
                    loss_type = "Riesz energy"
                else:
                    # For excited states, we also need to ensure the PDE is satisfied
                    # This helps with mode separation
                    physics_loss = riesz_energy + 0.5 * pde_loss
                    loss_type = "Riesz+PDE"

                # Total loss for optimization
                total_loss = physics_loss + constraint_loss

                # Backpropagate with explicit retain_graph to be safe
                try:
                    total_loss.backward()
                except RuntimeError as e:
                    if "Trying to backward through the graph a second time" in str(e):
                        if verbose:
                            print("Caught backward error. Retrying with fresh computation...")
                        optimizer.zero_grad()

                        # Create fresh computations
                        fresh_X = X_tensor.clone().detach().requires_grad_(True)
                        fresh_u_pred = model.forward(fresh_X)
                        fresh_constraint_loss = 10.0 * model.boundary_loss(boundary_points, boundary_values) + \
                                                50.0 * model.normalization_loss(
                            model.get_complete_solution(fresh_X, fresh_u_pred), dx) + \
                                                10.0 * model.symmetry_loss(fresh_X, lb, ub)

                        fresh_riesz_energy, fresh_lambda, fresh_pde_loss, _ = model.riesz_loss(
                            fresh_X, fresh_u_pred, gamma, p, potential_type,
                            model.compute_potential(fresh_X, potential_type), dx
                        )

                        # Recompute loss
                        if mode == 0:
                            fresh_physics_loss = fresh_riesz_energy
                        else:
                            fresh_physics_loss = fresh_riesz_energy + 0.5 * fresh_pde_loss

                        fresh_total_loss = fresh_physics_loss + fresh_constraint_loss
                        fresh_total_loss.backward()
                    else:
                        # If it's not the error we expected, re-raise it
                        raise

                # Gradient clipping is essential for stability, especially for higher modes
                # Use more aggressive clipping for higher modes
                clip_norm = 1.0 / (1.0 + 0.05 * mode)  # Decrease allowed norm for higher modes
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

                optimizer.step()
                scheduler.step()

                # Record history
                if epoch % 100 == 0:
                    lambda_history.append(lambda_value.item())
                    energy_history.append(riesz_energy.item())
                    pde_loss_history.append(pde_loss.item())
                    constraint_history.append(constraint_loss.item())

                    if verbose and epoch % 500 == 0:
                        print(f"Epoch {epoch}, {loss_type}: {physics_loss.item():.6f}, "
                              f"Constraints: {constraint_loss.item():.6f}, μ: {lambda_value.item():.4f}")

                # Early stopping logic (only after minimum epochs)
                if epoch >= early_stop_after:
                    current_loss = total_loss.item()

                    # Check if this is the best model so far
                    if current_loss < best_loss:
                        best_loss = current_loss
                        best_model_state = model.state_dict().copy()
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    # If no improvement for patience epochs, stop training
                    if patience_counter >= early_stop_patience:
                        if verbose:
                            print(f"Early stopping triggered at epoch {epoch}")
                            print(f"Best loss: {best_loss:.6f}")

                        # Restore best model
                        model.load_state_dict(best_model_state)
                        break

            # If training completed without early stopping, use the final model state
            if epoch == epochs - 1 and best_model_state is not None:
                # If we have a better model saved, use it
                if best_loss < total_loss.item():
                    model.load_state_dict(best_model_state)
                    if verbose:
                        print(f"Training completed. Using best model with loss: {best_loss:.6f}")

            # Final evaluation to get chemical potential
            with torch.no_grad():
                u_pred = model.forward(X_tensor)
                _, final_mu, _, _ = model.riesz_loss(X_tensor, u_pred, gamma, p, potential_type, potential, dx)

            # Record final chemical potential and save model
            mu_logs.append((gamma, final_mu.item()))
            models_by_gamma[gamma] = model

            # Update prev_model for next gamma value - transfer learning
            prev_model = model

        # Store results for this mode
        mu_table[mode] = mu_logs
        models_by_mode[mode] = models_by_gamma

    return models_by_mode, mu_table


def plot_wavefunction(models_by_mode, X_test, gamma_values, modes, p, save_dir="plots"):
    """
    Plot wavefunctions for different modes and gamma values with improved style.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Generate individual figures for each mode
    for mode in modes:
        if mode not in models_by_mode:
            continue

        # Create individual figure with better style
        plt.figure(figsize=(10, 6))

        # Add a grid that matches the paper's style
        plt.grid(True, linestyle='--', alpha=0.7)

        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        dx = X_test[1, 0] - X_test[0, 0]  # Spatial step size

        # Different line styles and colors that better match paper's Figure 11
        linestyles = ['-', '--', '-.', ':', '-', '--']
        colors = ['k', 'b', 'r', 'g', 'm', 'c']

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

                # For mode 0, ensure all wavefunctions are positive (no phase change)
                if mode == 0:
                    # Take absolute value to ensure positive values
                    # This is valid for ground state (mode 0) which should be nodeless
                    u_np = np.abs(u_np)

                # For better visualization of modes with sensitive structure around x=0
                # Apply a phase correction to match the paper's plots
                if mode % 2 == 0 and mode > 0:
                    # Even modes higher than 0 should have a consistent sign at origin
                    if u_np[len(u_np) // 2] > 0:
                        u_np = -u_np  # Flip sign if needed to match paper's convention

                # Plot wavefunction (not density)
                plt.plot(X_test.flatten(), u_np,
                         linestyle=linestyles[j % len(linestyles)],
                         color=colors[j % len(colors)],
                         linewidth=2.0,
                         label=f"γ={gamma:.1f}")

        # Configure individual figure with improved styling
        plt.title(f"Mode {mode} Wavefunction (p={p})", fontsize=16)
        plt.xlabel("x", fontsize=14)
        plt.ylabel("ψ(x)", fontsize=14)
        plt.xlim(-10, 10)  # Match paper's range
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"mode_{mode}_wavefunction_p{p}.png"), dpi=300)
        plt.close()

    # Also create a combined grid figure to show all modes
    plot_combined_grid(models_by_mode, X_test, gamma_values, modes, p, save_dir)


def plot_combined_grid(models_by_mode, X_test, gamma_values, modes, p, save_dir="plots"):
    """
    Create a grid of subplots showing all modes, styled closer to the paper's Figure 11.
    """
    # Determine grid dimensions
    n_modes = len(modes)
    n_cols = 2  # Paper uses 2 columns in Figure 11
    n_rows = (n_modes + n_cols - 1) // n_cols  # Ceiling division

    # Create figure with subplots - match paper style
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))

    # Flatten axes if it's a 2D array
    if n_rows > 1 or n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]  # Make it iterable

    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    dx = X_test[1, 0] - X_test[0, 0]  # Spatial step size

    # Different line styles and colors to match paper
    linestyles = ['-', '--', '-.', ':', '-', '--']
    colors = ['k', 'b', 'r', 'g', 'm', 'c']

    # Plot each mode in its subplot
    for i, mode in enumerate(modes):
        if i >= len(axes) or mode not in models_by_mode:
            continue

        ax = axes[i]

        # Add grid to match paper style
        ax.grid(True, linestyle='--', alpha=0.7)

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

                # For mode 0, ensure all wavefunctions are positive (no phase change)
                if mode == 0:
                    # Take absolute value to ensure positive values
                    u_np = np.abs(u_np)

                # Phase consistency for even modes > 0
                if mode % 2 == 0 and mode > 0:
                    # Even modes higher than 0 should have a consistent sign at origin
                    if u_np[len(u_np) // 2] > 0:
                        u_np = -u_np  # Flip sign if needed

                # Plot the wavefunction with proper styling
                ax.plot(X_test.flatten(), u_np,
                        linestyle=linestyles[j % len(linestyles)],
                        color=colors[j % len(colors)],
                        linewidth=1.5,
                        label=f"γ={gamma:.1f}")

        # Configure the subplot to match paper style
        ax.set_title(f"Mode {mode}", fontsize=12)
        ax.set_xlabel("x", fontsize=10)
        ax.set_ylabel("ψ(x)", fontsize=10)
        ax.set_xlim(-10, 10)
        ax.legend(fontsize=8)

    # Hide any unused subplots
    for i in range(len(modes), len(axes)):
        axes[i].axis('off')

    # Finalize and save combined figure
    fig.suptitle(f"Wavefunctions for All Modes (p={p})", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(save_dir, f"all_modes_wavefunctions_p{p}.png"), dpi=300)
    plt.close(fig)


def plot_mu_vs_gamma(mu_table, modes, p, save_dir="plots"):
    """
    Plot chemical potential vs. interaction strength for different modes.
    Styled to better match Figure 13 in the paper.
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))

    # Add a grid that matches the paper style
    plt.grid(True, linestyle='--', alpha=0.7)

    # Different markers and colors for different modes
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
                 linewidth=2.0,
                 markersize=8,
                 label=f"Mode {mode}")

    plt.xlabel("γ (Interaction Strength)", fontsize=14)
    plt.ylabel("μ (Chemical Potential)", fontsize=14)
    plt.title(f"Chemical Potential vs. Interaction Strength (p={p})", fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"mu_vs_gamma_all_modes_p{p}.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    # Setup parameters
    lb, ub = -15, 15  # Extended domain boundaries for better asymptotic behavior
    N_f = 5000  # Increased number of collocation points for better resolution
    epochs = 1001  # More epochs for better convergence
    max_epochs_per_mode = 10000  # Maximum epochs for each mode/gamma combination
    layers = [1, 128, 128, 128, 128, 1]  # Deeper neural network for better expressivity

    # Create uniform grid for training and testing
    X = np.linspace(lb, ub, N_f).reshape(-1, 1)
    X_test = np.linspace(lb, ub, 1000).reshape(-1, 1)  # Higher resolution for plotting

    # Gamma values from the paper
    gamma_values = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]

    # Include modes 0 through 7
    modes = [0, 1, 2, 3, 4, 5, 6, 7]

    # Loop through every nonlinearity power
    nonlinearity_powers = [2]  # Can extend to other powers like [1, 3, 4] for additional studies

    for p in nonlinearity_powers:
        # Create a specific directory for this p value
        p_save_dir = f"plots_p{p}_improved"
        os.makedirs(p_save_dir, exist_ok=True)

        # Train models with improved implementation
        print(f"Starting training for all modes and gamma values with p={p}...")
        models_by_mode, mu_table = train_gpe_model(
            gamma_values, modes, p, X, lb, ub, layers, epochs,
            potential_type='harmonic', lr=5e-4, verbose=True,  # Lower learning rate for stability
            max_epochs_per_mode=max_epochs_per_mode
        )
        print(f"Training completed for p={p}!")

        # Plot wavefunctions with improved styling
        print(f"Generating individual mode plots for p={p}...")
        plot_wavefunction(models_by_mode, X_test, gamma_values, modes, p)

        # Generate chemical potential plots
        print(f"Generating chemical potential vs. gamma plot for p={p}...")
        plot_mu_vs_gamma(mu_table, modes, p, p_save_dir)