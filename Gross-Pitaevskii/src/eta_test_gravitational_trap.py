import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.special import airy, gamma as gamma_func

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GrossPitaevskiiPINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) for solving the 1D Gross-Pitaevskii Equation
    with a gravitational trap potential.
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
        self.g = 1.0  # Gravitational acceleration parameter

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

    def gravitational_solution(self, x, n):
        """
        Compute the analytical solution for the gravitational trap for the linear case (gamma = 0).
        Based on equations (28)-(31) from the paper.

        For gravitational potential V(x) = mgx for x > 0 and infinity for x < 0.
        The solutions involve Airy functions.
        """
        # Convert to numpy for Airy function calculation
        x_np = x.cpu().detach().numpy()

        # Parameters for gravitational trap (equations 28-31 in the paper)
        xi_n = self.get_airy_zeros(n + 1)[-1]  # Get the (n+1)th zero of the Airy function
        l_0 = (self.hbar ** 2 / (2 * self.m ** 2 * self.g)) ** (1 / 3)  # Characteristic length

        # Calculate xi = (x/l_0 - E_n/(m*g*l_0))
        # Where E_n = -xi_n * m * g * l_0 is the energy of the nth state
        xi = x_np / l_0 - xi_n

        # Calculate the Airy function
        Ai_vals = np.zeros_like(xi)
        mask = x_np >= 0  # Only calculate for x >= 0 (where potential is finite)

        # Get Airy function values where x >= 0
        Ai_vals[mask] = airy(xi[mask])[0]

        # Normalization factor (from paper equation 31)
        # This is approximate and might need manual adjustment
        norm_factor = 1.0 / (l_0 * (airy(xi_n)[1]) ** 2) ** (0.5)

        # Convert back to tensor and apply normalization
        solution = norm_factor * torch.tensor(Ai_vals, dtype=torch.float32).to(device)

        # Set solution to zero for x < 0 (infinite potential barrier)
        solution[x < 0] = 0.0

        return solution

    def get_airy_zeros(self, num_zeros):
        """
        Get the first n zeros of the Airy function Ai(x).
        These are the negative zeros in ascending order by absolute value.
        """
        # Approximate values of the first several zeros of the Airy function
        # These are more precise than what scipy.special.ai_zeros provides for the small n we need
        airy_zeros = [
            -2.33811, -4.08795, -5.52056, -6.78671, -7.94413,
            -9.02265, -10.0401, -11.0085, -11.9361, -12.8288,
            -13.6915, -14.5272, -15.3394, -16.1307, -16.9039
        ]
        return np.array(airy_zeros[:num_zeros])

    def forward(self, inputs):
        """
        Forward pass through the neural network.
        """
        return self.network(inputs)

    def get_complete_solution(self, x, perturbation, mode=None):
        """
        Get the complete solution by combining the base gravitational solution with
        the neural network perturbation.
        """
        if mode is None:
            mode = self.mode
        base_solution = self.gravitational_solution(x, mode)
        return base_solution + perturbation

    def compute_potential(self, x, potential_type="gravitational", **kwargs):
        """
        Compute potential function for the 1D domain.
        """
        if potential_type == "gravitational":
            g = kwargs.get('g', self.g)  # Gravitational acceleration
            # V(x) = mgx for x >= 0, and infinity (practically a large value) for x < 0
            V = torch.zeros_like(x)
            mask_positive = (x >= 0)
            mask_negative = (x < 0)

            V[mask_positive] = self.m * g * x[mask_positive]
            V[mask_negative] = 1e6  # Very large value to approximate infinity

        elif potential_type == "harmonic":
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

    def pde_loss(self, inputs, predictions, gamma, potential_type="gravitational", precomputed_potential=None):
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

        return pde_loss, pde_residual, lambda_pde, u

    def riesz_loss(self, inputs, predictions, gamma, potential_type="gravitational", precomputed_potential=None):
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

        # Calculate normalization factor for proper integration
        dx = inputs[1] - inputs[0]  # Grid spacing
        norm_factor = torch.sum(u ** 2) * dx

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
        lambda_riesz = riesz_energy

        return riesz_energy, lambda_riesz, u

    def boundary_loss(self, boundary_points, boundary_values):
        """
        Compute the boundary loss for the boundary conditions.
        For gravitational trap, we need to enforce ψ(x) = 0 for x < 0.
        """
        u_pred = self.forward(boundary_points)
        full_u = self.get_complete_solution(boundary_points, u_pred)
        return torch.mean((full_u - boundary_values) ** 2)

    def negative_domain_loss(self, x):
        """
        Enforce zero wavefunction in the negative domain (x < 0) for gravitational trap.
        """
        # Select points where x < 0
        mask = x < 0
        if not torch.any(mask):
            return torch.tensor(0.0, device=device)

        x_neg = x[mask]
        u_pred = self.forward(x_neg)
        full_u = self.get_complete_solution(x_neg, u_pred)

        # Penalize any non-zero values in the negative domain
        return torch.mean(full_u ** 2)

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
                    potential_type='gravitational', lr=1e-3, verbose=True):
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
        Type of potential ('gravitational', 'harmonic', etc.)
    lr : float
        Learning rate
    verbose : bool
        Whether to print training progress

    Returns:
    --------
    tuple: (models_by_mode, mu_table)
        Trained models organized by mode and gamma, and chemical potential values
    """
    # Convert training data to tensors
    dx = X_train[1, 0] - X_train[0, 0]  # Assuming uniform grid
    X_tensor = torch.tensor(X_train, dtype=torch.float32, requires_grad=True).to(device)

    # Create boundary conditions
    # For gravitational trap, we need to enforce ψ(x) = 0 for x < 0
    # and ψ(x) → 0 as x → ∞
    boundary_points = torch.tensor([[lb], [0.0], [ub]], dtype=torch.float32).to(device)
    boundary_values = torch.zeros((3, 1), dtype=torch.float32).to(device)

    # Track models and chemical potentials
    models_by_mode = {}
    mu_table = {}

    # Sort gamma values
    gamma_values = sorted(gamma_values)

    for mode in modes:
        if verbose:
            print(f"\n===== Training for mode {mode} =====")

        mu_logs = []
        models_by_gamma = {}
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
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            # Create scheduler to decrease learning rate during training
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=100, min_lr=1e-5, verbose=verbose
            )

            # Pre-compute potential for efficiency
            potential = model.compute_potential(X_tensor, potential_type=potential_type)

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

                # For gravitational trap, enforce zero wavefunction for x < 0
                negative_domain_loss = model.negative_domain_loss(X_tensor)

                # Ensure proper normalization
                norm_loss = model.normalization_loss(model.get_complete_solution(X_tensor, u_pred), dx)

                # Combined constraint loss - the negative domain constraint is crucial for gravitational trap
                constraint_loss = 10.0 * boundary_loss + 50.0 * negative_domain_loss + 20.0 * norm_loss

                # Decide which loss to use based on mode
                if mode == 0:
                    # Use Riesz energy functional for mode 0 as specified in the paper (Algorithm 2)
                    riesz_energy, lambda_value, full_u = model.riesz_loss(
                        X_tensor, u_pred, gamma, potential_type, precomputed_potential=potential
                    )

                    # For mode 0, we want to minimize the energy while satisfying constraints
                    physics_loss = riesz_energy
                    loss_type = "Riesz energy"

                    # Track the constraints separately for monitoring
                    monitoring_loss = constraint_loss.item()
                else:
                    # Use PDE residual for other modes
                    pde_loss, _, lambda_value, full_u = model.pde_loss(
                        X_tensor, u_pred, gamma, potential_type, precomputed_potential=potential
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

            # Update prev_model for next gamma value
            prev_model = model

        # Store results for this mode
        mu_table[mode] = mu_logs
        models_by_mode[mode] = models_by_gamma

    return models_by_mode, mu_table


def plot_wavefunction_densities(models_by_mode, X_test, gamma_values, modes, save_dir="plots_gravitational"):
    """
    Plot wavefunction densities for different modes and gamma values.
    Specially formatted for the gravitational trap to match Figure 7 in the paper.
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

        # Different line styles and colors - match the paper's color scheme
        linestyles = ['-', '-', '-', '-', '-', '-', '-']
        colors = ['k', 'b', 'r', 'g', 'm', 'c', 'orange', 'purple']

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

                # Normalization - ensure it's properly normalized for physical interpretation
                u_np /= np.sqrt(np.sum(u_np ** 2) * dx)

                # Enforce zero for x < 0 (infinite potential barrier)
                u_np[X_test.flatten() < 0] = 0

                # Plot wavefunction density
                plt.plot(X_test.flatten(), u_np ** 2,
                         linestyle=linestyles[j % len(linestyles)],
                         color=colors[j % len(colors)],
                         label=f"γ={gamma:.1f}")

        # Configure individual figure to match the paper
        plt.title(f"Mode {mode} Wavefunction Density (Gravitational Trap)", fontsize=14)
        plt.xlabel("x", fontsize=12)
        plt.ylabel("|ψ(x)|²", fontsize=12)
        plt.grid(True)
        plt.legend()
        plt.xlim(-2, 15)  # Adjusted range to focus on the positive domain
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"grav_mode_{mode}_density.png"), dpi=300)
        plt.close()


def plot_combined_grid(models_by_mode, X_test, gamma_values, modes, save_dir="plots_gravitational"):
    """
    Create a grid of subplots showing all modes.
    Formatted to match Figure 7 in the paper.
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

    # Different line styles and colors - match paper's style
    linestyles = ['-', '-', '-', '-', '-', '-']
    colors = ['k', 'b', 'r', 'g', 'm', 'c', 'orange', 'purple']

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

                # Enforce zero for x < 0 (infinite potential barrier)
                u_np[X_test.flatten() < 0] = 0

                # Plot on the appropriate subplot
                ax.plot(X_test.flatten(), u_np ** 2,
                        linestyle=linestyles[j % len(linestyles)],
                        color=colors[j % len(colors)],
                        label=f"γ={gamma:.1f}")

        # Configure the subplot to match the paper's style
        ax.set_title(f"Mode {mode}", fontsize=12)
        ax.set_xlabel("x", fontsize=10)
        ax.set_ylabel("|ψ(x)|²", fontsize=10)
        ax.grid(True)
        ax.legend(fontsize=8)
        ax.set_xlim(-2, 15)  # Focus on positive domain

    # Hide any unused subplots
    for i in range(len(modes), len(axes)):
        axes[i].axis('off')

    # Finalize and save combined figure
    fig.suptitle("Wavefunction Densities for Gravitational Trap", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(save_dir, "grav_all_modes_combined.png"), dpi=300)
    plt.close(fig)


def plot_mu_vs_gamma(mu_table, modes, save_dir="plots_gravitational"):
    """
    Plot chemical potential vs. interaction strength for different modes.
    Formatted to show the relationship for the gravitational trap.
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

    plt.xlabel("γ (Interaction Strength)", fontsize=12)
    plt.ylabel("μ (Chemical Potential)", fontsize=12)
    plt.title("Chemical Potential vs. Interaction Strength for Gravitational Trap", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "grav_mu_vs_gamma_all_modes.png"), dpi=300)
    plt.close()


def advanced_initialization(m, mode):
    """Initialize network weights with consideration of the mode number"""
    if isinstance(m, nn.Linear):
        # Use Xavier initialization but scale based on mode
        gain = 1.0 / (1.0 + 0.1 * mode)  # Decrease gain for higher modes
        nn.init.xavier_uniform_(m.weight, gain=gain)

        # Initialize biases with small values
        m.bias.data.fill_(0.01)


if __name__ == "__main__":
    # Setup parameters
    lb, ub = -5, 20  # Domain boundaries adjusted for gravitational trap
    N_f = 5000  # Number of collocation points (increased for better resolution)
    epochs = 2001  # Increased epochs for better convergence
    layers = [1, 64, 64, 64, 1]  # Neural network architecture
    save_dir = "plots_gravitational"  # Define the output directory for plots

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Create uniform grid for training and testing
    X = np.linspace(lb, ub, N_f).reshape(-1, 1)
    X_test = np.linspace(lb, ub, 1000).reshape(-1, 1)  # Higher resolution for plotting

    # Gamma values from the paper
    gamma_values = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]

    # Include modes 0 through 7
    modes = [0, 1, 2, 3, 4, 5, 6, 7]

    # Train models
    print("Starting training for all modes and gamma values...")
    models_by_mode, mu_table = train_gpe_model(
        gamma_values, modes, X, lb, ub, layers, epochs,
        potential_type='gravitational', lr=1e-3, verbose=True
    )
    print("Training completed!")

    # Plot wavefunction densities for individual modes
    print("Generating individual mode plots...")
    plot_wavefunction_densities(models_by_mode, X_test, gamma_values, modes)

    # Plot μ vs γ for all modes
    print("Generating chemical potential vs. gamma plot...")
    plot_mu_vs_gamma(mu_table, modes)

    # Also create a combined grid figure to show all modes
    plot_combined_grid(models_by_mode, X_test, gamma_values, modes, save_dir)