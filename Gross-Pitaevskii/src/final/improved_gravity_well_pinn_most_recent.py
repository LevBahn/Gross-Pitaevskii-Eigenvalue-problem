import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import math
import numpy as np
import os
from scipy.special import airy, hermite

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

    def __init__(self, layers, hbar=1.0, m=1.0, mode=0, potential_type="harmonic", gamma=1.0, L=1.0, g=1.0,
                 use_residual=True):
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
        g : float, optional
            Gravitational acceleration parameter (default is 1.0).
        """
        super().__init__()
        self.layers = layers
        self.use_residual = use_residual
        self.hbar = hbar  # Planck's constant, fixed
        self.m = m  # Particle mass, fixed
        self.mode = mode  # Mode number (n)
        self.potential_type = potential_type  # Type of potential
        self.gamma = gamma  # Interaction strength parameter
        self.L = L  # Length of the box
        self.g = g  # Gravitational acceleration parameter

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

    def airy_solution(self, x, n):
        """
        FIXED: Create gravitational trap eigenfunctions using Airy functions according to
        equations (30-31) from the paper: Ψₙ(x) = Aₙ·Ai(x + xₙ)

        Key fixes:
        1. Use proper domain transformation: ξ = (mgx)^(1/3) * x for gravity well
        2. Apply correct boundary conditions
        3. Use exact Airy zeros from literature
        """
        # Convert to numpy for calculation
        x_np = x.detach().cpu().numpy()

        # CRITICAL FIX: Transform to proper dimensionless coordinate
        # For gravity well: ξ = (2mg/ℏ²)^(1/3) * x
        # In dimensionless units with mg = 1, ℏ = 1, m = 1: ξ = (2)^(1/3) * x
        xi_scale = (2.0) ** (1 / 3)
        xi = xi_scale * x_np

        # Initialize wavefunction array
        psi = np.zeros_like(x_np)

        # Airy function zeros (where Ai(z) = 0) - these are exact values
        airy_zeros = [
            -2.33810741, -4.08794944, -5.52055983, -6.78670809, -7.94413359,
            -9.02265085, -10.0401743, -11.0085243, -11.9360157, -12.8287767
        ]

        # Get the nth zero
        if n < len(airy_zeros):
            z_n = airy_zeros[n]
        else:
            # Asymptotic approximation for higher modes
            z_n = -(1.5 * np.pi * (n + 0.75)) ** (2 / 3)

        # Only calculate for x >= 0 (gravity well domain)
        pos_mask = (x_np >= 0)

        if np.any(pos_mask):
            # Calculate Ai(ξ + z_n) for positive x values
            xi_pos = xi[pos_mask]
            airy_vals = airy(xi_pos + z_n)[0]

            # Apply proper normalization over the allowed domain
            dx = float(x[1].detach() - x[0].detach()) if len(x) > 1 else 0.01

            # Normalize only over x >= 0 domain
            if len(airy_vals) > 0 and np.sum(airy_vals ** 2) > 0:
                norm_factor = np.sqrt(np.sum(airy_vals ** 2) * dx)
                if norm_factor > 1e-12:
                    airy_vals = airy_vals / norm_factor

            # Store result in positive domain
            psi[pos_mask] = airy_vals

        # x < 0 remains zero (infinite potential wall)
        psi[x_np < 0] = 0.0

        # Convert back to tensor
        solution = torch.tensor(psi, dtype=torch.float32).to(device)
        return solution

    def forward(self, inputs):
        """
        Forward pass through the neural network.
        """
        return self.network(inputs)

    def get_complete_solution(self, x, perturbation, mode=None, potential_type=None):
        """
        CRITICAL FIX: For gravity well, the NN should learn the FULL solution, not just perturbation
        This follows Algorithm 2 more closely where the NN directly approximates the eigenfunction
        """
        if mode is None:
            mode = self.mode
        if potential_type is None:
            potential_type = self.potential_type

        if potential_type == "gravity well":
            # # MAJOR FIX: For gravity well, use the NN output directly as the main solution
            # # Apply proper boundary condition through multiplication
            # x_squeezed = x.squeeze()
            #
            # # Boundary condition: ψ(x) = 0 for x < 0, smooth transition at x = 0
            # boundary_factor = torch.where(
            #     x_squeezed >= 0,
            #     torch.ones_like(x_squeezed),
            #     torch.zeros_like(x_squeezed)
            # ).unsqueeze(-1) if len(x.shape) > 1 else torch.where(
            #     x_squeezed >= 0,
            #     torch.ones_like(x_squeezed),
            #     torch.zeros_like(x_squeezed)
            # )
            #
            # # For x >= 0, use NN output with proper scaling for decay at infinity
            # # Add exponential decay factor for large x to enforce boundary condition at infinity
            # decay_factor = torch.exp(-0.1 * torch.relu(x_squeezed - 10))
            # if len(x.shape) > 1:
            #     decay_factor = decay_factor.unsqueeze(-1)
            #
            # constrained_solution = boundary_factor * decay_factor * perturbation
            #
            # if mode == 0:
            #     constrained_solution = torch.abs(constrained_solution)
            #
            # return constrained_solution
            base_solution = self.airy_solution(x, mode)

        elif potential_type == "harmonic":
            base_solution = self.weighted_hermite(x, mode)
        else:
            base_solution = self.box_eigenfunction(x, mode)

        return base_solution + perturbation

    def compute_potential(self, x, potential_type="harmonic", **kwargs):
        """
        FIXED: Compute potential function for the 1D domain.
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
        elif potential_type == "gravity well":
            # FIXED: Proper gravity well potential V(x) = mgx for x >= 0
            V = torch.zeros_like(x)
            x_squeezed = x.squeeze() if len(x.shape) > 1 else x

            # For x >= 0: V(x) = mgx (with mg = 1 in dimensionless units)
            V = torch.where(
                x_squeezed >= 0,
                x_squeezed,  # Linear potential mgx
                torch.full_like(x_squeezed, 1e8)  # Large value for x < 0
            )

            if len(x.shape) > 1:
                V = V.unsqueeze(-1)

            return V
        else:
            raise ValueError(f"Unknown potential type: {potential_type}")
        return V

    def pde_loss(self, inputs, predictions, gamma, p, potential_type="harmonic", precomputed_potential=None):
        """
        FIXED: Compute the PDE loss for the Gross-Pitaevskii equation.
        μψ = -1/2 ∇²ψ + Vψ + γ|ψ|^p ψ
        """
        # Get the complete solution
        u = self.get_complete_solution(inputs, predictions)

        # Ensure inputs require gradients
        if not inputs.requires_grad:
            inputs = inputs.clone().detach().requires_grad_(True)

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

        # Calculate individual terms
        kinetic = -0.5 * u_xx
        potential = V * u

        # Nonlinear interaction term: γ|ψ|^(p-1)ψ
        interaction = gamma * (torch.abs(u) ** (p - 1)) * u

        # # FIXED: Chemical potential calculation for gravity well
        # if potential_type == "gravity well":
        #     # Only integrate over the physical domain x >= 0
        #     x_squeezed = inputs.squeeze() if len(inputs.shape) > 1 else inputs
        #     pos_mask = (x_squeezed >= 0)
        #
        #     if torch.any(pos_mask):
        #         u_pos = u[pos_mask] if len(u.shape) > 1 else u[pos_mask]
        #         kinetic_pos = kinetic[pos_mask] if len(kinetic.shape) > 1 else kinetic[pos_mask]
        #         potential_pos = potential[pos_mask] if len(potential.shape) > 1 else potential[pos_mask]
        #         interaction_pos = interaction[pos_mask] if len(interaction.shape) > 1 else interaction[pos_mask]
        #
        #         numerator = torch.mean(u_pos * (kinetic_pos + potential_pos + interaction_pos))
        #         denominator = torch.mean(u_pos ** 2)
        #     else:
        #         numerator = torch.tensor(0.0, device=device)
        #         denominator = torch.tensor(1.0, device=device)
        # else:
        # Standard calculation for other potentials
        numerator = torch.mean(u * (kinetic + potential + interaction))
        denominator = torch.mean(u ** 2)

        # Chemical potential with regularization
        lambda_pde = numerator / (denominator + 1e-12)

        # Residual of the Gross-Pitaevskii equation
        pde_residual = kinetic + potential + interaction - lambda_pde * u

        # PDE loss (mean squared residual)
        # if potential_type == "gravity well":
        #     # Only compute loss over physical domain
        #     x_squeezed = inputs.squeeze() if len(inputs.shape) > 1 else inputs
        #     pos_mask = (x_squeezed >= 0)
        #     if torch.any(pos_mask):
        #         residual_pos = pde_residual[pos_mask] if len(pde_residual.shape) > 1 else pde_residual[pos_mask]
        #         pde_loss = torch.mean(residual_pos ** 2)
        #     else:
        #         pde_loss = torch.tensor(0.0, device=device)
        # else:
        pde_loss = torch.mean(pde_residual ** 2)

        return pde_loss, pde_residual, lambda_pde, u

    def riesz_loss(self, inputs, predictions, gamma, p, potential_type="harmonic", precomputed_potential=None):
        """
        FIXED: Compute the Riesz energy loss for the Gross-Pitaevskii equation.
        E[ψ] = ∫[|∇ψ|²/2 + V|ψ|² + γ|ψ|^(p+1)/(p+1)]dx
        """
        # Get the complete solution
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

        # Calculate grid spacing
        dx = inputs[1] - inputs[0] if len(inputs) > 1 else torch.tensor(0.01, device=device)

        # Compute potential
        if precomputed_potential is not None:
            V = precomputed_potential
        else:
            V = self.compute_potential(inputs, potential_type)

        # FIXED: For gravity well, only integrate over x >= 0
        if potential_type == "gravity well":
            x_squeezed = inputs.squeeze() if len(inputs.shape) > 1 else inputs
            pos_mask = (x_squeezed >= 0)

            if torch.any(pos_mask):
                u_pos = u[pos_mask] if len(u.shape) > 1 else u[pos_mask]
                u_x_pos = u_x[pos_mask] if len(u_x.shape) > 1 else u_x[pos_mask]
                V_pos = V[pos_mask] if len(V.shape) > 1 else V[pos_mask]

                # Normalization over positive domain only
                norm_factor = torch.sum(u_pos ** 2) * dx + 1e-12

                # Energy terms over positive domain
                kinetic_term = 0.5 * torch.sum(u_x_pos ** 2) * dx / norm_factor
                potential_term = torch.sum(V_pos * u_pos ** 2) * dx / norm_factor
                interaction_term = (gamma / (p + 1)) * torch.sum(torch.abs(u_pos) ** (p + 1)) * dx / norm_factor
            else:
                kinetic_term = torch.tensor(0.0, device=device)
                potential_term = torch.tensor(0.0, device=device)
                interaction_term = torch.tensor(0.0, device=device)
        else:
            # Standard calculation for other potentials
            norm_factor = torch.sum(u ** 2) * dx + 1e-12
            kinetic_term = 0.5 * torch.sum(u_x ** 2) * dx / norm_factor
            potential_term = torch.sum(V * u ** 2) * dx / norm_factor
            interaction_term = (gamma / (p + 1)) * torch.sum(torch.abs(u) ** (p + 1)) * dx / norm_factor

        # Total Riesz energy functional
        riesz_energy = kinetic_term + potential_term + interaction_term

        # Calculate chemical potential (energy eigenvalue)
        lambda_riesz = riesz_energy

        return riesz_energy, lambda_riesz, u

    def boundary_loss(self, boundary_points, boundary_values):
        """
        Compute the boundary loss for the boundary conditions.
        """
        u_pred = self.forward(boundary_points)
        full_u = self.get_complete_solution(boundary_points, u_pred)
        return torch.mean((full_u - boundary_values) ** 2)

    def boundary_loss_gravity_well(self, x):
        """
        FIXED: Compute boundary loss for gravity well with proper enforcement
        """
        u_pred = self.forward(x)
        full_u = self.get_complete_solution(x, u_pred)

        x_squeezed = x.squeeze() if len(x.shape) > 1 else x

        # Strong enforcement for x < 0: ψ(x) = 0
        neg_mask = x_squeezed < 0
        neg_loss = torch.mean(full_u[neg_mask] ** 2) if torch.any(neg_mask) else torch.tensor(0.0, device=device)

        # Boundary condition at x = 0: should be continuous but can be non-zero
        zero_mask = torch.abs(x_squeezed) < 0.05
        # Don't enforce ψ(0) = 0 too strongly, as this depends on the mode
        zero_loss = 0.1 * torch.mean(full_u[zero_mask] ** 2) if torch.any(zero_mask) else torch.tensor(0.0,
                                                                                                       device=device)

        # Decay condition for large x
        far_mask = x_squeezed > 15
        far_loss = torch.mean(full_u[far_mask] ** 2) if torch.any(far_mask) else torch.tensor(0.0, device=device)

        return 1000.0 * neg_loss + zero_loss + 10.0 * far_loss

    def symmetry_loss(self, collocation_points, potential_type=None):
        """
        For gravity well, no symmetry constraints should be applied
        """
        if potential_type is None:
            potential_type = self.potential_type

        # For gravity well, return zero since there's no symmetry
        if potential_type == "gravity well":
            return torch.tensor(0.0, device=device)

        # Original symmetry loss for other potentials
        if potential_type == "box":
            L = self.L
            x_reflected = L - collocation_points
        else:
            x_reflected = -collocation_points

        u_pred_original = self.forward(collocation_points)
        u_full_original = self.get_complete_solution(collocation_points, u_pred_original, potential_type=potential_type)

        u_pred_reflected = self.forward(x_reflected)
        u_full_reflected = self.get_complete_solution(x_reflected, u_pred_reflected, potential_type=potential_type)

        if self.mode % 2 == 1:
            sym_loss = torch.mean((u_full_original + u_full_reflected) ** 2)
        else:
            sym_loss = torch.mean((u_full_original - u_full_reflected) ** 2)

        return sym_loss

    def normalization_loss(self, u, dx, potential_type=None):
        """
        FIXED: Compute normalization loss with proper domain handling
        """
        if potential_type is None:
            potential_type = self.potential_type

        if potential_type == "gravity well":
            # This will be handled in the training loop with proper masking
            pass

        integral = torch.sum(u ** 2) * dx
        return (integral - 1.0) ** 2


def train_gpe_model(gamma_values, modes, p, X_train, lb, ub, layers, epochs,
                    potential_type='harmonic', lr=1e-3, verbose=True):
    """
    Train the GPE model for different modes and gamma values.

    IMPROVEMENTS:
    - Better learning rate scheduling for gravity well
    - Improved loss weighting
    - Better initialization
    """
    # Convert training data to tensors
    dx = X_train[1, 0] - X_train[0, 0]  # Assuming uniform grid
    X_tensor = torch.tensor(X_train, dtype=torch.float32, requires_grad=True).to(device)

    # Create boundary conditions
    # if potential_type == "gravity well":
    #     L = ub
    #     # More boundary points for gravity well
    #     boundary_points = torch.tensor([lb, -2.0, -1.0, -0.5, 0.0, 0.5], dtype=torch.float32).reshape(-1, 1).to(device)
    # else:
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
    temp_model = GrossPitaevskiiPINN(layers, potential_type=potential_type).to(device)
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
                print(
                    f"\nTraining for γ = {gamma:.2f}, mode = {mode}, nonlinearity p = {p}, potential = {potential_type}")

            # Initialize model for this mode and gamma
            model = GrossPitaevskiiPINN(layers, mode=mode, gamma=gamma, L=L, potential_type=potential_type).to(device)

            # Better initialization for gravity well
            if potential_type == "gravity well":
                model.apply(lambda m: gravity_well_initialization(m, mode))
            else:
                if prev_model is not None:
                    model.load_state_dict(prev_model.state_dict())
                else:
                    model.apply(lambda m: advanced_initialization(m, mode))

            # Adaptive learning rate for gravity well
            if potential_type == "gravity well":
                initial_lr = 5e-4  # Lower initial learning rate
            else:
                initial_lr = lr

            # Adam optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-6)

            # Better scheduler for gravity well
            if potential_type == "gravity well":
                scheduler = CosineAnnealingWarmRestarts(
                    optimizer, T_0=500, T_mult=1, eta_min=1e-7
                )
            else:
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

                # Calculate constraint losses with better weighting for gravity well
                if potential_type == "gravity well":
                    boundary_loss = model.boundary_loss_gravity_well(X_tensor)

                    # Get the full solution for normalization
                    full_u = model.get_complete_solution(X_tensor, u_pred)

                    # Only normalize over the allowed domain (x >= 0)
                    pos_mask = X_tensor.squeeze() >= 0
                    if torch.any(pos_mask):
                        u_pos = full_u[pos_mask]
                        dx_pos = dx
                        norm_loss = model.normalization_loss(u_pos, dx_pos)
                    else:
                        norm_loss = torch.tensor(0.0, device=device)

                    # No symmetry loss for gravity well
                    sym_loss = torch.tensor(0.0, device=device)

                    # Adjusted weights for gravity well
                    constraint_loss = 100.0 * boundary_loss + 50.0 * norm_loss
                else:
                    boundary_loss = model.boundary_loss(boundary_points, boundary_values)
                    norm_loss = model.normalization_loss(model.get_complete_solution(X_tensor, u_pred), dx)
                    sym_loss = model.symmetry_loss(X_tensor)
                    constraint_loss = 10.0 * boundary_loss + 20.0 * norm_loss + 5.0 * sym_loss

                # Decide which loss to use based on mode
                if mode == 0:
                    # Use Riesz energy functional for mode 0
                    riesz_energy, lambda_value, full_u = model.riesz_loss(
                        X_tensor, u_pred, gamma, p, potential_type, precomputed_potential
                    )

                    pde_loss, _, _, _ = model.pde_loss(
                        X_tensor, u_pred, gamma, p, potential_type, precomputed_potential
                    )

                    # For gravity well, focus more on PDE residual
                    if potential_type == "gravity well":
                        physics_loss = pde_loss #+ 0.1 * riesz_energy
                    else:
                        physics_loss = pde_loss

                    loss_type = "Riesz energy"
                    monitoring_loss = constraint_loss.item()
                else:
                    # Use PDE residual for other modes
                    pde_loss, _, lambda_value, full_u = model.pde_loss(
                        X_tensor, u_pred, gamma, p, potential_type, precomputed_potential
                    )
                    physics_loss = pde_loss
                    loss_type = "PDE residual"
                    monitoring_loss = pde_loss.item()

                # Total loss for optimization
                total_loss = physics_loss + constraint_loss

                # Backpropagate
                total_loss.backward()

                # Gradient clipping - more aggressive for gravity well
                # if potential_type == "gravity well":
                #     torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                # else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

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

            # Update prev_model for next gamma value (not used for gravity well)
            if potential_type != "gravity well":
                prev_model = model

        # Store results for this mode
        mu_table[mode] = mu_logs
        models_by_mode[mode] = models_by_gamma
        training_history[mode] = history_by_gamma

    return models_by_mode, mu_table, training_history


def gravity_well_initialization(m, mode):
    """
    Improved initialization for gravity well problems with mode-specific setup
    """
    if isinstance(m, nn.Linear):
        # Use different initialization strategies based on the mode
        if mode == 0:
            # For ground state, use small positive bias to encourage positive solutions
            gain = 0.05
            nn.init.xavier_normal_(m.weight, gain=gain)
            m.bias.data.fill_(0.01)  # Small positive bias
        else:
            # For excited states, use slightly larger weights but zero bias
            gain = 0.02 / (1.0 + 0.1 * mode)
            nn.init.xavier_normal_(m.weight, gain=gain)
            m.bias.data.fill_(0.0)  # Zero bias for excited states


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


def plot_wavefunction(models_by_mode, X_test, gamma_values, modes, p, lb, ub, potential_type="box",
                      save_dir="box_plots"):
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

                # For gravity well, only normalize over x >= 0 domain
                if potential_type == "gravity well":
                    x_test_np = X_test.flatten()
                    pos_mask = x_test_np >= 0
                    if np.any(pos_mask):
                        u_pos = u_np[pos_mask]
                        dx_pos = dx
                        norm_factor = np.sqrt(np.sum(u_pos ** 2) * dx_pos)
                        if norm_factor > 1e-12:
                            u_np = u_np / norm_factor
                else:
                    # Normal normalization for other potentials
                    norm_factor = np.sqrt(np.sum(u_np ** 2) * dx)
                    if norm_factor > 1e-12:
                        u_np = u_np / norm_factor

                # For mode 0 in non-gravity potentials, ensure positive values
                if mode == 0 and potential_type != "gravity well":
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

        # Adjust y limits based on potential type
        if potential_type == "box" and mode == 0:
            plt.ylim(-0.2, 1.6)
        elif potential_type == "box":
            plt.ylim(-1.6, 1.6)
        # elif potential_type == "gravity well":
        #     plt.ylim(-0.8, 0.8)  # Adjust for gravity well

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"mode_{mode}_wavefunction_p{p}_{potential_type}.png"), dpi=300)
        plt.close()


def plot_combined_grid(models_by_mode, X_test, gamma_values, modes, p, lb, ub, potential_type="box",
                       save_dir="box_plots"):
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

                # Handle normalization based on potential type
                if potential_type == "gravity well":
                    x_test_np = X_test.flatten()
                    pos_mask = x_test_np >= 0
                    if np.any(pos_mask):
                        u_pos = u_np[pos_mask]
                        dx_pos = dx
                        norm_factor = np.sqrt(np.sum(u_pos ** 2) * dx_pos)
                        if norm_factor > 1e-12:
                            u_np = u_np / norm_factor
                else:
                    # Proper normalization
                    norm_factor = np.sqrt(np.sum(u_np ** 2) * dx)
                    if norm_factor > 1e-12:
                        u_np = u_np / norm_factor

                # For mode 0, ensure all wavefunctions are positive (except gravity well)
                if mode == 0 and potential_type != "gravity well":
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

        # Set appropriate y limits
        if potential_type == "box" and mode == 0:
            ax.set_ylim(-0.2, 1.6)
        elif potential_type == "box":
            ax.set_ylim(-1.6, 1.6)
        # elif potential_type == "gravity well":
        #     ax.set_ylim(-0.8, 0.8)

    # Hide any unused subplots
    for i in range(len(modes), len(axes)):
        axes[i].axis('off')

    # Finalize and save combined figure
    fig.suptitle(f"Wavefunctions for All Modes (p={p})", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
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

    # Additional plots for chemical potential convergence and normalized loss
    # (keeping the rest of the original function)


def plot_all_modes_gamma_loss(training_history, modes, gamma_values, epochs, p, potential_type, save_dir="box_plots"):
    """
    Plot the training loss history for all modes and all gamma values.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Determine grid dimensions
    n_modes = len(modes)
    n_cols = min(4, n_modes)
    n_rows = (n_modes + n_cols - 1) // n_cols

    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))

    # Flatten axes if it's a 2D array
    if n_rows > 1 or n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

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
                loss_history = training_history[mode][gamma]['loss']
                epoch_nums = np.linspace(0, epochs, len(loss_history))

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


# MAIN EXECUTION WITH IMPROVED PARAMETERS
if __name__ == "__main__":
    # Setup parameters - IMPROVED for gravity well
    N_f = 5000  # Increased collocation points for gravity well
    epochs = 2001  # More epochs for gravity well convergence
    layers = [1, 64, 64, 64, 1]  # Optimized network architecture

    # Gamma values from the paper
    gamma_values = [0.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0]

    # Include modes 0 through 2 (can extend to more once working)
    modes = [0, 1, 2]

    # Nonlinearity powers
    nonlinearity_powers = [3]

    for p in nonlinearity_powers:
        # Focus on gravity well first
        all_potentials = ['gravity well']

        for potential_type in all_potentials:
            # IMPROVED domain for gravity well
            if potential_type == 'box':
                lb, ub = 0, 1
            elif potential_type == 'gravity well':
                lb, ub = -1, 20  # Optimized domain
            else:
                lb, ub = -10, 10

            # Create uniform grid with better resolution for gravity well
            # if potential_type == 'gravity well':
            #     # Use non-uniform grid with higher density near x=0
            #     x_neg = np.linspace(lb, 0, N_f // 8)  # Negative domain
            #     x_pos = np.linspace(0, ub, 7 * N_f // 8)  # Positive domain (higher density)
            #     X = np.concatenate([x_neg[:-1], x_pos]).reshape(-1, 1)  # Remove duplicate at x=0
            #
            #     # Test grid
            #     x_test_neg = np.linspace(lb, 0, 100)
            #     x_test_pos = np.linspace(0, ub, 900)
            #     X_test = np.concatenate([x_test_neg[:-1], x_test_pos]).reshape(-1, 1)
            # else:
            X = np.linspace(lb, ub, N_f).reshape(-1, 1)
            X_test = np.linspace(lb, ub, 1000).reshape(-1, 1)

            # Create specific directory
            p_save_dir = f"plots_p{p}_{potential_type}_fixed"
            os.makedirs(p_save_dir, exist_ok=True)

            # Train models with improved implementation
            print(f"\nStarting training for {potential_type} potential...")
            models_by_mode, mu_table, training_history = train_gpe_model(
                gamma_values, modes, p, X, lb, ub, layers, epochs,
                potential_type, lr=5e-4, verbose=True  # Lower learning rate for stability
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