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
    """Custom activation: tanh(x) + 1 + eps"""

    def __init__(self, eps=np.finfo(float).eps):
        super(ShiftedTanh, self).__init__()
        self.eps = eps

    def forward(self, x):
        return torch.tanh(x) + 1.0 + self.eps


class GrossPitaevskiiPINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) for solving the 1D Gross-Pitaevskii Equation.
    Implements Box → Harmonic potential transition via β parameter with PL-PINN strategy.
    """

    def __init__(self, layers, hbar=1.0, m=1.0, mode=0, beta=1.0, L=1.0, omega=2.0,
                 use_residual=True, use_perturbation=True):
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
        beta : float, optional
            Potential transition parameter: β=0 is Box, β=1 is Harmonic.
        L : float, optional
            Length of the box (default is 1.0).
        omega : float, optional
            Harmonic oscillator frequency (default is 2.0).
        use_perturbation: boolean, optional
            Get the complete solution by combining the base solution with the neural network perturbation.
        """
        super().__init__()
        self.layers = layers
        self.use_residual = use_residual
        self.network = self.build_network()
        self.hbar = hbar
        self.m = m
        self.mode = mode
        self.beta = beta
        self.L = L
        self.omega = omega
        self.use_perturbation = use_perturbation

        # PL-PINN normalization constant (set during training initialization)
        self.normal_const = None

    def build_network(self):
        """
        Build the neural network with ShiftedTanh activation functions between layers.
        """
        layers = []
        for i in range(len(self.layers) - 1):
            layers.append(nn.Linear(self.layers[i], self.layers[i + 1]))
            if i < len(self.layers) - 2:
                layers.append(ShiftedTanh())
        return nn.Sequential(*layers)

    def box_eigenfunction(self, x, n):
        """
        Compute the analytic eigenfunction for a particle in a box.

        For the linear case (η = 0), the solution is:
        φ_n(x) = sqrt(2/L) * sin(n*π*x/L)

        This is the base solution we start from (β=0).
        """
        # For mode 0, n=1 in the sine function
        n_actual = n + 1  # Convert mode number to quantum number

        # Normalization factor
        norm_factor = torch.sqrt(torch.tensor(2.0 / self.L))

        # Shift x to [0, L] for sine function
        x_shifted = x + self.L / 2

        # Sine function with proper scaling
        phi_n = norm_factor * torch.sin(n_actual * torch.pi * x_shifted / self.L )

        return phi_n

    def weighted_hermite(self, x: torch.Tensor, n: int) -> torch.Tensor:
        """
        Compute weighted Hermite polynomial: (1/√(2^n n! √π)) * H_n(x) * exp(-x²/2)

        This represents the harmonic oscillator eigenfunction.
        """
        fact_n = float(math.factorial(n))
        norm_factor = ((2.0 ** n) * fact_n * math.sqrt(math.pi)) ** (-0.5)

        # Build H_n(x) via recurrence in torch:
        # H_0(x) = 1
        # H_1(x) = 2x
        # H_k = 2x*H_{k-1} - 2(k-1)*H_{k-2}
        if n == 0:
            Hn = torch.ones_like(x)
        elif n == 1:
            Hn = 2.0 * x
        else:
            Hnm2 = torch.ones_like(x)  # H_0(x)
            Hnm1 = 2.0 * x  # H_1(x)
            for k in range(1, n):
                Hn = 2.0 * x * Hnm1 - 2.0 * float(k) * Hnm2
                Hnm2, Hnm1 = Hnm1, Hn

        weight = torch.exp(-0.5 * x ** 2)

        # Combine with the normalization factor
        return torch.tensor(norm_factor, dtype=x.dtype, device=x.device) * (Hn * weight)

    def forward(self, inputs):
        """
        Forward pass through the neural network.
        Returns the RAW network output (before PL-PINN normalization).
        """
        return self.network(inputs)

    def get_normalized_perturbation(self, x, perturb_const):
        """
        Get the normalized perturbation according to PL-PINN strategy.

        At initialization (when normal_const is first set):
            ψ = (u_raw / max(u_raw)) * δ

        During training:
            ψ = (u_raw * δ) / normal_const

        This keeps the perturbation small and controlled.
        """
        u_raw = self.forward(x)

        if self.normal_const is None:
            raise ValueError("normal_const must be set before calling get_normalized_perturbation")

        # Apply PL-PINN normalization strategy
        psi = (perturb_const * u_raw) / self.normal_const

        return psi

    def get_complete_solution(self, x, perturbation, mode=None):
        """
        Get the complete solution by combining the base Box solution with the perturbation.
        u(x) = φ_box(x) + ψ_NN(x)
        """
        if mode is None:
            mode = self.mode
        base_solution = self.box_eigenfunction(x, mode)
        return base_solution + perturbation

    def compute_pde_residual(self, x, perturb_const, eta=0, p=3, beta=1.0,
                             potential_type="box_to_harmonic", omega=2.0):
        """
        Compute the PDE residual for the Gross-Pitaevskii equation with transition potential.

        PDE: -Δu + V_β(x)u + η|u|^(p-1)u = λu

        where V_β(x) = (1-β)*V_box(x) + β*V_harmonic(x)
              V_box(x) = 0 (inside domain)
              V_harmonic(x) = (1/2)*ω²*x²
        """
        x.requires_grad_(True)

        # Get normalized perturbation using PL-PINN strategy
        psi = self.get_normalized_perturbation(x, perturb_const)

        # Get complete solution
        u = self.get_complete_solution(x, psi, self.mode)

        # Compute first derivative
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]

        # Compute second derivative
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]

        # Compute transition potential: V_β(x) = (1-β)*0 + β*(1/2)*ω²*x²
        if potential_type == "box_to_harmonic":
            # Box potential is V=0 inside domain
            V_box = torch.zeros_like(x)
            # Harmonic potential: V = (1/2)*ω²*x²
            V_harmonic = (1/2) * (omega ** 2) * (x ** 2)
            # Transition potential
            V = (1.0 - beta) * V_box + beta * V_harmonic
        elif potential_type == "box":
            V = torch.zeros_like(x)
        elif potential_type == "harmonic":
            V = 0.5 * (omega ** 2) * (x ** 2)
        else:
            raise ValueError(f"Unknown potential_type: {potential_type}")

        # Nonlinear term: η|u|^(p-1)u
        nonlinear_term = eta * torch.abs(u) ** (p - 1) * u

        # PDE residual: -u_xx + V*u + η|u|^(p-1)u - λ*u
        residual = -u_xx + V * u + nonlinear_term - self.lam * u

        return residual

    def compute_loss(self, x, x_boundary, perturb_const, alpha_PDE=1.0, alpha_BC=10.0, alpha_L2=20.0,
                     eta=0, p=3, beta=1.0, potential_type="box_to_harmonic", omega=2.0):
        """
        Compute the total loss function consisting of:
        1. PDE residual loss
        2. Boundary condition loss
        3. L² normalization loss
        """
        # PDE residual loss
        pde_residual = self.compute_pde_residual(x, perturb_const, eta, p, beta, potential_type, omega)
        loss_pde = torch.mean(pde_residual ** 2)

        # Boundary condition loss: u(boundary) = 0
        psi_boundary = self.get_normalized_perturbation(x_boundary, perturb_const)
        u_boundary = self.get_complete_solution(x_boundary, psi_boundary, self.mode)
        loss_bc = torch.mean(u_boundary ** 2)

        # L² normalization loss: ||u||_L² = 1
        psi = self.get_normalized_perturbation(x, perturb_const)
        u = self.get_complete_solution(x, psi, self.mode)
        # Approximate integral using trapezoidal rule
        domain_length = x[-1] - x[0]
        L2_norm_squared = torch.mean(u ** 2) * domain_length
        loss_L2 = (1.0 - L2_norm_squared) ** 2

        # Total loss
        total_loss = alpha_PDE * loss_pde + alpha_BC * loss_bc + alpha_L2 * loss_L2

        return total_loss, loss_pde, loss_bc, loss_L2


def train_gpe_model(gamma, beta_values, modes, p, X, lb, ub, layers, epochs, tol, perturb_const,
                    potential_type="box_to_harmonic", omega=2.0, lr=1e-3, verbose=True):
    """
    Train the Gross-Pitaevskii PINN for Box → Harmonic transition with PL-PINN strategy.

    Parameters:
    -----------
    gamma : float
        Interaction strength (η in the paper), fixed at 0 for linear regime
    beta_values : list
        List of β values for potential transition (0=Box, 1=Harmonic)
    modes : list
        List of mode numbers to compute
    p : int
        Nonlinearity power
    X : numpy array
        Training points
    lb, ub : float
        Domain boundaries
    layers : list
        Neural network architecture
    epochs : int
        Maximum number of training epochs per β value
    tol : float
        Tolerance for early stopping
    perturb_const : float
        Perturbation constant δ for PL-PINN normalization
    potential_type : str
        Type of potential transition
    omega : float
        Harmonic oscillator frequency
    lr : float
        Learning rate
    verbose : bool
        Print training progress
    """

    models_by_mode = {}
    lambda_table = {mode: {} for mode in modes}
    training_history = {mode: {} for mode in modes}
    constant_history = {mode: {} for mode in modes}
    epochs_history = {mode: {} for mode in modes}

    # Convert to torch tensors
    X_train = torch.tensor(X, dtype=torch.float32).to(device)
    X_boundary = torch.tensor([[lb], [ub]], dtype=torch.float32).to(device)

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"Training Mode {mode}")
        print(f"{'='*60}")

        # Initialize model for β=0 (Box potential)
        model = GrossPitaevskiiPINN(layers, mode=mode, beta=0.0, L=(ub-lb), omega=omega).to(device)

        # Initialize eigenvalue λ as learnable parameter
        # For Box: λ_n = (nπ)²/L² where n = mode + 1
        n_actual = mode + 1
        lambda_init = (n_actual * np.pi / (ub - lb)) ** 2
        model.lam = nn.Parameter(torch.tensor([lambda_init], dtype=torch.float32).to(device))

        # CRITICAL: Initialize normalization constant for PL-PINN strategy
        # This is done ONCE at the very beginning
        with torch.no_grad():
            u_init = model.forward(X_train)
            normal_const = torch.max(torch.abs(u_init)).detach().clone()
            model.normal_const = normal_const
            constant_history[mode]['init'] = normal_const.item()

        if verbose:
            print(f"Initial λ_{mode} = {lambda_init:.6f}")
            print(f"Normalization constant = {normal_const.item():.6f}")
            # Check initial perturbation size
            psi_init = model.get_normalized_perturbation(X_train, perturb_const)
            L2_init = torch.sqrt(torch.mean(psi_init ** 2) * (ub - lb))
            print(f"Initial perturbation L² norm: {L2_init.item():.6f}")

        # Store model for β=0
        models_by_mode[(mode, 0.0)] = {
            'model': model,
            'lambda': model.lam.item(),
            'normal_const': model.normal_const.item()
        }
        lambda_table[mode][0.0] = model.lam.item()

        # Progressive training over β values
        for beta_idx, beta in enumerate(beta_values[1:], 1):
            print(f"\n{'-'*60}")
            print(f"Mode {mode}, β = {beta:.3f} (Step {beta_idx}/{len(beta_values)-1})")
            print(f"{'-'*60}")

            # Update model's β parameter
            model.beta = beta

            # Setup optimizer and scheduler
            optimizer = torch.optim.Adam(list(model.parameters()), lr=lr)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=500, T_mult=2)

            # Training loop for current β
            loss_history = []
            pde_history = []
            bc_history = []
            l2_history = []
            lambda_history = []

            best_loss = float('inf')
            patience_counter = 0
            patience_limit = 2000

            for epoch in range(epochs):
                optimizer.zero_grad()

                # Compute loss with current β
                # Pass perturb_const for PL-PINN normalization
                total_loss, loss_pde, loss_bc, loss_L2 = model.compute_loss(
                    X_train, X_boundary, perturb_const,
                    alpha_PDE=1.0, alpha_BC=10.0, alpha_L2=20.0,
                    eta=gamma, p=p, beta=beta, potential_type=potential_type, omega=omega
                )

                # Backward pass
                total_loss.backward()
                optimizer.step()
                scheduler.step()

                # Record history
                loss_history.append(total_loss.item())
                pde_history.append(loss_pde.item())
                bc_history.append(loss_bc.item())
                l2_history.append(loss_L2.item())
                lambda_history.append(model.lam.item())

                # Early stopping check
                if total_loss.item() < best_loss - tol:
                    best_loss = total_loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience_limit:
                    if verbose:
                        print(f"Early stopping at epoch {epoch} (Loss: {total_loss.item():.2e})")
                    epochs_history[mode][beta] = epoch
                    break

                # Print progress
                if verbose and (epoch % 500 == 0 or epoch == epochs - 1):
                    print(f"Epoch {epoch:5d} | Loss: {total_loss.item():.2e} | "
                          f"PDE: {loss_pde.item():.2e} | BC: {loss_bc.item():.2e} | "
                          f"L²: {loss_L2.item():.2e} | λ: {model.lam.item():.6f}")

            # If didn't early stop, record full epochs
            if patience_counter < patience_limit:
                epochs_history[mode][beta] = epochs

            # Store results for this β
            models_by_mode[(mode, beta)] = {
                'model': model,
                'lambda': model.lam.item(),
                'normal_const': model.normal_const.item()
            }
            lambda_table[mode][beta] = model.lam.item()
            training_history[mode][beta] = {
                'total_loss': loss_history,
                'pde_loss': pde_history,
                'bc_loss': bc_history,
                'l2_loss': l2_history
            }
            constant_history[mode][beta] = lambda_history

            if verbose:
                print(f"\nCompleted β={beta:.3f}: λ_{mode} = {model.lam.item():.6f}")
                # Compute final solution quality
                with torch.no_grad():
                    psi_final = model.get_normalized_perturbation(X_train, perturb_const)
                    u_final = model.get_complete_solution(X_train, psi_final, mode)
                    L2_final = torch.sqrt(torch.mean(u_final ** 2) * (ub - lb))
                    print(f"Final L² norm: {L2_final.item():.6f}")
                    print(f"Perturbation L² norm: {torch.sqrt(torch.mean(psi_final ** 2) * (ub - lb)).item():.6f}")

    return models_by_mode, lambda_table, training_history, constant_history, epochs_history


def save_models(models_by_mode, lambda_table, training_history, constant_history, epochs_history, filename, save_dir):
    """
    Save trained models and training data to disk.
    """
    # Prepare data for saving
    save_data = {
        'lambda_table': lambda_table,
        'training_history': training_history,
        'constant_history': constant_history,
        'epochs_history': epochs_history
    }

    # Save model states and normalization constants
    model_states = {}
    normal_consts = {}
    for key, model_dict in models_by_mode.items():
        mode, beta = key
        model_states[(mode, beta)] = model_dict['model'].state_dict()
        normal_consts[(mode, beta)] = model_dict['normal_const']

    save_data['model_states'] = model_states
    save_data['normal_consts'] = normal_consts
    save_data['model_config'] = {
        'layers': models_by_mode[list(models_by_mode.keys())[0]]['model'].layers,
        'L': models_by_mode[list(models_by_mode.keys())[0]]['model'].L,
        'omega': models_by_mode[list(models_by_mode.keys())[0]]['model'].omega
    }

    # Save to file
    filepath = os.path.join(save_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(save_data, f)

    print(f"\nModels and training data saved to: {filepath}")


def load_models(filename, save_dir):
    """
    Load trained models and training data from disk.
    """
    filepath = os.path.join(save_dir, filename)

    with open(filepath, 'rb') as f:
        save_data = pickle.load(f)

    # Reconstruct models
    models_by_mode = {}
    model_states = save_data['model_states']
    normal_consts = save_data['normal_consts']
    config = save_data['model_config']

    for key, state_dict in model_states.items():
        mode, beta = key
        model = GrossPitaevskiiPINN(
            config['layers'],
            mode=mode,
            beta=beta,
            L=config['L'],
            omega=config['omega']
        ).to(device)
        model.load_state_dict(state_dict)
        model.normal_const = torch.tensor(normal_consts[key]).to(device)
        models_by_mode[(mode, beta)] = {
            'model': model,
            'lambda': save_data['lambda_table'][mode][beta],
            'normal_const': normal_consts[key]
        }

    print(f"\nModels loaded from: {filepath}")

    return (models_by_mode,
            save_data['lambda_table'],
            save_data['training_history'],
            save_data['constant_history'],
            save_data['epochs_history'])


def plot_wavefunction(models_by_mode, X_test, beta_values, modes, p, constant_history,
                      perturb_const, potential_type, lb, ub, save_dir):
    """
    Plot wavefunctions for all modes with all β values on single plot per mode.
    Matches the style of η variation plots (all curves overlaid with different colors).
    """
    os.makedirs(save_dir, exist_ok=True)

    # Select specific β values to plot (matching the η visualization)
    plot_betas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    plot_betas = [b for b in plot_betas if b in beta_values]

    # Define colors and line styles matching the η plots
    colors = ['black', 'cyan', 'green', 'blue', 'gray', 'magenta']
    linestyles = ['-', '-', '-.', '-', '-', '-.']

    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    for mode in modes:
        # Create single plot for this mode
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Plot all β values on same axes
        for idx, beta in enumerate(plot_betas):
            if (mode, beta) not in models_by_mode:
                continue

            model_dict = models_by_mode[(mode, beta)]
            model = model_dict['model']

            # Get solution using PL-PINN strategy
            with torch.no_grad():
                model.eval()
                psi = model.get_normalized_perturbation(X_tensor, perturb_const)
                u = model.get_complete_solution(X_tensor, psi, mode)
                u_np = u.cpu().numpy().flatten()

            # Plot solution with color and style matching η plots
            ax.plot(X_test, u_np,
                    color=colors[idx % len(colors)],
                    linestyle=linestyles[idx % len(linestyles)],
                    linewidth=2.5,
                    label=f'β={beta:.1f}')

        # Formatting to match η plots
        ax.set_xlabel('x', fontsize=20)
        ax.set_ylabel('u(x)', fontsize=20)
        ax.set_title(f'Mode {mode} Wavefunction', fontsize=24)
        ax.axhline(0, color='k', linewidth=0.8, alpha=0.3)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=18, loc='best')

        # Set reasonable axis limits
        ax.set_xlim(lb, ub)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'mode_{mode}_wavefunction_p{p}_{potential_type}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved wavefunction plot for mode {mode}")


def plot_all_modes_grid(models_by_mode, X_test, beta_values, modes, p, constant_history,
                        perturb_const, potential_type, lb, ub, save_dir):
    """
    Create a grid showing all modes together for paper-style comprehensive visualization.
    Shows modes 0-5 in a 2x3 grid with all β values overlaid.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Select specific β values to plot
    plot_betas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    plot_betas = [b for b in plot_betas if b in beta_values]

    # Define colors and line styles
    colors = ['black', 'cyan', 'green', 'blue', 'gray', 'magenta']
    linestyles = ['-', '-', '-.', '-', '-', '-.']

    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    # Create 2x3 grid for modes 0-5
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    for mode_idx, mode in enumerate(modes[:6]):  # Limit to 6 modes for 2x3 grid
        ax = axes[mode_idx]

        # Plot all β values for this mode
        for idx, beta in enumerate(plot_betas):
            if (mode, beta) not in models_by_mode:
                continue

            model_dict = models_by_mode[(mode, beta)]
            model = model_dict['model']

            # Get solution
            with torch.no_grad():
                model.eval()
                psi = model.get_normalized_perturbation(X_tensor, perturb_const)
                u = model.get_complete_solution(X_tensor, psi, mode)
                u_np = u.cpu().numpy().flatten()

            # Plot with consistent styling
            ax.plot(X_test, u_np,
                    color=colors[idx % len(colors)],
                    linestyle=linestyles[idx % len(linestyles)],
                    linewidth=2.0,
                    label=f'β={beta:.1f}')

        # Formatting
        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('u(x)', fontsize=14)
        ax.set_title(f'Mode {mode}', fontsize=16)
        ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(lb, ub)

        # Only show legend on first subplot
        if mode_idx == 0:
            ax.legend(fontsize=12, loc='best')

    plt.suptitle(f'Box → Harmonic Transition: All Modes, p={p}', fontsize=20, y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'all_modes_wavefunction_p{p}_{potential_type}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved combined all-modes grid plot")


def plot_lambda_vs_beta(lambda_table, modes, p, potential_type, save_dir):
    """
    Plot eigenvalue λ vs β for all modes.
    Shows transition from Box eigenvalues to Harmonic eigenvalues.
    """
    plt.figure(figsize=(12, 8))

    colors = ['k', 'b', 'r', 'g', 'm', 'c', 'orange', 'purple']
    markers = ['o', 's', '^', 'v', 'D', 'x', '*', '+']

    for i, mode in enumerate(modes):
        if mode in lambda_table:
            betas = sorted(lambda_table[mode].keys())
            lambdas = [lambda_table[mode][b] for b in betas]

            plt.plot(betas, lambdas,
                     color=colors[i % len(colors)],
                     marker=markers[i % len(markers)],
                     linestyle='-',
                     linewidth=2,
                     markersize=6,
                     label=f'Mode {mode}')

    plt.xlabel(r'$\beta$ (Transition Parameter)', fontsize=16)
    plt.ylabel(r'$\lambda$ (Eigenvalue)', fontsize=16)
    plt.title(f'Eigenvalue Evolution: Box → Harmonic Transition (p={p})', fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=14)

    # Add annotation
    plt.text(0.02, 0.98, 'β=0: Box\nβ=1: Harmonic',
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'lambda_vs_beta_p{p}_{potential_type}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved λ vs β plot")


def plot_improved_loss_visualization(training_history, modes, beta_values, epochs, p, potential_type, save_dir):
    """
    Create comprehensive loss visualization showing convergence behavior.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Select representative β values
    plot_betas = [0.0, 0.2, 0.5, 0.8, 1.0]
    plot_betas = [b for b in plot_betas if b in beta_values]

    for mode in modes:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Total loss over epochs for different β
        ax = axes[0, 0]
        for beta in plot_betas:
            if beta in training_history[mode]:
                history = training_history[mode][beta]
                ax.semilogy(history['total_loss'], label=f'β={beta:.1f}', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss (log scale)')
        ax.set_title(f'Total Loss Convergence - Mode {mode}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: PDE residual loss
        ax = axes[0, 1]
        for beta in plot_betas:
            if beta in training_history[mode]:
                history = training_history[mode][beta]
                ax.semilogy(history['pde_loss'], label=f'β={beta:.1f}', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('PDE Loss (log scale)')
        ax.set_title(f'PDE Residual - Mode {mode}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Boundary condition loss
        ax = axes[1, 0]
        for beta in plot_betas:
            if beta in training_history[mode]:
                history = training_history[mode][beta]
                ax.semilogy(history['bc_loss'], label=f'β={beta:.1f}', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('BC Loss (log scale)')
        ax.set_title(f'Boundary Condition Loss - Mode {mode}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: L² normalization loss
        ax = axes[1, 1]
        for beta in plot_betas:
            if beta in training_history[mode]:
                history = training_history[mode][beta]
                ax.semilogy(history['l2_loss'], label=f'β={beta:.1f}', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('L² Loss (log scale)')
        ax.set_title(f'L² Normalization Loss - Mode {mode}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle(f'Training Convergence Analysis: Mode {mode}, p={p}', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'loss_analysis_mode_{mode}_p{p}_{potential_type}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved loss analysis plot for mode {mode}")


def plot_all_modes_beta_loss(training_history, modes, beta_values, epochs, p, potential_type, save_dir):
    """
    Create heatmap showing final loss for all mode-β combinations.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Prepare data
    final_losses = np.zeros((len(modes), len(beta_values)))

    for i, mode in enumerate(modes):
        for j, beta in enumerate(beta_values):
            if mode in training_history and beta in training_history[mode]:
                history = training_history[mode][beta]
                final_losses[i, j] = history['total_loss'][-1]
            else:
                final_losses[i, j] = np.nan

    # Create heatmap
    plt.figure(figsize=(14, 8))

    im = plt.imshow(final_losses, cmap='viridis', aspect='auto',
                    interpolation='nearest', norm=mpl.colors.LogNorm())

    plt.colorbar(im, label='Final Total Loss')

    # Set ticks
    plt.yticks(range(len(modes)), [f'Mode {m}' for m in modes])

    # Show fewer β labels for readability
    tick_indices = np.linspace(0, len(beta_values)-1, min(10, len(beta_values)), dtype=int)
    plt.xticks(tick_indices, [f'{beta_values[i]:.2f}' for i in tick_indices], rotation=45)

    plt.xlabel(r'$\beta$ (Transition Parameter)', fontsize=14)
    plt.ylabel('Mode', fontsize=14)
    plt.title(f'Final Loss Heatmap: Box → Harmonic (p={p})', fontsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'loss_heatmap_p{p}_{potential_type}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved loss heatmap")


def plot_epochs_until_stopping(epochs_history, modes, beta_values, p, potential_type, save_dir):
    """
    Plot the number of epochs until early stopping for different modes and β values.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Plot 1: Line plot showing epochs vs β for each mode
    plt.figure(figsize=(12, 7))
    colors = ['k', 'b', 'r', 'g', 'm', 'c', 'orange', 'purple']
    markers = ['o', 's', '^', 'v', 'D', 'x', '*', '+']

    for i, mode in enumerate(modes):
        if mode in epochs_history:
            betas = []
            epochs_list = []

            for beta in beta_values:
                if beta in epochs_history[mode]:
                    betas.append(beta)
                    epochs_list.append(epochs_history[mode][beta])

            if betas:
                plt.plot(betas, epochs_list,
                         color=colors[i % len(colors)],
                         marker=markers[i % len(markers)],
                         linestyle='-',
                         linewidth=2,
                         markersize=6,
                         label=f"Mode {mode}")

    plt.xlabel(r'$\beta$ (Transition Parameter)', fontsize=14)
    plt.ylabel('Epochs Until Convergence', fontsize=14)
    plt.title(f'Training Efficiency: Box → Harmonic (p={p})', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'epochs_vs_beta_p{p}_{potential_type}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved epochs analysis plot")

    # Print statistics
    print("\n=== Early Stopping Statistics ===")
    for mode in modes:
        if mode in epochs_history:
            epochs_list = list(epochs_history[mode].values())
            print(f"Mode {mode}:")
            print(f"  Average epochs: {np.mean(epochs_list):.1f}")
            print(f"  Min epochs: {np.min(epochs_list)}")
            print(f"  Max epochs: {np.max(epochs_list)}")
            print(f"  Std epochs: {np.std(epochs_list):.1f}")


if __name__ == "__main__":
    # Setup parameters
    lb, ub = -5, 5  # Domain boundaries for Box potential
    N_f = 4000  # Number of collocation points
    epochs = 1001  # Maximum epochs per β value
    layers = [1, 64, 64, 64, 1]  # Neural network architecture

    # Create uniform grid for training and testing
    X = np.linspace(lb, ub, N_f).reshape(-1, 1)
    X_test = np.linspace(lb, ub, 1000).reshape(-1, 1)

    # Fix interaction strength η = 0 (linear regime - varying potential only)
    gamma = 0

    # Harmonic oscillator frequency
    omega = 2.0

    # β transition parameter: 0 (Box) → 1 (Harmonic)
    beta_step = 5.0
    beta_values = [k * beta_step for k in range(21)]  # [0.0, 0.05, 0.10, ..., 1.0]

    # Include modes 0 through 5
    modes = [0, 1, 2]

    # Set the perturbation constant (δ in paper)
    perturb_const = 0.01

    # Set the tolerance for early stopping
    tol = 1e-6

    # Nonlinearity powers (p=3 is Gross-Pitaevskii)
    nonlinearity_powers = [3]

    for p in nonlinearity_powers:

        # Specify potential type
        potential_type = "box_to_harmonic"

        # Train neural network or load existing models
        train_new = True  # Set to True to train, False to load
        filename = f"vary_potential_box_to_harmonic_p{p}.pkl"

        # Create plotting and model saving directory
        p_save_dir = f"vary_potential_box_to_harmonic_p{p}"
        os.makedirs(p_save_dir, exist_ok=True)

        if train_new:
            # Train models
            print("="*70)
            print(" "*15 + "STARTING PL-PINN TRAINING")
            print("="*70)
            print(f"Configuration:")
            print(f"  Domain: [{lb}, {ub}]")
            print(f"  β range: {beta_values[0]:.2f} → {beta_values[-1]:.2f} ({len(beta_values)} steps)")
            print(f"  Modes: {modes}")
            print(f"  Nonlinearity: p={p}")
            print(f"  Network: {layers}")
            print(f"  Harmonic ω: {omega}")
            print(f"  Perturbation δ: {perturb_const}")
            print("="*70)

            models_by_mode, lambda_table, training_history, constant_history, epochs_history = train_gpe_model(
                gamma, beta_values, modes, p, X, lb, ub, layers, epochs, tol, perturb_const,
                potential_type=potential_type, omega=omega, lr=1e-3, verbose=True)

            # Save results
            save_models(models_by_mode, lambda_table, training_history, constant_history,
                        epochs_history, filename, p_save_dir)
        else:
            # Load existing models
            print("Loading existing models...")
            models_by_mode, lambda_table, training_history, constant_history, epochs_history = load_models(
                filename, p_save_dir)

        # Generate plots
        print("\n" + "="*70)
        print(" "*20 + "GENERATING PLOTS")
        print("="*70)

        print("Generating individual mode wavefunction plots...")
        plot_wavefunction(models_by_mode, X_test, beta_values, modes, p, constant_history,
                          perturb_const, potential_type, lb, ub, p_save_dir)

        print("Generating combined all-modes grid plot...")
        plot_all_modes_grid(models_by_mode, X_test, beta_values, modes, p, constant_history,
                            perturb_const, potential_type, lb, ub, p_save_dir)

        print("Generating λ vs β plot...")
        plot_lambda_vs_beta(lambda_table, modes, p, potential_type, p_save_dir)

        print("Generating loss analysis plots...")
        plot_improved_loss_visualization(training_history, modes, beta_values, epochs, p,
                                         potential_type, p_save_dir)
        plot_all_modes_beta_loss(training_history, modes, beta_values, epochs, p,
                                 potential_type, p_save_dir)

        print("Generating early stopping analysis...")
        plot_epochs_until_stopping(epochs_history, modes, beta_values, p, potential_type, p_save_dir)

        print("\n" + "="*70)
        print(" "*20 + "TRAINING COMPLETE")
        print("="*70)
        print(f"Results saved to: {p_save_dir}/")
        print("="*70)