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

mpl.rcParams['text.usetex'] = False

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
    MODIFIED: Now tracks both beta (potential) and gamma (interaction) parameters.
    """

    def __init__(self, layers, hbar=1.0, m=1.0, mode=0, beta=1.0, gamma=0.0, use_perturbation=True):
        """
        Parameters
        ----------
        layers : list of int
            Neural network architecture
        hbar : float, optional
            Reduced Planck's constant (default is 1.0)
        m : float, optional
            Mass of the particle (default is 1.0)
        mode : int, optional
            Mode number (default is 0)
        beta : float, optional
            Potential strength parameter (default is 1.0)
        gamma : float, optional
            Interaction strength parameter (default is 0.0)
        use_perturbation: boolean, optional
            Use perturbation learning approach (default is True)
        """
        super().__init__()
        self.layers = layers
        self.network = self.build_network()
        self.hbar = hbar
        self.m = m
        self.mode = mode
        self.beta = beta  # Potential parameter
        self.gamma = gamma  # Interaction strength parameter
        self.use_perturbation = use_perturbation

    def build_network(self):
        """Build the neural network with tanh activation functions between layers."""
        layers = []
        for i in range(len(self.layers) - 1):
            layers.append(nn.Linear(self.layers[i], self.layers[i + 1]))
            if i < len(self.layers) - 2:
                layers.append(ShiftedTanh())
        return nn.Sequential(*layers)

    def weighted_hermite(self, x: torch.Tensor, n: int) -> torch.Tensor:
        """Compute weighted Hermite polynomial (base solution for beta=1)."""
        fact_n = float(math.factorial(n))
        norm_factor = ((2.0 ** n) * fact_n * math.sqrt(math.pi)) ** (-0.5)

        if n == 0:
            Hn = torch.ones_like(x)
        elif n == 1:
            Hn = 2.0 * x
        else:
            Hnm2 = torch.ones_like(x)
            Hnm1 = 2.0 * x
            for k in range(1, n):
                Hn = 2.0 * x * Hnm1 - 2.0 * float(k) * Hnm2
                Hnm2, Hnm1 = Hnm1, Hn

        weight = torch.exp(-0.5 * x ** 2)
        return torch.tensor(norm_factor, dtype=x.dtype, device=x.device) * (Hn * weight)

    def forward(self, inputs):
        """Forward pass through the neural network."""
        return self.network(inputs)

    def get_complete_solution(self, x, perturbation, mode=None):
        """Get complete solution by combining base Hermite solution with neural network perturbation."""
        if mode is None:
            mode = self.mode
        base_solution = self.weighted_hermite(x, mode)
        return base_solution + perturbation

    def compute_potential(self, x, potential_type="harmonic", **kwargs):
        """
        Compute potential function for the 1D domain.
        MODIFIED: Now includes beta scaling.
        """
        if potential_type == "harmonic":
            V = self.beta * (x ** 2)  # Apply beta scaling
        else:
            raise ValueError(f"Unknown potential type: {potential_type}")
        return V

    def pde_loss(self, inputs, predictions, p, potential_type="harmonic", precomputed_potential=None):
        """
        Compute the PDE loss for the Gross-Pitaevskii equation.
        MODIFIED: Uses self.beta and self.gamma internally (removed from arguments).
        μψ = -∇²ψ + V(β)ψ + γ|ψ|^p ψ
        """
        # Get complete solution
        if self.use_perturbation:
            u = self.get_complete_solution(inputs, predictions)
        else:
            u = predictions

        # Compute derivatives
        u_x = torch.autograd.grad(
            outputs=u, inputs=inputs,
            grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]

        u_xx = torch.autograd.grad(
            outputs=u_x, inputs=inputs,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True, retain_graph=True
        )[0]

        # Compute potential (includes beta)
        if precomputed_potential is not None:
            V = precomputed_potential
        else:
            V = self.compute_potential(inputs, potential_type)

        # Calculate chemical potential
        kinetic = -u_xx
        potential = V * u
        interaction = self.gamma * u ** p  # Use self.gamma

        numerator = torch.mean(u * (kinetic + potential + interaction))
        denominator = torch.mean(u ** 2)
        lambda_pde = numerator / denominator

        # PDE residual
        pde_residual = kinetic + potential + interaction - lambda_pde * u
        pde_loss = torch.mean(pde_residual ** 2)

        return pde_loss, lambda_pde

    def boundary_loss(self, boundary_points, boundary_values):
        """Compute the boundary loss for the boundary conditions."""
        u_pred = self.forward(boundary_points)

        if self.use_perturbation:
            full_u = self.get_complete_solution(boundary_points, u_pred)
        else:
            full_u = u_pred

        return torch.mean((full_u - boundary_values) ** 2)

    def normalization_loss(self, u, dx):
        """Compute normalization loss using proper numerical integration."""
        integral = torch.sum(u ** 2) * dx
        return (integral - 1.0) ** 2


def train_gpe_model_two_stage(modes, beta_values, gamma_values, p, X_train, lb, ub, layers,
                              epochs_per_stage, tol, perturb_const,
                              potential_type='harmonic', lr=1e-3, verbose=True):
    """
    Two-stage training approach:
    Stage 1: Vary beta with gamma=0 (linear problem)
    Stage 2: Fix beta at max value, vary gamma (nonlinear problem)

    Parameters:
    -----------
    modes : list of int
        List of modes to train (0, 1, 2, 3, etc.)
    beta_values : list of float
        Beta values for Stage 1
    gamma_values : list of float
        Gamma values for Stage 2
    p : int
        Parameter for nonlinearity power
    X_train : numpy.ndarray
        Training points array
    lb, ub : float
        Lower and upper boundaries of the domain
    layers : list of int
        Network architecture
    epochs_per_stage : int
        Number of training epochs per parameter step
    tol : float
        The desired tolerance for early stopping
    perturb_const : float
        A small constant representing the perturbation parameter
    potential_type : str
        Type of potential ('harmonic', etc.)
    lr : float
        Learning rate
    verbose : bool
        Whether to print training progress

    Returns:
    --------
    tuple: (models_by_mode, lambda_table, training_history, constant_history, epochs_history)
        Models organized by (beta, gamma) tuples, eigenvalues, and training histories
    """
    dx = X_train[1, 0] - X_train[0, 0]
    X_tensor = torch.tensor(X_train, dtype=torch.float32, requires_grad=True).to(device)

    boundary_points = torch.tensor([[lb], [ub]], dtype=torch.float32).to(device)
    boundary_values = torch.zeros((2, 1), dtype=torch.float32).to(device)

    models_by_mode = {}
    lambda_table = {}
    training_history = {}
    constant_history = {}
    epochs_history = {}

    beta_values = sorted(beta_values)
    gamma_values = sorted(gamma_values)

    print(f"\n{'=' * 70}")
    print("TWO-STAGE PERTURBATION LEARNING")
    print(f"{'=' * 70}")
    print(f"Stage 1: β ∈ {beta_values} with γ=0 (Linear)")
    print(f"Stage 2: γ ∈ {gamma_values} with β={beta_values[-1]} (Nonlinear)")
    print(f"Tolerance: {tol}, Perturbation: {perturb_const}")
    print(f"{'=' * 70}\n")

    for mode in modes:
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"MODE {mode}")
            print(f"{'=' * 60}")

        models_dict = {}  # Store models by (beta, gamma) tuples
        lambda_dict = {}
        history_dict = {}
        epochs_dict = {}

        # ===== STAGE 1: Vary Beta with Gamma=0 =====
        if verbose:
            print(f"\n--- Stage 1: Varying β (γ=0) ---")

        prev_model = None
        stage1_gamma = 0.0

        for beta in beta_values:
            if verbose:
                print(f"\nβ={beta:.2f}, γ={stage1_gamma:.2f}")

            model = GrossPitaevskiiPINN(
                layers, mode=mode, beta=beta, gamma=stage1_gamma
            ).to(device)

            # Transfer from previous beta or pre-train
            if prev_model is not None:
                model.load_state_dict(prev_model.state_dict())
            elif beta == beta_values[0]:
                if verbose:
                    print("  Pre-training on analytical solution...")
                model = pretrain_on_analytical_solution(
                    model, mode, X_train, epochs=2000, lr=1e-3, verbose=False
                )
            else:
                model.apply(lambda m: advanced_initialization(m, mode))

            # Train this (beta, gamma) combination
            model, hist, final_epoch, normal_const = train_single_parameter_step(
                model, mode, X_tensor, boundary_points, boundary_values,
                dx, epochs_per_stage, lr, tol, perturb_const, p,
                potential_type, verbose, constant_history
            )

            # Store results with (beta, gamma) key
            param_key = (beta, stage1_gamma)
            models_dict[param_key] = model

            # Get final lambda
            final_lambda = hist['lambda'][-1] if hist['lambda'] else 0
            lambda_dict[param_key] = final_lambda

            history_dict[param_key] = hist
            epochs_dict[param_key] = final_epoch

            prev_model = model

        # ===== STAGE 2: Vary Gamma with Beta Fixed =====
        if verbose:
            print(f"\n--- Stage 2: Varying γ (β={beta_values[-1]:.2f}) ---")

        stage2_beta = beta_values[-1]
        prev_model = models_dict[(stage2_beta, 0.0)]  # Start from end of Stage 1

        for gamma in gamma_values:
            if gamma == 0.0:
                # Already trained in Stage 1
                continue

            if verbose:
                print(f"\nβ={stage2_beta:.2f}, γ={gamma:.2f}")

            model = GrossPitaevskiiPINN(
                layers, mode=mode, beta=stage2_beta, gamma=gamma
            ).to(device)

            # Transfer from previous gamma
            if prev_model is not None:
                model.load_state_dict(prev_model.state_dict())

            # Train this (beta, gamma) combination
            model, hist, final_epoch, normal_const = train_single_parameter_step(
                model, mode, X_tensor, boundary_points, boundary_values,
                dx, epochs_per_stage, lr, tol, perturb_const, p,
                potential_type, verbose, constant_history
            )

            # Store results
            param_key = (stage2_beta, gamma)
            models_dict[param_key] = model

            final_lambda = hist['lambda'][-1] if hist['lambda'] else 0
            lambda_dict[param_key] = final_lambda

            history_dict[param_key] = hist
            epochs_dict[param_key] = final_epoch

            prev_model = model

        # Store all results for this mode
        models_by_mode[mode] = models_dict
        lambda_table[mode] = lambda_dict
        training_history[mode] = history_dict
        epochs_history[mode] = epochs_dict

    return models_by_mode, lambda_table, training_history, constant_history, epochs_history


def train_single_parameter_step(model, mode, X_tensor, boundary_points, boundary_values,
                                dx, epochs, lr, tol, perturb_const, p, potential_type,
                                verbose, constant_history):
    """
    Train model for a single (beta, gamma) combination.
    Helper function for two-stage training.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=200, T_mult=2, eta_min=1e-6
    )

    lambda_history = []
    loss_history = []
    constraint_history = []

    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    patience = 2000
    final_epoch = epochs

    # Get or initialize normalization constant
    if mode not in constant_history:
        normal_const = None
    else:
        normal_const = constant_history[mode]

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        u_pred = model.forward(X_tensor)

        # Handle normalization constant
        if normal_const is None and epoch == 0:
            normal_const = torch.max(torch.abs(u_pred)).detach().clone() + 1e-8
            constant_history[mode] = normal_const

        # Apply perturbation scaling
        if normal_const is not None:
            u_pred = perturb_const * u_pred / normal_const
        else:
            u_pred = perturb_const * u_pred

        # Compute losses
        boundary_loss = model.boundary_loss(boundary_points, boundary_values)
        norm_loss = model.normalization_loss(
            model.get_complete_solution(X_tensor, u_pred), dx
        )
        constraint_loss = 10.0 * boundary_loss + 20.0 * norm_loss

        # PDE loss (model uses its own beta and gamma)
        pde_loss, lambda_value = model.pde_loss(X_tensor, u_pred, p, potential_type)

        total_loss = pde_loss + constraint_loss

        # Backprop
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(total_loss)

        current_loss = total_loss.item()

        # Early stopping
        if current_loss < best_loss:
            best_loss = current_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        # Record history
        if epoch % 10 == 0:
            loss_history.append(current_loss)

        if epoch % 100 == 0:
            lambda_history.append(lambda_value.item())
            constraint_history.append(constraint_loss.item())

            if verbose and epoch % 500 == 0:
                print(f"  Epoch {epoch}: λ={lambda_value.item():.4f}, "
                      f"Loss={current_loss:.6f}")

        # Early stopping
        if current_loss <= tol:
            if verbose:
                print(f"  Converged at epoch {epoch}")
            final_epoch = epoch
            break

        if patience_counter >= patience:
            if verbose:
                print(f"  Early stop at epoch {epoch}")
            final_epoch = epoch
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    history = {
        'loss': loss_history,
        'constraint': constraint_history,
        'lambda': lambda_history
    }

    return model, history, final_epoch, normal_const


def plot_wavefunction_two_stage(models_by_mode, X_test, beta_values, gamma_values,
                                modes, p, constant_history, perturb_const,
                                potential_type, save_dir="results"):
    """
    Plot wavefunctions for two-stage training.
    Creates separate plots for Stage 1 (varying β) and Stage 2 (varying γ).
    """
    os.makedirs(save_dir, exist_ok=True)

    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    dx = X_test[1, 0] - X_test[0, 0]

    for mode in modes:
        if mode not in models_by_mode:
            continue

        # ===== PLOT STAGE 1: Varying Beta (γ=0) =====
        plt.figure(figsize=(10, 6))

        linestyles = ['-', '--', '-.', ':', '-']
        colors = ['k', 'b', 'r', 'g', 'm', 'c']

        stage1_gamma = 0.0
        for j, beta in enumerate(beta_values):
            param_key = (beta, stage1_gamma)

            if param_key not in models_by_mode[mode]:
                continue

            model = models_by_mode[mode][param_key]
            model.eval()
            const = constant_history[mode]

            with torch.no_grad():
                u_pred = model.forward(X_tensor)
                u_pred = u_pred * (perturb_const / const)
                full_u = model.get_complete_solution(X_tensor, u_pred)
                u_np = full_u.cpu().numpy().flatten()
                u_np /= np.sqrt(np.sum(u_np ** 2) * dx)

                if mode == 0:
                    u_np = np.abs(u_np)

                plt.plot(X_test.flatten(), u_np,
                         linestyle=linestyles[j % len(linestyles)],
                         color=colors[j % len(colors)],
                         label=f"β={beta:.1f}")

        plt.title(f"Mode {mode}: Stage 1 (Varying β, γ=0)", fontsize=16)
        plt.xlabel("x", fontsize=14)
        plt.ylabel("ψ(x)", fontsize=14)
        plt.grid(True)
        plt.legend(fontsize=10)
        plt.xlim(-10, 10)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir,
                                 f"mode_{mode}_stage1_beta_variation_p{p}.png"), dpi=300)
        plt.close()

        # ===== PLOT STAGE 2: Varying Gamma (β=β_max) =====
        plt.figure(figsize=(10, 6))

        stage2_beta = beta_values[-1]
        for j, gamma in enumerate(gamma_values):
            param_key = (stage2_beta, gamma)

            if param_key not in models_by_mode[mode]:
                continue

            model = models_by_mode[mode][param_key]
            model.eval()
            const = constant_history[mode]

            with torch.no_grad():
                u_pred = model.forward(X_tensor)
                u_pred = u_pred * (perturb_const / const)
                full_u = model.get_complete_solution(X_tensor, u_pred)
                u_np = full_u.cpu().numpy().flatten()
                u_np /= np.sqrt(np.sum(u_np ** 2) * dx)

                if mode == 0:
                    u_np = np.abs(u_np)

                if gamma % 5 == 0 or gamma == 0:  # Plot subset
                    plt.plot(X_test.flatten(), u_np,
                             linestyle=linestyles[j % len(linestyles)],
                             color=colors[j % len(colors)],
                             label=f"γ={gamma:.1f}")

        plt.title(f"Mode {mode}: Stage 2 (Varying γ, β={stage2_beta:.1f})",
                  fontsize=16)
        plt.xlabel("x", fontsize=14)
        plt.ylabel("ψ(x)", fontsize=14)
        plt.grid(True)
        plt.legend(fontsize=10)
        plt.xlim(-10, 10)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir,
                                 f"mode_{mode}_stage2_gamma_variation_p{p}.png"), dpi=300)
        plt.close()


def plot_eigenvalues_two_stage(lambda_table, modes, beta_values, gamma_values,
                               p, potential_type, save_dir="results"):
    """
    Plot eigenvalues for both stages.
    """
    os.makedirs(save_dir, exist_ok=True)

    markers = ['o', 's', '^', 'v', 'D', 'x']
    colors = ['k', 'b', 'r', 'g', 'm', 'c']

    # ===== PLOT STAGE 1: λ vs β =====
    plt.figure(figsize=(10, 6))

    stage1_gamma = 0.0
    for i, mode in enumerate(modes):
        if mode not in lambda_table:
            continue

        betas = []
        lambdas = []

        for beta in beta_values:
            param_key = (beta, stage1_gamma)
            if param_key in lambda_table[mode]:
                betas.append(beta)
                lambdas.append(lambda_table[mode][param_key])

        if betas:
            plt.plot(betas, lambdas,
                     marker=markers[i % len(markers)],
                     color=colors[i % len(colors)],
                     linewidth=2, markersize=8,
                     label=f"Mode {mode}")

    plt.xlabel("β (Potential Strength)", fontsize=14)
    plt.ylabel("λ (Eigenvalue)", fontsize=14)
    plt.title("Stage 1: Eigenvalue vs β (γ=0)", fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"eigenvalues_stage1_p{p}.png"), dpi=300)
    plt.close()

    # ===== PLOT STAGE 2: λ vs γ =====
    plt.figure(figsize=(10, 6))

    stage2_beta = beta_values[-1]
    for i, mode in enumerate(modes):
        if mode not in lambda_table:
            continue

        gammas = []
        lambdas = []

        for gamma in gamma_values:
            param_key = (stage2_beta, gamma)
            if param_key in lambda_table[mode]:
                gammas.append(gamma)
                lambdas.append(lambda_table[mode][param_key])

        if gammas:
            plt.plot(gammas, lambdas,
                     marker=markers[i % len(markers)],
                     color=colors[i % len(colors)],
                     linewidth=2, markersize=8,
                     label=f"Mode {mode}")

    plt.xlabel("γ (Interaction Strength)", fontsize=14)
    plt.ylabel("λ (Eigenvalue)", fontsize=14)
    plt.title(f"Stage 2: Eigenvalue vs γ (β={stage2_beta:.1f})", fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"eigenvalues_stage2_p{p}.png"), dpi=300)
    plt.close()


def save_models_two_stage(models_by_mode, lambda_table, training_history,
                          constant_history, epochs_history, beta_values, gamma_values,
                          filename="gpe_models_two_stage.pkl", save_dir="results"):
    """Save two-stage training results."""
    os.makedirs(save_dir, exist_ok=True)

    models_state_dicts = {}
    for mode in models_by_mode:
        models_state_dicts[mode] = {}
        for param_key, model in models_by_mode[mode].items():
            beta, gamma = param_key
            models_state_dicts[mode][param_key] = {
                'state_dict': model.cpu().state_dict(),
                'layers': model.layers,
                'mode': model.mode,
                'beta': beta,
                'gamma': gamma
            }

    data = {
        'models_state_dicts': models_state_dicts,
        'lambda_table': lambda_table,
        'training_history': training_history,
        'constant_history': constant_history,
        'epochs_history': epochs_history,
        'beta_values': beta_values,
        'gamma_values': gamma_values
    }

    filepath = os.path.join(save_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"\nTwo-stage models saved to {filepath}")


def load_models_two_stage(filename="gpe_models_two_stage.pkl", save_dir="results"):
    """Load two-stage training results."""
    filepath = os.path.join(save_dir, filename)

    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    models_by_mode = {}
    for mode in data['models_state_dicts']:
        models_by_mode[mode] = {}
        for param_key, model_data in data['models_state_dicts'][mode].items():
            model = GrossPitaevskiiPINN(
                layers=model_data['layers'],
                mode=model_data['mode'],
                beta=model_data['beta'],
                gamma=model_data['gamma']
            ).to(device)
            model.load_state_dict(model_data['state_dict'])
            model.eval()
            models_by_mode[mode][param_key] = model

    print(f"Two-stage models loaded from {filepath}")
    return (models_by_mode, data['lambda_table'], data['training_history'],
            data['constant_history'], data['epochs_history'],
            data['beta_values'], data['gamma_values'])


def advanced_initialization(m, mode):
    """Initialize network weights with consideration of the mode number"""
    if isinstance(m, nn.Linear):
        gain = 1.0 / (1.0 + 0.2 * mode)
        nn.init.xavier_normal_(m.weight, gain=gain)

        if mode > 3:
            m.bias.data.fill_(0.001)
        else:
            m.bias.data.fill_(0.01)


def pretrain_on_analytical_solution(model, mode, X_train, epochs=5000, lr=1e-3, verbose=False):
    """Pre-train network to output the analytical harmonic oscillator solution."""
    model.train()
    X_tensor = torch.tensor(X_train, dtype=torch.float32, requires_grad=True).to(device)

    with torch.no_grad():
        analytical_target = model.weighted_hermite(X_tensor.squeeze(), mode).unsqueeze(-1).detach()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    closure = None

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

            loss = optimizer.step(closure)

        if verbose and epoch % 500 == 0:
            print(f"  Pre-training epoch {epoch}, loss: {loss.item():.2e}")

        if loss.item() < 1e-12:
            if verbose:
                print(f"  Pre-training converged at epoch {epoch}, loss: {loss.item():.2e}")
            break

    if verbose:
        print(f"  Pre-training finished, final loss: {loss.item():.2e}")

    return model


if __name__ == "__main__":
    # Setup parameters
    lb, ub = -10, 10
    N_f = 4000
    epochs_per_stage = 5001
    layers = [1, 64, 64, 64, 1]

    # Create grids
    X = np.linspace(lb, ub, N_f).reshape(-1, 1)
    X_test = np.linspace(lb, ub, 1000).reshape(-1, 1)

    # ===== DEFINE TWO-STAGE PARAMETER RANGES =====

    # Stage 1: Vary beta (potential strength) with gamma=0
    beta_stage1 = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]

    # Stage 2: Vary gamma (interaction strength) with beta=100
    gamma_stage2 = [0, 5, 10, 15, 20, 25, 30]

    # Modes to compute
    modes = [0, 1, 2, 3, 4, 5]

    # Training hyperparameters
    perturb_const = 0.01
    tol = 0.00001
    p = 3  # Nonlinearity power
    potential_type = "harmonic"

    print("=" * 70)
    print("TWO-STAGE PERTURBATION LEARNING FOR GROSS-PITAEVSKII EQUATION")
    print("=" * 70)
    print(f"Stage 1: Varying β with γ=0 (Linear)")
    print(f"  β values: {beta_stage1}")
    print(f"Stage 2: Varying γ with β={beta_stage1[-1]} (Nonlinear)")
    print(f"  γ values: {gamma_stage2}")
    print(f"Modes: {modes}")
    print(f"Epochs per step: {epochs_per_stage}")
    print(f"Perturbation constant: {perturb_const}")
    print(f"Tolerance: {tol}")
    print("=" * 70)

    # Create save directory
    save_dir = "two_stage_results"
    os.makedirs(save_dir, exist_ok=True)

    # Train or load
    train_new = True  # Set to False to load existing
    filename = f"two_stage_models_p{p}_{potential_type}.pkl"

    if train_new:
        print("\nStarting two-stage training...")

        (models_by_mode, lambda_table, training_history,
         constant_history, epochs_history) = train_gpe_model_two_stage(
            modes=modes,
            beta_values=beta_stage1,
            gamma_values=gamma_stage2,
            p=p,
            X_train=X,
            lb=lb, ub=ub,
            layers=layers,
            epochs_per_stage=epochs_per_stage,
            tol=tol,
            perturb_const=perturb_const,
            potential_type=potential_type,
            lr=1e-3,
            verbose=True
        )

        # Save results
        save_models_two_stage(
            models_by_mode, lambda_table, training_history,
            constant_history, epochs_history, beta_stage1, gamma_stage2,
            filename, save_dir
        )
    else:
        print("\nLoading existing models...")
        (models_by_mode, lambda_table, training_history,
         constant_history, epochs_history,
         beta_stage1, gamma_stage2) = load_models_two_stage(filename, save_dir)

    # Generate plots
    print("\nGenerating plots...")

    print("  - Wavefunction plots...")
    plot_wavefunction_two_stage(
        models_by_mode, X_test, beta_stage1, gamma_stage2,
        modes, p, constant_history, perturb_const, potential_type, save_dir
    )

    print("  - Eigenvalue plots...")
    plot_eigenvalues_two_stage(
        lambda_table, modes, beta_stage1, gamma_stage2,
        p, potential_type, save_dir
    )

    # Print summary statistics
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)

    for mode in modes:
        print(f"\nMode {mode}:")

        # Stage 1 summary
        print("  Stage 1 (varying β, γ=0):")
        print(f"  {'β':<8} {'λ':<12} {'Epochs':<10}")
        print("  " + "-" * 30)
        stage1_gamma = 0.0
        for beta in beta_stage1:
            param_key = (beta, stage1_gamma)
            if param_key in lambda_table[mode]:
                lambda_val = lambda_table[mode][param_key]
                epochs_val = epochs_history[mode][param_key]
                print(f"  {beta:<8.1f} {lambda_val:<12.4f} {epochs_val:<10d}")

        # Stage 2 summary
        print(f"\n  Stage 2 (varying γ, β={beta_stage1[-1]}):")
        print(f"  {'γ':<8} {'λ':<12} {'Epochs':<10}")
        print("  " + "-" * 30)
        stage2_beta = beta_stage1[-1]
        for gamma in gamma_stage2:
            param_key = (stage2_beta, gamma)
            if param_key in lambda_table[mode]:
                lambda_val = lambda_table[mode][param_key]
                epochs_val = epochs_history[mode][param_key]
                print(f"  {gamma:<8.1f} {lambda_val:<12.4f} {epochs_val:<10d}")

    # Validation against analytical solutions for Stage 1
    print("\n" + "=" * 70)
    print("VALIDATION: Stage 1 (γ=0) vs Analytical Solution")
    print("=" * 70)
    print("\nFor harmonic oscillator with γ=0: λ_n = β(2n + 1)")

    for mode in modes:
        print(f"\nMode {mode}:")
        print(f"  {'β':<8} {'λ_computed':<14} {'λ_analytical':<14} {'Error %':<10}")
        print("  " + "-" * 50)

        stage1_gamma = 0.0
        for beta in beta_stage1:
            param_key = (beta, stage1_gamma)
            if param_key in lambda_table[mode]:
                lambda_computed = lambda_table[mode][param_key]
                # Analytical: λ_n = β(2n + 1)
                lambda_analytical = beta * (2 * mode + 1)
                error_pct = abs(lambda_computed - lambda_analytical) / lambda_analytical * 100

                print(f"  {beta:<8.1f} {lambda_computed:<14.4f} "
                      f"{lambda_analytical:<14.4f} {error_pct:<10.4f}")

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print(f"Results saved to: {save_dir}/")
    print("=" * 70)
    print("\nGenerated files:")
    print(f"  - {filename}")
    print(f"  - mode_*_stage1_beta_variation_p{p}.png")
    print(f"  - mode_*_stage2_gamma_variation_p{p}.png")
    print(f"  - eigenvalues_stage1_p{p}.png")
    print(f"  - eigenvalues_stage2_p{p}.png")