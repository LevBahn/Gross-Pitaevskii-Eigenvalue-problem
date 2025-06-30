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

    def __init__(self, layers, hbar=1.0, m=1.0, mode=0, gamma=1.0, w=0):
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

    def weighted_hermite(self, x: torch.Tensor, n: int) -> torch.Tensor:
        """
        Compute the analytic eigenfunction for a particle in a box.

        For the linear case (gamma = 0), the solution is:
        phi_n(x) = sqrt(2/L) * sin(n*pi*x/L)

        This corresponds to equation (22) in the paper.
        """
        # For mode 0, n=1 in the sine function per equation (22)
        n_actual = n + 1  # Convert mode number to quantum number (n=0 → first excited state with n_actual=1)

        # Normalization factor
        norm_factor = norm_factor = torch.sqrt(torch.tensor(2.0 / 30))

        # Sine function with proper scaling
        phi_n = norm_factor * torch.sin(n_actual * torch.pi * x / 30)  

        return phi_n


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

    def compute_potential(self, x, w, potential_type="harmonic", **kwargs):
        """
        Compute potential function for the 1D domain.
        """
        if potential_type == "harmonic":
            V = w*x
        return V

    def pde_loss(self, inputs, predictions, w, gamma, potential_type="harmonic", precomputed_potential=None):
        """
        Compute the PDE loss for the Gross-Pitaevskii equation.
        μψ = - ∇²ψ + Vψ + γ|ψ|²ψ
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
            V = self.compute_potential(inputs, w, potential_type)

        # Calculate chemical potential
        kinetic = -u_xx
        potential = V * u
        interaction = gamma * u ** 3

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
        full_u = self.get_complete_solution(boundary_points, u_pred)
        return torch.mean((full_u - boundary_values) ** 2)


    def normalization_loss(self, u, dx):
        """
        Compute normalization loss using proper numerical integration.
        """
        integral = torch.sum(u ** 2) * dx
        return (integral - 1.0) ** 2



def train_gpe_model(gamma_values, w_values, modes, X_train, lb, ub, layers, 
                    epochs, tol, perturb_const,
                    potential_type='harmonic', lr=1e-5, verbose=True):
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
    constant_history = {}

    # Sort gamma values
    gamma_values = sorted(gamma_values)

    print(f"Tolerance : {tol}, Perturbation constant : {perturb_const}")    
    gamma = gamma_values[0]
    count=0

    for mode in modes:
        if verbose:
            print(f"\n===== Training for mode {mode} =====")

        mu_logs = []
        models_by_gamma = {}
        history_by_gamma = {}
        
        prev_model = None


        for w in w_values:
            
            if verbose:
                print(f"\nTraining for w = {w:.2f}, mode = {mode}")

            # Initialize model for this mode and gamma
            model = GrossPitaevskiiPINN(layers, mode=mode, gamma=gamma, w=w).to(device)
            # If this isn't the first gamma value, initialize with previous model's weights
            if prev_model is not None:
                model.load_state_dict(prev_model.state_dict())
            else:
                # Use the advanced initialization that considers mode number
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
            
            Loss_criteria = 100

            

            for epoch in range(epochs):
                optimizer.zero_grad()

                # Forward pass
                u_pred = model.forward(X_tensor)
                if epoch == 0 and gamma == 0:
                    normal_const= torch.max(u_pred).detach().clone()
                    constant_history[mode]=normal_const
                    u_pred=u_pred/normal_const
                    u_pred = perturb_const * u_pred
                else: 
                    u_pred = perturb_const* u_pred
                    u_pred = u_pred/normal_const

           

                # Calculate common constraint losses for all modes
                boundary_loss = model.boundary_loss(boundary_points, boundary_values)
                norm_loss = model.normalization_loss(model.get_complete_solution(X_tensor, u_pred), dx)

                # Combined constraint loss
                constraint_loss = 10.0 * boundary_loss + 20.0 * norm_loss

                # Decide which loss to use based on mode
            
                # Use PDE residual for other modes
                pde_loss, lambda_value= model.pde_loss(X_tensor, u_pred, w,  gamma, potential_type)
                physics_loss = pde_loss
                loss_type = "PDE residual"


                # Total loss for optimization
                total_loss = physics_loss + constraint_loss

                # Backpropagate
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
                optimizer.step()
                scheduler.step(total_loss)


                if Loss_criteria > total_loss:
                    Loss_criteria = total_loss
                    models_by_gamma[w] = model
                    count =0
                    
                if Loss_criteria <= total_loss:
                    count = count + 1

                if count == 3000:
                    print(f"Training is stopped as there was no improvement for last {count} epochs with error {Loss_criteria}")
                    break

        
                

                # Record history
                if epoch % 100 == 0:
                    lambda_history.append(lambda_value.item())
                    loss_history.append(total_loss.item())
                    constraint_history.append(constraint_loss)

                    if verbose and epoch % 500 == 0:
                        print(f"Epoch {epoch}, μ: {lambda_value.item():.4f}")

                        print(f"{"Total Loss"} : {total_loss.item():.6f}, {loss_type}: {physics_loss.item():.6f}, "
                             f"Constraints: {constraint_loss:.6f}")
                
                if total_loss <= tol:
                    print(f"Early Stop as we have small enough error : {total_loss}")
                    break
            
            print(f"Train for w:{w} and mode:{mode} is completed with total error: {total_loss}")

            # Record final chemical potential and save model
            final_mu = lambda_history[-1] if lambda_history else 0
            mu_logs.append((gamma, final_mu))
            

            # Save the training history
            history_by_gamma[w] = {
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
        

    return models_by_mode, mu_table, training_history, constant_history


def plot_wavefunction(models_by_mode, X_test, gamma_values, w_values,
                       modes, constant_history, perturb_const,
                         save_dir="Gross-Pitaevskii/src/final/refine/test"):
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
        for j, w in enumerate(w_values):
            if w not in models_by_mode[mode]:
                continue


            model = models_by_mode[mode][w]
            model.eval()
            const = constant_history[mode]

            with torch.no_grad():
                u_pred = model.forward(X_tensor)
                u_pred = u_pred*(perturb_const/const)
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
                        label=f"w={w:.2f}")

        # Configure individual figure
        plt.title(f"Mode {mode} Wavefunction", fontsize=18)
        plt.xlabel("x", fontsize=18)
        plt.ylabel(r"$\psi(x)$", fontsize=18)
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.xlim(0, 30)  # Match paper's range
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"mode_{mode}_wavefunction.png"), dpi=300)
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



if __name__ == "__main__":
    # Setup parameters
    lb, ub = 0, 30  # Domain boundaries
    N_f = 6000  # Number of collocation points
    epochs = 10001  # Increased epochs for better convergence
    layers = [1, 64, 64, 64, 1]  # Neural network architecture

    # Create uniform grid for training and testing
    X = np.linspace(lb, ub, N_f).reshape(-1, 1)
    X_test = np.linspace(lb, ub, 1000).reshape(-1, 1)  # Higher resolution for plotting

    # Ga mma values from the paper
    gamma_values = [k*(0.1) for k in range(1)]
    w_values = [k*(0.01) for k in range(20)]

    # Include modes 0 through 5
    modes = [0]

    # Set the perturbation constant
    perturb_const = 0.001
    
    # Set the tolerance
    tol = 0.00001

    # Train models
    print("Starting training for all modes and gamma values...")
    models_by_mode, mu_table, training_history, constant_history = train_gpe_model(
        gamma_values, w_values, modes, X, lb, ub, layers, epochs, tol, perturb_const, 
        potential_type='harmonic', lr=1e-2, verbose=True )
    print("Training completed!")


    # Plot wavefunctions (not densities) for individual modes
    print("Generating individual mode plots...")
    plot_wavefunction(models_by_mode, X_test, gamma_values, w_values, modes, constant_history, perturb_const)