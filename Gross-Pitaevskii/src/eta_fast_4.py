"""
Improved Physics-Informed Neural Network (PINN) for solving the 1D Gross-Pitaevskii Equation.
This implementation follows Algorithm 2 from the paper at https://arxiv.org/pdf/1208.2123
with enhanced techniques for better accuracy and training stability.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.special import hermite

# Set device for computation (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FourierFeatureTransform(nn.Module):
    """
    Fourier feature transformation for better handling of high-frequency components.

    Transforms input coordinates into a higher dimensional space using random Fourier features,
    which helps neural networks learn high-frequency functions more easily.

    Parameters
    ----------
    input_dim : int
        Dimension of input features.
    mapping_size : int
        Number of Fourier features to create.
    scale : float
        Scaling factor for the random frequency matrix.
    """

    def __init__(self, input_dim, mapping_size, scale=10):
        super().__init__()
        # Create a random frequency matrix (fixed during training)
        self.B = nn.Parameter(torch.randn(input_dim, mapping_size) * scale, requires_grad=False)

    def forward(self, x):
        """
        Transform input using random Fourier features.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, input_dim]

        Returns
        -------
        torch.Tensor
            Transformed tensor with Fourier features of shape [batch_size, 2*mapping_size]
        """
        # Project input to higher dimension
        x_proj = x @ self.B
        # Return concatenated sin and cos of projection
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ResidualBlock(nn.Module):
    """
    Residual block with skip connections to improve gradient flow.

    Parameters
    ----------
    dim : int
        Number of features in the hidden layer.
    activation : callable
        Activation function to use.
    """

    def __init__(self, dim, activation=torch.tanh):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.activation = activation

    def forward(self, x):
        """
        Forward pass through the residual block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output with residual connection: x + f(x)
        """
        # Implement the residual connection
        return x + self.lin2(self.activation(self.lin1(x)))


class SinusoidalLayer(nn.Module):
    """
    Layer with sinusoidal activations to better capture oscillatory behavior.
    Useful for higher mode numbers which exhibit more oscillations.

    Parameters
    ----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)  # For the frequency
        self.linear2 = nn.Linear(in_features, out_features)  # For the linear term

        # Initialize with proper frequency scaling
        with torch.no_grad():
            # Initialize frequency parameters to capture correct oscillation scale
            nn.init.uniform_(self.linear1.weight, -0.5, 0.5)
            self.linear1.bias.data.fill_(0.0)

            # Initialize amplitude parameters with small values
            nn.init.uniform_(self.linear2.weight, -0.1, 0.1)
            self.linear2.bias.data.fill_(0.0)

    def forward(self, x):
        """
        Forward pass through the sinusoidal layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output with sin(linear1(x)) + linear2(x)
        """
        return torch.sin(self.linear1(x)) + self.linear2(x)


class ImprovedGrossPitaevskiiPINN(nn.Module):
    """
    Improved Physics-Informed Neural Network for solving the 1D Gross-Pitaevskii Equation.

    This implementation includes improved techniques for energy minimization,
    proper normalization, and better handling of boundary conditions and symmetries.

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
    use_fourier : bool, optional
        Whether to use Fourier feature embedding (default is True).
    fourier_scale : float, optional
        Scale for Fourier features if used (default is 5.0).
    """

    def __init__(self, layers, hbar=1.0, m=1.0, mode=0, gamma=1.0, use_fourier=True, fourier_scale=5.0):
        super().__init__()
        self.layers = layers
        self.mode = mode  # Mode number (n)
        self.gamma = gamma  # Interaction strength parameter
        self.hbar = hbar  # Planck's constant
        self.m = m  # Particle mass
        self.use_fourier = use_fourier
        self.fourier_scale = fourier_scale

        # Maximum order for Hermite polynomial basis
        self.max_hermite_order = max(10, mode + 3)

        # Build the network with architecture adapted to mode number
        self.network = self.build_network()

        # Track training progress
        self.epoch = 0
        self.best_energy = float('inf')

    def build_network(self):
        """
        Build the neural network with architecture adapted to the mode number.
        For higher modes, we use more complex architectures.

        Returns
        -------
        nn.Sequential
            The constructed neural network
        """
        layers = []

        # Add Fourier feature transform for better handling of high frequencies
        if self.use_fourier and self.mode > 1:
            # More Fourier features for higher modes
            mapping_size = 16 if self.mode <= 3 else 32
            self.fourier_transform = FourierFeatureTransform(
                input_dim=1,
                mapping_size=mapping_size,
                scale=self.fourier_scale * (1.0 + 0.2 * self.mode)
            )
            # Input embedding layer with enlarged dim for Fourier features
            layers.append(nn.Linear(2 * mapping_size, self.layers[1]))
        else:
            # Standard input layer
            layers.append(nn.Linear(1, self.layers[1]))

        # First activation layer
        if self.mode >= 3:
            # Use sinusoidal layer for higher modes to better capture oscillations
            layers.append(SinusoidalLayer(self.layers[1], self.layers[1]))
        else:
            # Use tanh for lower modes
            layers.append(nn.Tanh())

        # Hidden layers with residual blocks for better gradient flow
        for i in range(1, len(self.layers) - 2):
            # Add residual blocks for deeper networks
            layers.append(ResidualBlock(self.layers[i]))
            layers.append(nn.Tanh())

        # Output layer
        layers.append(nn.Linear(self.layers[-2], self.layers[-1]))

        return nn.Sequential(*layers)

    def forward(self, inputs):
        """
        Forward pass through the neural network.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape [batch_size, 1]

        Returns
        -------
        torch.Tensor
            Network output tensor
        """
        # Apply Fourier transform if enabled (for higher modes)
        if self.use_fourier and self.mode > 1:
            inputs = self.fourier_transform(inputs)

        return self.network(inputs)

    def parity_enforced_forward(self, x):
        """
        Enforce correct parity (symmetry/antisymmetry) in the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output with enforced parity
        """
        u_pred = self.forward(x)
        u_reflected = self.forward(-x)

        # For even modes (0,2,4...)
        if self.mode % 2 == 0:
            return (u_pred + u_reflected) / 2  # Force symmetry
        else:
            return (u_pred - u_reflected) / 2  # Force antisymmetry

    def hermite_features(self, x, max_order=None):
        """
        Compute Hermite polynomial features up to max_order for input enrichment.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        max_order : int, optional
            Maximum order of Hermite polynomials

        Returns
        -------
        torch.Tensor
            Tensor with Hermite polynomial features
        """
        if max_order is None:
            max_order = self.max_hermite_order

        features = []
        for n in range(max_order):
            features.append(self.weighted_hermite(x, n).unsqueeze(1))
        return torch.cat(features, dim=1)

    def weighted_hermite(self, x, n):
        """
        Compute the weighted Hermite polynomial solution for the linear case (gamma = 0).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        n : int
            Order of Hermite polynomial

        Returns
        -------
        torch.Tensor
            Weighted Hermite polynomial values
        """
        # Calculate Hermite polynomial values
        H_n = hermite(n)(x.cpu().detach().numpy())

        # Normalization factor
        norm_factor = (2 ** n * math.factorial(n) * np.sqrt(np.pi)) ** (-0.5)

        # Convert to tensor and apply exponential damping
        weighted_hermite = norm_factor * torch.exp(-x ** 2 / 2) * torch.tensor(H_n, dtype=torch.float32).to(device)

        return weighted_hermite

    def get_complete_solution(self, x, perturbation=None, mode=None, enforce_parity=True):
        """
        Get the complete solution by combining the base Hermite solution with the neural network perturbation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        perturbation : torch.Tensor, optional
            Neural network perturbation
        mode : int, optional
            Mode number to use (defaults to self.mode)
        enforce_parity : bool, optional
            Whether to enforce parity in the solution

        Returns
        -------
        torch.Tensor
            Complete solution
        """
        if mode is None:
            mode = self.mode

        if perturbation is None:
            if enforce_parity:
                perturbation = self.parity_enforced_forward(x)
            else:
                perturbation = self.forward(x)

        base_solution = self.weighted_hermite(x, mode)
        return base_solution + perturbation

    def compute_potential(self, x, potential_type="harmonic", **kwargs):
        """
        Compute potential function for the 1D domain.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        potential_type : str, optional
            Type of potential: "harmonic", "gaussian", or "periodic"
        **kwargs : dict
            Additional parameters for the specific potential

        Returns
        -------
        torch.Tensor
            Potential values
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

    def compute_derivatives(self, x, u):
        """
        Compute first and second derivatives of u with respect to x using autograd.

        Parameters
        ----------
        x : torch.Tensor
            Input points, must have requires_grad=True
        u : torch.Tensor
            Function values at input points

        Returns
        -------
        tuple: (u_x, u_xx)
            First and second derivatives
        """
        # Make sure inputs require gradients
        if not x.requires_grad:
            x = x.detach().clone().requires_grad_(True)

        # Compute first derivative
        u_x = torch.autograd.grad(
            outputs=u,
            inputs=x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,  # Essential to avoid backward through graph second time error
        )[0]

        # Compute second derivative
        u_xx = torch.autograd.grad(
            outputs=u_x,
            inputs=x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True,  # Essential to avoid backward through graph second time error
        )[0]

        return u_x, u_xx

    def simpson_integrate(self, f, dx):
        """
        Perform numerical integration using Simpson's rule for better accuracy.

        Parameters
        ----------
        f : torch.Tensor
            Function values at grid points
        dx : float
            Grid spacing

        Returns
        -------
        float
            Integrated value
        """
        n = f.shape[0]
        if n % 2 == 0:  # Even number of points
            f = torch.cat([f, f[-1:]])  # Add one more point
            n += 1

        weights = torch.ones_like(f)
        weights[1:n - 1:2] = 4.0  # Odd indices except the last
        weights[2:n - 2:2] = 2.0  # Even indices except the first and last

        return dx / 3.0 * torch.sum(f * weights)

    def energy_functional(self, x, gamma, potential_type="harmonic", precomputed_potential=None,
                          use_simpson=True, return_components=False):
        """
        Strictly implement the energy functional from Algorithm 2 in the paper:
        E[ψ] = ∫[|∇ψ|²/2 + V|ψ|² + γ|ψ|⁴/2]dx
        """
        # Use parity-enforced forward pass to ensure correct symmetry
        perturbation = self.parity_enforced_forward(x)

        # Get the complete solution
        u = self.get_complete_solution(x, perturbation, enforce_parity=False)

        # Calculate dx for proper integration
        dx = x[1] - x[0]

        # Normalize the wavefunction
        if use_simpson:
            norm_sq = self.simpson_integrate(u ** 2, dx)
        else:
            norm_sq = torch.sum(u ** 2) * dx

        u_normalized = u / torch.sqrt(norm_sq)

        # Compute derivatives - use different approach based on context
        if not torch.is_grad_enabled() or return_components:
            # For evaluation, use numerical differentiation
            with torch.no_grad():
                # Calculate gradient using numpy's gradient function
                u_np = u_normalized.detach().cpu().numpy().flatten()
                dx_val = dx.item()
                u_x_np = np.gradient(u_np, dx_val)
                u_x = torch.tensor(u_x_np, dtype=torch.float32).reshape(-1, 1).to(device)
        else:
            # During training, use autograd with retain_graph
            try:
                u_x = torch.autograd.grad(
                    outputs=u_normalized,
                    inputs=x,
                    grad_outputs=torch.ones_like(u_normalized),
                    create_graph=True,
                    retain_graph=True,
                )[0]
            except RuntimeError:
                # Fallback to numerical differentiation if autograd fails
                with torch.no_grad():
                    u_np = u_normalized.detach().cpu().numpy().flatten()
                    dx_val = dx.item()
                    u_x_np = np.gradient(u_np, dx_val)
                    u_x = torch.tensor(u_x_np, dtype=torch.float32).reshape(-1, 1).to(device)

        # Compute potential
        if precomputed_potential is not None:
            V = precomputed_potential
        else:
            V = self.compute_potential(x, potential_type)

        # Calculate energy components EXACTLY as in the paper
        if use_simpson:
            kinetic_energy = 0.5 * self.simpson_integrate(u_x ** 2, dx)
            potential_energy = self.simpson_integrate(V * u_normalized ** 2, dx)
            interaction_energy = 0.5 * gamma * self.simpson_integrate(u_normalized ** 4, dx)
        else:
            kinetic_energy = 0.5 * torch.sum(u_x ** 2) * dx
            potential_energy = torch.sum(V * u_normalized ** 2) * dx
            interaction_energy = 0.5 * gamma * torch.sum(u_normalized ** 4) * dx

        # Total energy
        total_energy = kinetic_energy + potential_energy + interaction_energy

        if return_components:
            return total_energy, kinetic_energy, potential_energy, interaction_energy, u_normalized
        else:
            return total_energy, u_normalized

    def chemical_potential(self, x, gamma, potential_type="harmonic", precomputed_potential=None, u_normalized=None):
        """
        Calculate chemical potential μ.
        The chemical potential is defined as μ = ∂E/∂N where E is the energy and N is the particle number.
        For a normalized wavefunction, this simplifies to the expectation value of the Hamiltonian.
        """
        # Get normalized wavefunction if not provided
        if u_normalized is None:
            with torch.no_grad():
                perturbation = self.parity_enforced_forward(x)
                u = self.get_complete_solution(x, perturbation, enforce_parity=False)

                # Normalize
                dx = x[1] - x[0]
                norm_sq = torch.sum(u ** 2) * dx
                u_normalized = u / torch.sqrt(norm_sq)

        # Make sure we're using the original x for computing the wavefunction
        # but need fresh tensors for derivatives
        x_fresh = x.detach().clone().requires_grad_(True)

        # Recompute u_normalized using x_fresh to ensure proper gradient connection
        with torch.no_grad():
            perturbation = self.parity_enforced_forward(x_fresh)
            u = self.get_complete_solution(x_fresh, perturbation, enforce_parity=False)
            dx = x_fresh[1] - x_fresh[0]
            norm_sq = torch.sum(u ** 2) * dx

        u_fresh = u / torch.sqrt(norm_sq)
        u_fresh.requires_grad_(True)

        # Compute derivatives with proper graph connection
        try:
            u_x = torch.autograd.grad(
                outputs=u_fresh,
                inputs=x_fresh,
                grad_outputs=torch.ones_like(u_fresh),
                create_graph=True,
                retain_graph=True,
            )[0]

            u_xx = torch.autograd.grad(
                outputs=u_x,
                inputs=x_fresh,
                grad_outputs=torch.ones_like(u_x),
                create_graph=True,
            )[0]
        except RuntimeError as e:
            # If we hit allow_unused error, use a different approach
            if "allow_unused" in str(e):
                # Approximate derivatives using finite differences
                with torch.no_grad():
                    dx_val = x_fresh[1] - x_fresh[0]
                    u_np = u_fresh.detach().cpu().numpy().flatten()

                    # First derivative (central difference)
                    u_x_np = np.gradient(u_np, dx_val.item())

                    # Second derivative
                    u_xx_np = np.gradient(u_x_np, dx_val.item())

                    # Convert back to tensors
                    u_x = torch.tensor(u_x_np, dtype=torch.float32).to(device)
                    u_xx = torch.tensor(u_xx_np, dtype=torch.float32).to(device)
            else:
                raise e

        # Compute potential
        if precomputed_potential is not None:
            V = precomputed_potential
        else:
            V = self.compute_potential(x_fresh, potential_type)

        # Calculate terms in the GPE
        kinetic_term = -0.5 * u_xx
        potential_term = V * u_fresh
        interaction_term = gamma * u_fresh ** 3

        # Calculate μ
        local_mu = (kinetic_term + potential_term + interaction_term) / (u_fresh + 1e-12)
        mu = torch.mean(local_mu)

        return mu

    def boundary_loss(self, boundary_points, boundary_values):
        """
        Compute the boundary loss for enforcing boundary conditions.

        Parameters
        ----------
        boundary_points : torch.Tensor
            Coordinates of boundary points
        boundary_values : torch.Tensor
            Target values at boundary points

        Returns
        -------
        torch.Tensor
            Boundary loss value
        """
        # Ensure boundary points require grad
        if not boundary_points.requires_grad:
            boundary_points = boundary_points.clone().detach().requires_grad_(True)

        u_pred = self.forward(boundary_points)
        full_u = self.get_complete_solution(boundary_points, u_pred)
        return torch.mean((full_u - boundary_values) ** 2)

    def symmetry_loss(self, x):
        """
        Compute the symmetry loss to enforce u(x) = u(-x) for even modes
        and u(x) = -u(-x) for odd modes.

        Parameters
        ----------
        x : torch.Tensor
            Input points

        Returns
        -------
        torch.Tensor
            Symmetry loss value
        """
        # Evaluate u(x) and u(-x)
        u_original = self.forward(x)
        u_reflected = self.forward(-x)

        # For odd modes, apply anti-symmetry condition
        if self.mode % 2 == 1:
            return torch.mean((u_original + u_reflected) ** 2)
        else:
            return torch.mean((u_original - u_reflected) ** 2)

    def normalization_loss(self, u, dx, use_simpson=True):
        """
        Compute normalization loss to ensure ∫|ψ|²dx = 1.

        Parameters
        ----------
        u : torch.Tensor
            Wavefunction values
        dx : float
            Grid spacing
        use_simpson : bool, optional
            Whether to use Simpson's rule for integration

        Returns
        -------
        torch.Tensor
            Normalization loss value
        """
        if use_simpson:
            integral = self.simpson_integrate(u ** 2, dx)
        else:
            integral = torch.sum(u ** 2) * dx

        return (integral - 1.0) ** 2

    def node_constraint_loss(self, x, u):
        """
        Stricter implementation of node constraint loss to enforce
        the correct number of zeros in the wavefunction.
        """
        # Sort x and u by x values to ensure proper order
        sorted_indices = torch.argsort(x.squeeze())
        x_sorted = x.squeeze()[sorted_indices]
        u_sorted = u.squeeze()[sorted_indices]

        # Compute signs and transitions
        signs = torch.sign(u_sorted)
        sign_changes = torch.abs(signs[1:] - signs[:-1])

        # Find indices where sign changes occur (value = 2 when crossing zero)
        transitions = torch.nonzero(sign_changes > 1.0, as_tuple=True)[0]

        # Count zero crossings
        crossings = len(transitions)

        # Calculate difference from expected number of nodes
        error = abs(crossings - self.mode)

        # Exponential penalty that grows much faster with larger differences
        penalty = 10.0 * torch.exp(torch.tensor(error, device=device))

        return penalty

    def frequency_loss(self, u, dx):
        """
        Add constraints in the frequency domain to better capture oscillations.
        Particularly useful for higher modes.

        Parameters
        ----------
        u : torch.Tensor
            Wavefunction values
        dx : float
            Grid spacing

        Returns
        -------
        torch.Tensor
            Frequency domain loss value
        """
        if self.mode < 3:
            return torch.tensor(0.0, device=device)

        # Compute FFT
        ft = torch.fft.rfft(u.squeeze())

        # For mode n, we expect significant components up to frequency n
        # Create expected spectrum (Gaussian centered at mode number)
        freqs = torch.fft.rfftfreq(u.shape[0], dx.item())
        n_freqs = freqs.shape[0]

        # Simple approximation of expected spectrum shape
        expected_peak = min(self.mode, n_freqs - 1)
        expected_spectrum = torch.exp(
            -(torch.arange(n_freqs, device=device) - expected_peak) ** 2 / (self.mode / 2 + 1) ** 2)

        # Normalize both spectra
        ft_norm = torch.abs(ft) / (torch.sum(torch.abs(ft)) + 1e-10)
        expected_spectrum = expected_spectrum / (torch.sum(expected_spectrum) + 1e-10)

        # Compute loss as difference between actual and expected spectra shapes
        # Focus on the first several components which are most important
        max_freq = min(2 * self.mode + 3, n_freqs)
        return torch.mean((ft_norm[:max_freq] - expected_spectrum[:max_freq]) ** 2)

    def smoothness_loss(self, x, u):
        """
        Penalize rapid oscillations that don't match the expected pattern.
        """
        # Compute second derivative
        u_x, u_xx = self.compute_derivatives(x, u)

        # For modes 0-1, heavily penalize any sharp changes
        if self.mode <= 1:
            # Strong smoothness constraint for low modes
            return 0.1 * torch.mean(u_xx ** 2)
        else:
            # For higher modes, enforce smoother oscillations
            # The expected number of oscillations is roughly mode+1
            # So we penalize higher frequency components
            # This is a simple approach - just penalize the magnitude of 2nd derivative
            return 0.05 * torch.mean(u_xx ** 2)

    def find_normalized_solution(self, x):
        """
        Returns a properly normalized solution at the current state of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input points

        Returns
        -------
        torch.Tensor
            Normalized wavefunction
        """
        with torch.no_grad():
            u_pred = self.parity_enforced_forward(x)
            full_u = self.get_complete_solution(x, u_pred)

            # Normalize
            dx = x[1] - x[0]
            norm = torch.sqrt(torch.sum(full_u ** 2) * dx)

            return full_u / norm


def advanced_initialization(m, mode):
    """
    Initialize network weights with consideration of the mode number.
    Higher modes get special initialization to better capture oscillations.

    Parameters
    ----------
    m : nn.Module
        Neural network module to initialize
    mode : int
        Mode number to guide initialization
    """
    if isinstance(m, nn.Linear):
        # Scale gain inversely with mode
        gain = 1.0 / (1.0 + 0.1 * mode)

        # Xavier uniform initialization
        nn.init.xavier_uniform_(m.weight, gain=gain)

        # Initialize biases
        m.bias.data.fill_(0.01)

    elif isinstance(m, SinusoidalLayer):
        # Initialize for the SinusoidalLayer implementation
        # Higher frequencies for higher modes
        scale = 1.0 + 0.2 * mode

        # Initialize linear1 (for frequencies) with higher values
        nn.init.xavier_uniform_(m.linear1.weight, gain=scale)
        m.linear1.bias.data.fill_(0.01)

        # Initialize linear2 (for linear component) normally
        nn.init.xavier_uniform_(m.linear2.weight, gain=0.5)
        m.linear2.bias.data.fill_(0.01)


def initialize_model_for_mode(model, mode, x_tensor):
    """
    Much more precise initialization strategy for different modes.
    Initializes very close to the analytical solution for each mode.
    """
    # Generate analytical solution for this mode at gamma=0
    with torch.no_grad():
        # Get the analytical solution for mode n at gamma=0
        analytical = model.weighted_hermite(x_tensor, mode)

        # Normalize the analytical solution
        dx = x_tensor[1] - x_tensor[0]
        norm = torch.sqrt(torch.sum(analytical ** 2) * dx)
        analytical = analytical / norm

        # For all modes, start extremely close to analytical solution
        # by setting network weights to be very small
        for param in model.parameters():
            param.data *= 0.001

        # For odd modes, ensure antisymmetry
        if mode % 2 == 1:
            # Set biases to zero in all layers to help enforce antisymmetry
            for name, param in model.named_parameters():
                if 'bias' in name:
                    param.data.fill_(0.0)


def train_gpe_model(gamma_values, modes, X_train, lb, ub,
                    base_layers, high_mode_layers, epochs,
                    potential_type='harmonic', lr=1e-3, verbose=True,
                    use_fine_curriculum=True, use_fourier=True):
    """
    Train the GPE model for different modes and gamma values using energy minimization.
    Implements Algorithm 2 from the paper with enhancements for better stability and accuracy.

    Parameters
    ----------
    gamma_values : list of float
        Interaction strength values to train for
    modes : list of int
        Mode numbers to train for
    X_train : numpy.ndarray
        Training points
    lb, ub : float
        Lower and upper bounds of the domain
    base_layers, high_mode_layers : list of int
        Neural network architectures for different mode complexities
    epochs : int
        Number of training epochs
    potential_type : str, optional
        Type of potential to use
    lr : float, optional
        Initial learning rate
    verbose : bool, optional
        Whether to print training progress
    use_fine_curriculum : bool, optional
        Whether to use finer curriculum learning for gamma values
    use_fourier : bool, optional
        Whether to use Fourier feature embedding for higher modes

    Returns
    -------
    tuple
        (models_by_mode, mu_table) containing trained models and chemical potentials
    """
    # Calculate grid spacing
    dx = X_train[1, 0] - X_train[0, 0]  # Assuming uniform grid

    # Create boundary conditions
    boundary_points = torch.tensor([[lb], [ub]], dtype=torch.float32).to(device)
    boundary_values = torch.zeros((2, 1), dtype=torch.float32).to(device)

    # Track models and chemical potentials
    models_by_mode = {}
    mu_table = {}

    # Sort modes and gamma values
    modes = sorted(modes)
    gamma_values = sorted(gamma_values)

    # Define finer curriculum for gamma if enabled
    if use_fine_curriculum:
        # Much finer curriculum with more steps at lower gamma values
        fine_gamma_curriculum = []

        # More detailed curriculum starting from exactly 0
        fine_gamma_curriculum.extend([0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0])

        # Add more intermediate values up to 10
        fine_gamma_curriculum.extend([2.0, 3.0, 5.0, 7.0, 10.0])

        # Add more steps between 10 and 50
        for gamma in gamma_values:
            if gamma > 10.0 and gamma not in fine_gamma_curriculum:
                fine_gamma_curriculum.append(gamma)

        # Sort the curriculum
        fine_gamma_curriculum = sorted(fine_gamma_curriculum)
    else:
        fine_gamma_curriculum = gamma_values

    # We'll train modes in ascending order and use lower mode solutions to initialize higher modes
    for mode in modes:
        if verbose:
            print(f"\n{'=' * 50}")
            print(f"Training for mode {mode}")
            print(f"{'=' * 50}")

        mu_logs = []
        models_by_gamma = {}
        prev_model = None

        # Choose appropriate layer architecture based on mode
        if mode <= 2:
            layers = base_layers
        else:
            layers = high_mode_layers

        # Curriculum learning - train progressively on increasing gamma values
        for gamma in fine_gamma_curriculum:
            if verbose:
                print(f"\nTraining for γ = {gamma:.2f}, mode = {mode}")

            # Skip higher gamma values if they're not in the target set
            # (only use them for curriculum learning)
            save_model = gamma in gamma_values

            # Initialize model for this mode and gamma
            model = ImprovedGrossPitaevskiiPINN(
                layers, mode=mode, gamma=gamma, use_fourier=use_fourier
            ).to(device)

            # If this isn't the first gamma value, initialize with previous model's weights
            if prev_model is not None:
                model.load_state_dict(prev_model.state_dict())
            else:
                # Initialize fresh model according to mode
                X_tensor = torch.tensor(X_train, dtype=torch.float32, requires_grad=True).to(device)
                initialize_model_for_mode(model, mode, X_tensor)

            # Create optimizer with weight decay
            # Use a small weight decay to prevent overfitting
            weight_decay = 1e-5 * (1 + 0.1 * mode)  # More regularization for higher modes
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            # Create learning rate scheduler for better convergence
            # These parameters work well for this problem
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=200, verbose=verbose,
                min_lr=1e-6
            )

            # Track best model and its energy
            best_energy = float('inf')
            best_state = None

            # Extended training for γ=0 to get accurate base solutions
            if gamma == 0.0:
                current_epochs = epochs * 3  # Triple the epochs for gamma=0
            else:
                current_epochs = epochs

            # Train the model
            for epoch in range(current_epochs):
                # Create tensor with requires_grad=True for each epoch
                X_tensor = torch.tensor(X_train, dtype=torch.float32, requires_grad=True).to(device)

                # Precompute potential for efficiency
                V = None
                if potential_type == "harmonic":
                    V = 0.5 * X_tensor ** 2

                # Model training step
                optimizer.zero_grad()

                # Compute energy using the energy functional
                # For all modes, we use the energy minimization approach (Algorithm 2)
                energy, u_normalized = model.energy_functional(X_tensor, gamma, potential_type, V)

                # Base loss is the energy to minimize
                loss = energy

                # Add strong node constraints for all modes
                if mode > 0:
                    # Higher weight for node constraints
                    node_weight = 5.0 * (1 + 0.5 * mode)  # Scale with mode number
                    node_loss = model.node_constraint_loss(X_tensor, u_normalized)
                    loss = loss + node_weight * node_loss

                # Add stronger symmetry enforcement - especially important for even modes
                if mode % 2 == 0:
                    # Much stronger symmetry enforcement for even modes
                    sym_weight = 3.0
                else:
                    # Still enforce antisymmetry for odd modes
                    sym_weight = 1.0

                sym_loss = model.symmetry_loss(X_tensor)
                loss = loss + sym_weight * sym_loss

                # Frequency domain constraints for higher modes
                if mode >= 2:
                    freq_weight = 0.5
                    freq_loss = model.frequency_loss(u_normalized, dx)
                    loss = loss + freq_weight * freq_loss

                # Boundary conditions
                boundary_weight = 1.0
                boundary_loss = model.boundary_loss(boundary_points, boundary_values)
                loss = loss + boundary_weight * boundary_loss

                # Extra normalization enforcement
                norm_weight = 1.0
                norm_loss = model.normalization_loss(u_normalized, dx)
                loss = loss + norm_weight * norm_loss

                # Add smoothness constraint
                smooth_weight = 0.5 if mode <= 2 else 0.2
                smooth_loss = model.smoothness_loss(X_tensor, u_normalized)
                loss = loss + smooth_weight * smooth_loss

                # Backpropagate
                loss.backward()

                # Stronger gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

                # Update weights
                optimizer.step()

                # Update learning rate
                scheduler.step(loss)

                # Track best model
                if energy.item() < best_energy:
                    best_energy = energy.item()
                    best_state = model.state_dict().copy()
                    model.best_energy = best_energy

                # Print progress
                if verbose and epoch % 100 == 0:
                    # For monitoring only, use a simpler calculation of chemical potential
                    with torch.no_grad():
                        # Get normalized wavefunction
                        X_eval = torch.tensor(X_train, dtype=torch.float32).to(device)
                        u_eval_pred = model.parity_enforced_forward(X_eval)
                        u_eval = model.get_complete_solution(X_eval, u_eval_pred, enforce_parity=False)

                        # Normalize
                        dx_eval = X_eval[1] - X_eval[0]
                        norm_eval = torch.sqrt(torch.sum(u_eval ** 2) * dx_eval)
                        u_eval_normalized = u_eval / norm_eval

                        # Calculate derivatives using finite differences
                        u_np = u_eval_normalized.cpu().numpy().flatten()
                        dx_val = dx_eval.item()

                        # Calculate first derivative
                        u_x_np = np.gradient(u_np, dx_val)

                        # Calculate second derivative
                        u_xx_np = np.gradient(u_x_np, dx_val)

                        # Calculate potential
                        V_np = 0.5 * X_eval.cpu().numpy().flatten() ** 2  # Harmonic potential

                        # Calculate terms in GPE
                        kinetic_np = -0.5 * u_xx_np
                        potential_np = V_np * u_np
                        interaction_np = gamma * u_np ** 3

                        # Calculate μ
                        local_mu_np = (kinetic_np + potential_np + interaction_np) / (u_np + 1e-12)
                        mu_val = np.mean(local_mu_np)

                    # Print status
                    print(f"Epoch {epoch}, Energy: {energy.item():.6f}, μ: {mu_val:.6f}")

                    # For higher modes, print constraint values
                    if mode > 0 and epoch % 500 == 0:
                        print(f"  Node constraint: {node_loss.item():.6f}, Sym: {sym_loss.item():.6f}, "
                              f"Smooth: {smooth_loss.item():.6f}")

                # Early stopping if energy is very low and stable
                if epoch > 1000 and energy.item() < 1e-6:
                    if verbose:
                        print(f"Converged early at epoch {epoch}")
                    break

            # Load best model
            if best_state is not None:
                model.load_state_dict(best_state)

            # Compute final energy and chemical potential using numerical methods
            with torch.no_grad():
                X_final = torch.tensor(X_train, dtype=torch.float32).to(device)

                # Get normalized wavefunction and energy components
                final_energy, kinetic, potential, interaction, u_normalized = model.energy_functional(
                    X_final, gamma, potential_type, return_components=True
                )

                # Calculate chemical potential using numerical methods
                u_np = u_normalized.cpu().numpy().flatten()
                x_np = X_final.cpu().numpy().flatten()
                dx_val = X_final[1, 0].item() - X_final[0, 0].item()

                # First derivative
                u_x_np = np.gradient(u_np, dx_val)

                # Second derivative
                u_xx_np = np.gradient(u_x_np, dx_val)

                # Potential
                V_np = 0.5 * x_np ** 2  # Harmonic potential

                # Calculate terms in GPE
                kinetic_np = -0.5 * u_xx_np
                potential_np = V_np * u_np
                interaction_np = gamma * u_np ** 3

                # Calculate μ
                local_mu_np = (kinetic_np + potential_np + interaction_np) / (u_np + 1e-12)
                mu_final = np.mean(local_mu_np)

            if verbose:
                print(f"Final Energy: {final_energy.item():.6f}, μ: {mu_final:.6f}")
                print(f"Energy components - Kinetic: {kinetic.item():.6f}, Potential: {potential.item():.6f}, "
                      f"Interaction: {interaction.item():.6f}")

            # Record results for target gamma values
            if save_model:
                mu_logs.append((gamma, mu_final))
                models_by_gamma[gamma] = model

            # Update prev_model for next gamma value
            prev_model = model

        # Store results for this mode
        mu_table[mode] = mu_logs
        models_by_mode[mode] = models_by_gamma

    return models_by_mode, mu_table


def plot_wavefunction_densities(models_by_mode, X_test, gamma_values, modes, save_dir="plots"):
    """
    Plot wavefunction densities for different modes and gamma values.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Generate analytical solutions for comparison
    analytical_solutions = {}
    for mode in modes:
        model = list(models_by_mode[mode].values())[0]  # Get any model for this mode

        with torch.no_grad():
            X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            dx = X_test[1, 0] - X_test[0, 0]

            # Get analytical solution
            analytical = model.weighted_hermite(X_tensor, mode).cpu().numpy().flatten()
            norm = np.sqrt(np.sum(analytical ** 2) * dx)
            analytical_solutions[mode] = analytical / norm

    # Generate individual figures for each mode
    for mode in modes:
        if mode not in models_by_mode:
            continue

        # Create individual figure
        plt.figure(figsize=(10, 7))

        # Convert to tensor but no need for gradients in evaluation
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
                # Use parity enforced forward for cleaner results
                u_pred = model.parity_enforced_forward(X_tensor)
                full_u = model.get_complete_solution(X_tensor, u_pred, enforce_parity=False)
                u_np = full_u.cpu().numpy().flatten()

                # Normalization
                u_np /= np.sqrt(np.sum(u_np ** 2) * dx)

                # Plot wavefunction density
                plt.plot(X_test.flatten(), u_np ** 2,
                         linestyle=linestyles[j % len(linestyles)],
                         color=colors[j % len(colors)],
                         linewidth=2,
                         label=f"γ={gamma:.1f}")

        # Also plot the analytical solution for γ=0 as a reference
        if mode <= 3:  # Only for lower modes to avoid cluttering
            plt.plot(X_test.flatten(), analytical_solutions[mode] ** 2,
                     linestyle='--',
                     color='gray',
                     linewidth=1,
                     label=f"Analytical γ=0")

        # Configure individual figure
        plt.title(f"Mode {mode} Wavefunction Density", fontsize=16)
        plt.xlabel("x", fontsize=14)
        plt.ylabel("|ψ(x)|²", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.xlim(-8, 8)  # Focused range
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"mode_{mode}_density.png"), dpi=300)
        plt.close()

    # Also create a combined grid figure to show all modes
    plot_combined_grid(models_by_mode, X_test, gamma_values, modes, save_dir)


def plot_combined_grid(models_by_mode, X_test, gamma_values, modes, save_dir="plots"):
    """
    Create a grid of subplots showing all modes.
    """
    # Determine grid dimensions
    n_modes = len(modes)
    n_cols = min(4, n_modes)  # Max 4 columns
    n_rows = (n_modes + n_cols - 1) // n_cols  # Ceiling division

    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

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
                # Use parity enforced forward for cleaner results
                u_pred = model.parity_enforced_forward(X_tensor)
                full_u = model.get_complete_solution(X_tensor, u_pred, enforce_parity=False)
                u_np = full_u.cpu().numpy().flatten()

                # Proper normalization
                u_np /= np.sqrt(np.sum(u_np ** 2) * dx)

                # Plot on the appropriate subplot
                ax.plot(X_test.flatten(), u_np ** 2,
                        linestyle=linestyles[j % len(linestyles)],
                        color=colors[j % len(colors)],
                        label=f"γ={gamma:.1f}")

        # Configure the subplot
        ax.set_title(f"Mode {mode}", fontsize=12)
        ax.set_xlabel("x", fontsize=10)
        ax.set_ylabel("|ψ(x)|²", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_xlim(-8, 8)  # Focused range

    # Hide any unused subplots
    for i in range(len(modes), len(axes)):
        axes[i].axis('off')

    # Finalize and save combined figure
    plt.suptitle("Wavefunction Densities for All Modes", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(save_dir, "all_modes_combined.png"), dpi=300)
    plt.close(fig)


def plot_mu_vs_gamma(mu_table, modes, save_dir="plots"):
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
                 linewidth=2,
                 label=f"Mode {mode}")

    plt.xlabel("γ (Interaction Strength)", fontsize=14)
    plt.ylabel("μ (Chemical Potential)", fontsize=14)
    plt.title("Chemical Potential vs. Interaction Strength for All Modes", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "mu_vs_gamma_all_modes.png"), dpi=300)
    plt.close()


def plot_individual_wavefunctions(models_by_mode, X_test, gamma_values, modes, save_dir="plots"):
    """
    Plot individual wavefunction shapes (not densities) for comparison.
    """
    os.makedirs(save_dir, exist_ok=True)

    for mode in modes:
        if mode not in models_by_mode:
            continue

        plt.figure(figsize=(10, 6))

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
                # Use parity enforced forward for cleaner results
                u_pred = model.parity_enforced_forward(X_tensor)
                full_u = model.get_complete_solution(X_tensor, u_pred, enforce_parity=False)
                u_np = full_u.cpu().numpy().flatten()

                # Normalization
                u_np /= np.sqrt(np.sum(u_np ** 2) * dx)

                # Plot wavefunction (not density)
                plt.plot(X_test.flatten(), u_np,
                         linestyle=linestyles[j % len(linestyles)],
                         color=colors[j % len(colors)],
                         linewidth=2,
                         label=f"γ={gamma:.1f}")

        # Configure plot
        plt.title(f"Mode {mode} Wavefunction", fontsize=16)
        plt.xlabel("x", fontsize=14)
        plt.ylabel("ψ(x)", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.xlim(-8, 8)  # Focused range
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"mode_{mode}_wavefunction.png"), dpi=300)
        plt.close()


def compare_with_analytical(models_by_mode, X_test, save_dir="plots"):
    """
    Compare neural network solutions with analytical solutions for gamma=0 case.
    """
    os.makedirs(save_dir, exist_ok=True)

    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    x_np = X_test.flatten()
    dx = X_test[1, 0] - X_test[0, 0]  # Spatial step size

    # For each mode, compare the gamma=0 case with analytical
    for mode in range(5):  # Only showing first 5 modes
        if mode not in models_by_mode or 0.0 not in models_by_mode[mode]:
            continue

        plt.figure(figsize=(10, 6))

        # Get the model prediction
        model = models_by_mode[mode][0.0]
        model.eval()

        with torch.no_grad():
            u_pred = model.parity_enforced_forward(X_tensor)
            full_u = model.get_complete_solution(X_tensor, u_pred, enforce_parity=False)
            u_nn = full_u.cpu().numpy().flatten()
            u_nn /= np.sqrt(np.sum(u_nn ** 2) * dx)

            # Get analytical solution (using model's weighted_hermite function)
            analytical_u = model.weighted_hermite(X_tensor, mode).cpu().numpy().flatten()
            analytical_u /= np.sqrt(np.sum(analytical_u ** 2) * dx)

            # Plot both solutions
            plt.plot(x_np, u_nn ** 2, 'b-', linewidth=2, label='PINN Solution')
            plt.plot(x_np, analytical_u ** 2, 'r--', linewidth=2, label='Analytical Solution')

            # Calculate and display error
            error = np.mean((u_nn - analytical_u) ** 2)
            plt.title(f"Mode {mode}, γ=0: PINN vs Analytical (MSE={error:.2e})", fontsize=16)

        plt.xlabel("x", fontsize=14)
        plt.ylabel("|ψ(x)|²", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.xlim(-8, 8)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"mode_{mode}_gamma0_comparison.png"), dpi=300)
        plt.close()


def plot_energy_components(models_by_mode, X_test, gamma_values, modes, save_dir="plots"):
    """
    Plot energy components (kinetic, potential, interaction) for each mode and gamma.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Spatial step size
    dx = X_test[1, 0] - X_test[0, 0]

    # Initialize dictionary to store energy data
    energy_data = {mode: {} for mode in modes}

    for mode in modes:
        if mode not in models_by_mode:
            continue

        for gamma in gamma_values:
            if gamma not in models_by_mode[mode]:
                continue

            model = models_by_mode[mode][gamma]
            model.eval()

            try:
                # Calculate energy components using the model's energy_functional method
                X_tensor = torch.tensor(X_test, dtype=torch.float32, requires_grad=True).to(device)

                with torch.no_grad():
                    # Get energy components
                    total_energy, kinetic_energy, potential_energy, interaction_energy, _ = model.energy_functional(
                        X_tensor, gamma, potential_type="harmonic", return_components=True
                    )

                    # Store energy data
                    energy_data[mode][gamma] = {
                        'kinetic': kinetic_energy.item(),
                        'potential': potential_energy.item(),
                        'interaction': interaction_energy.item(),
                        'total': total_energy.item()
                    }

            except Exception as e:
                print(f"Error calculating energy for mode {mode}, gamma {gamma}: {e}")
                continue

    # Plot total energy vs gamma for each mode
    plt.figure(figsize=(10, 8))
    markers = ['o', 's', '^', 'v', 'D', 'x', '*', '+']
    colors = ['k', 'b', 'r', 'g', 'm', 'c', 'orange', 'purple']

    for i, mode in enumerate(modes):
        if mode not in energy_data or not energy_data[mode]:
            continue

        gammas = sorted(energy_data[mode].keys())
        if not gammas:
            continue

        total_energies = [energy_data[mode][g]['total'] for g in gammas]

        plt.plot(gammas, total_energies,
                 marker=markers[i % len(markers)],
                 color=colors[i % len(colors)],
                 linestyle='-',
                 linewidth=2,
                 label=f"Mode {mode}")

    plt.xlabel("γ (Interaction Strength)", fontsize=14)
    plt.ylabel("Total Energy", fontsize=14)
    plt.title("Total Energy vs. Interaction Strength for All Modes", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "total_energy_vs_gamma.png"), dpi=300)
    plt.close()

    # For each mode, plot stacked energy components
    for mode in modes:
        if mode not in energy_data or not energy_data[mode]:
            continue

        gammas = sorted(energy_data[mode].keys())
        if not gammas:
            continue

        kinetic = [energy_data[mode][g]['kinetic'] for g in gammas]
        potential = [energy_data[mode][g]['potential'] for g in gammas]
        interaction = [energy_data[mode][g]['interaction'] for g in gammas]

        plt.figure(figsize=(12, 8))
        plt.bar(gammas, kinetic, label='Kinetic', alpha=0.7, color='skyblue')
        plt.bar(gammas, potential, bottom=kinetic, label='Potential', alpha=0.7, color='salmon')
        plt.bar(gammas, interaction, bottom=[k + p for k, p in zip(kinetic, potential)],
                label='Interaction', alpha=0.7, color='lightgreen')

        plt.xlabel("γ (Interaction Strength)", fontsize=14)
        plt.ylabel("Energy Components", fontsize=14)
        plt.title(f"Energy Components vs. Interaction Strength for Mode {mode}", fontsize=16)
        plt.grid(True, axis='y', alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"mode_{mode}_energy_components.png"), dpi=300)
        plt.close()


def save_results(models_by_mode, mu_table, save_dir="results"):
    """
    Save models and chemical potential values to files.

    Parameters
    ----------
    models_by_mode : dict
        Dictionary of trained models indexed by mode and gamma
    mu_table : dict
        Dictionary of chemical potential values indexed by mode and gamma
    save_dir : str, optional
        Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save chemical potential data
    mu_data_file = os.path.join(save_dir, "chemical_potentials.txt")
    with open(mu_data_file, 'w') as f:
        f.write("Mode\tGamma\tChemical Potential\n")
        for mode in sorted(mu_table.keys()):
            for gamma, mu in mu_table[mode]:
                f.write(f"{mode}\t{gamma:.1f}\t{mu:.6f}\n")

    # Save models
    for mode in models_by_mode:
        mode_dir = os.path.join(save_dir, f"mode_{mode}")
        os.makedirs(mode_dir, exist_ok=True)

        for gamma, model in models_by_mode[mode].items():
            model_file = os.path.join(mode_dir, f"gamma_{gamma:.1f}.pt")
            torch.save(model.state_dict(), model_file)


def validate_results(models_by_mode, X_test, gamma_values, modes, save_dir="validation"):
    """
    Perform additional validation checks on the trained models to ensure they satisfy
    key physical properties of the Gross-Pitaevskii equation solutions.

    Parameters
    ----------
    models_by_mode : dict
        Dictionary of trained models indexed by mode and gamma
    X_test : numpy.ndarray
        Test points for evaluation
    gamma_values : list of float
        Interaction strength values to validate
    modes : list of int
        Mode numbers to validate
    save_dir : str, optional
        Directory to save validation results
    """
    os.makedirs(save_dir, exist_ok=True)

    # Convert test points to tensor
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    dx = X_test[1, 0] - X_test[0, 0]  # Spatial step size

    # Create validation summary file
    summary_file = os.path.join(save_dir, "validation_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("Validation Summary for Gross-Pitaevskii Solutions\n")
        f.write("=" * 50 + "\n\n")

        for mode in modes:
            if mode not in models_by_mode:
                continue

            f.write(f"Mode {mode}:\n")
            f.write("-" * 20 + "\n")

            for gamma in gamma_values:
                if gamma not in models_by_mode[mode]:
                    continue

                model = models_by_mode[mode][gamma]
                model.eval()

                with torch.no_grad():
                    # Get normalized wavefunction
                    u_pred = model.parity_enforced_forward(X_tensor)
                    full_u = model.get_complete_solution(X_tensor, u_pred, enforce_parity=False)
                    u_np = full_u.cpu().numpy().flatten()

                    # Normalize
                    norm = np.sqrt(np.sum(u_np ** 2) * dx)
                    u_np /= norm

                    # Check for normalization
                    actual_norm = np.sqrt(np.sum(u_np ** 2) * dx)
                    norm_error = abs(actual_norm - 1.0)

                    # Check for correct number of nodes (zeros)
                    x_np = X_test.flatten()
                    zeros = np.where(np.diff(np.signbit(u_np)))[0]
                    num_zeros = len(zeros)

                    # Compute numerical derivatives
                    u_x = np.gradient(u_np, dx)
                    u_xx = np.gradient(u_x, dx)

                    # Compute potential
                    V = 0.5 * x_np ** 2

                    # Compute terms in GPE
                    kinetic = -0.5 * u_xx
                    potential = V * u_np
                    interaction = gamma * u_np ** 3

                    # Calculate chemical potential
                    mu_local = (kinetic + potential + interaction) / (u_np + 1e-10)
                    mu = np.mean(mu_local)
                    mu_std = np.std(mu_local)  # Standard deviation - should be small for a good solution

                    # Write validation results
                    f.write(f"  γ={gamma:.1f}:\n")
                    f.write(f"    Normalization error: {norm_error:.2e}\n")
                    f.write(f"    Number of nodes: {num_zeros} (expected: {mode})\n")
                    f.write(f"    Chemical potential: {mu:.6f} (std: {mu_std:.6f})\n")
                    f.write(f"    Energy components - Kinetic: {np.sum(0.5 * u_x ** 2) * dx:.6f}, "
                            f"Potential: {np.sum(V * u_np ** 2) * dx:.6f}, "
                            f"Interaction: {0.5 * gamma * np.sum(u_np ** 4) * dx:.6f}\n")
                    f.write("\n")

            f.write("\n")


if __name__ == "__main__":
    print("Starting Gross-Pitaevskii equation solver using PINNs")

    # Setup parameters
    lb, ub = -10, 10  # Domain boundaries
    N_f = 4000  # Number of collocation points (training points)
    N_test = 1000  # Number of test points for evaluation
    epochs = 3000  # Number of training epochs

    # Define network architectures for different mode complexities
    base_layers = [1, 64, 64, 64, 1]  # For modes 0-2 (simpler)
    high_mode_layers = [1, 128, 128, 128, 1]  # For modes 3+ (more complex)

    # Create uniform grid for training and testing
    # Training grid (coarser)
    X = np.linspace(lb, ub, N_f).reshape(-1, 1)
    # Testing grid (finer for better visualization)
    X_test = np.linspace(lb, ub, N_test).reshape(-1, 1)

    # Gamma values from the paper
    gamma_values = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]

    # Include modes 0 through 7
    modes = [0, 1, 2, 3, 4, 5, 6, 7]

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create directories for results
    results_dir = "gpe_results"
    plots_dir = os.path.join(results_dir, "plots")
    models_dir = os.path.join(results_dir, "models")
    validation_dir = os.path.join(results_dir, "validation")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)

    # Train models with improved training procedure
    print("\nStarting model training with energy minimization approach (Algorithm 2)...")
    models_by_mode, mu_table = train_gpe_model(
        gamma_values=gamma_values,
        modes=modes,
        X_train=X,
        lb=lb, ub=ub,
        base_layers=base_layers,
        high_mode_layers=high_mode_layers,
        epochs=epochs,
        potential_type='harmonic',
        lr=1e-3,
        verbose=True,
        use_fine_curriculum=True,
        use_fourier=True
    )
    print("Training completed!")

    # Save models and chemical potential data
    print("\nSaving models and chemical potential data...")
    save_results(models_by_mode, mu_table, models_dir)

    # Validate results against physical properties
    print("\nValidating results against physical properties...")
    validate_results(models_by_mode, X_test, gamma_values, modes, validation_dir)

    # Plot wavefunctions and energy data
    print("\nGenerating plots...")

    # Plot wavefunction densities
    print("- Plotting wavefunction densities...")
    plot_wavefunction_densities(models_by_mode, X_test, gamma_values, modes, plots_dir)

    # Plot chemical potential vs gamma
    print("- Plotting chemical potential vs. gamma...")
    plot_mu_vs_gamma(mu_table, modes, plots_dir)

    # Plot individual wavefunctions (not just densities)
    print("- Plotting individual wavefunctions...")
    plot_individual_wavefunctions(models_by_mode, X_test, gamma_values, modes, plots_dir)

    # Compare with analytical solutions for gamma=0
    print("- Comparing with analytical solutions for γ=0...")
    compare_with_analytical(models_by_mode, X_test, plots_dir)

    # Plot energy components
    print("- Analyzing and plotting energy components...")
    plot_energy_components(models_by_mode, X_test, gamma_values, modes, plots_dir)

    print("\nAll tasks completed successfully!")
    print(f"Results are saved in the '{results_dir}' directory.")