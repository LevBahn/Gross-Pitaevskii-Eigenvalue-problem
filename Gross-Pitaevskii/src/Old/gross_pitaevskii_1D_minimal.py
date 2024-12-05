import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad

# Check if GPU is available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SineActivation(nn.Module):
    """
    Custom sine activation function for the neural network.
    """
    def forward(self, x):
        return torch.sin(x)


class GrossPitaevskiiPINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) for solving the 1D Gross-Pitaevskii Equation..

    Parameters
    ----------
    layers : list of int
        Neural network architecture, each entry defines the number of neurons in that layer.
    hbar : float, optional
        Reduced Planck's constant (default is 1.0).
    m : float, optional
        Mass of the particle (default is 1.0).
    g : float, optional
        Interaction strength (default is 100.0).
    """
    def __init__(self, layers, hbar=1.0, m=1.0, g=100.0):
        super().__init__()
        self.layers = layers
        self.network = self.build_network()
        self.g = g  # Interaction strength
        self.hbar = hbar  # Planck's constant, fixed
        self.m = m  # Particle mass, fixed

    def build_network(self):
        """
        Build the neural network with sine activation functions between layers.

        Returns
        -------
        nn.Sequential
            A PyTorch sequential model representing the neural network architecture.
        """
        layers = []
        for i in range(len(self.layers) - 1):
            layers.append(nn.Linear(self.layers[i], self.layers[i + 1]))
            if i < len(self.layers) - 2:  # Apply activation function for hidden layers
                layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the neural network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor containing spatial points.

        Returns
        -------
        torch.Tensor
            Output tensor representing the predicted solution.
        """
        return self.network(x)

    def compute_potential(self, x, V0=1.0, x0=np.pi / 2, sigma=0.5):
        """
        Compute the Gaussian potential V(x).

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of spatial coordinates x.
        V0 : float, optional
            Amplitude of the potential (default is 1.0).
        x0 : float, optional
            x-coordinate of the Gaussian center (default is pi/2).
        sigma : float, optional
            Standard deviation of the Gaussian (default is 0.5).

        Returns
        -------
        torch.Tensor
            Tensor representing the potential at the input spatial points.
        """
        V = V0 * torch.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
        return V

    def boundary_loss(self, x_bc, u_bc):
        """
        Compute the boundary loss (MSE) for the boundary conditions.

        Parameters
        ----------
        x_bc : torch.Tensor
            Input tensor of boundary spatial points.
        u_bc : torch.Tensor
            Tensor of boundary values (for Dirichlet conditions).

        Returns
        -------
        torch.Tensor
            Mean squared error (MSE) at the boundary points.
        """
        u_pred = self.forward(x_bc)
        return torch.mean((u_pred - u_bc) ** 2)

    def riesz_loss(self, predictions, x):
        """
        Compute the Riesz energy loss for the Gross-Pitaevskii equation.

        Parameters
        ----------
        predictions : torch.Tensor
            Predicted solution from the network.
        inputs : torch.Tensor
            Input tensor of spatial coordinates.

        Returns
        -------
        torch.Tensor
            Riesz energy loss value.
        """
        u = predictions

        if not x.requires_grad:
            x = x.clone().detach().requires_grad_(True)
        u_x = torch.autograd.grad(outputs=predictions, inputs=x,
                                  grad_outputs=torch.ones_like(predictions),
                                  create_graph=True, retain_graph=True)[0]

        laplacian_term = torch.sum(u_x ** 2) # Kinetic term
        V = self.compute_potential(x)
        potential_term = torch.sum(V * u ** 2) # Potential term
        interaction_term = 0.5 * self.g * torch.sum(u ** 4)  # Interaction term

        riesz_energy = 0.5 * (laplacian_term + potential_term + interaction_term)
        return riesz_energy

    def pde_loss(self, x, predictions):
        """
        Compute the PDE loss for the Gross-Pitaevskii equation.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of spatial coordinates.
        predictions : torch.Tensor
            Predicted solution from the network.

        Returns
        -------
        tuple
            Tuple containing:
                - torch.Tensor: PDE loss value.
                - torch.Tensor: PDE residual.
                - torch.Tensor: Smallest eigenvalue (lambda).
        """
        u = predictions

        # Compute first and second derivatives with respect to x
        u_x = grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

        # Compute λ from the energy functional
        V = self.compute_potential(x)
        lambda_pde = torch.mean(u_x ** 2 + V * u ** 2 + self.g * u ** 4) / torch.mean(u ** 2)

        # Residual of the 1D Gross-Pitaevskii equation
        pde_residual = -u_xx + V * u + self.g * torch.abs(u ** 2) * u - lambda_pde * u

        # Regularization: See https://arxiv.org/abs/2010.05075

        # Term 1: L_f = 1 / (f(x, λ))^2, penalizes the network if the PDE residual is close to zero to avoid trivial eigenfunctions
        L_f = 1 / (torch.mean(u ** 2) + 1e-2)

        # Term 2: L_λ = 1 / λ^2, penalizes small eigenvalues λ, ensuring non-trivial eigenvalues
        L_lambda = 1 / (lambda_pde ** 2 + 1e-6)

        # Term 3: L_drive = e^(-λ + c), encourages λ to grow, preventing collapse to small values
        L_drive = torch.exp(-lambda_pde + 1.0)

        # PDE loss (residual plus regularization terms)
        pde_loss = torch.mean(pde_residual ** 2) + L_lambda + L_f
        return pde_loss, pde_residual, lambda_pde

    def total_loss(self, x, x_bc, u_bc):
        """
        Compute the total loss combining boundary, Riesz energy, and PDE losses,
        and print each component to monitor during training.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of spatial coordinates for the interior points.
        x_bc : torch.Tensor
            Input tensor of boundary spatial points.
        u_bc : torch.Tensor
            Tensor of boundary values (for Dirichlet conditions).

        Returns
        -------
        torch.Tensor
            Total loss value.
        """
        data_loss = self.boundary_loss(x_bc, u_bc)
        riesz_energy = self.riesz_loss(self.forward(x), x)
        pde_loss, _, _ = self.pde_loss(x, self.forward(x))

        return data_loss + riesz_energy + pde_loss


def prepare_training_data(N_u, N_f, lb, ub):
    """
    Prepare boundary and collocation points for training.

    Parameters
    ----------
    N_u : int
        Number of boundary points.
    N_f : int
        Number of collocation points.
    lb : np.ndarray
        Lower bounds of the domain.
    ub : np.ndarray
        Upper bounds of the domain.

    Returns
    -------
    collocation_points : np.ndarray
        Collocation points.
    boundary_points : np.ndarray
        Boundary points.
    boundary_values : np.ndarray
        Boundary values.
    """

    # Boundary of interval
    boundary_points = np.array([[lb], [ub]])
    boundary_values = np.zeros((2, 1))

    # Dynamically sample points inside the interval
    collocation_points = np.random.rand(N_f, 1) * (ub - lb) + lb

    return collocation_points, boundary_points, boundary_values


def initialize_weights(m):
    """
    Initialize the weights of the neural network layers using Xavier uniform initialization.

    Parameters
    ----------
    m : torch.nn.Module
        A layer of the neural network. If it is a linear layer, its weights and biases are initialized.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def train_pinn(N_u=500, N_f=10000, layers=[1, 100, 100, 100, 1], epochs=1000):
    """
    Train the Physics-Informed Neural Network (PINN) for the 1D Gross-Pitaevskii equation.
    This version includes dynamic learning rate scheduling and custom weight initialization.

    Parameters
    ----------
    N_u : int, optional
        Number of boundary points (default is 500).
    N_f : int, optional
        Number of collocation points (interior points) for the physics-based loss (default is 10,000).
    layers : list of int, optional
        Architecture of the neural network (default is [1, 100, 100, 100, 1]).
    epochs : int, optional
        Number of training epochs (default is 1000).

    Returns
    -------
    GrossPitaevskiiPINN
        The trained model.
    """
    # Instantiate the PINN model and initialize its weights
    model = GrossPitaevskiiPINN(layers).to(device)
    model.apply(initialize_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5, verbose=True)

    # Generate grid of x-values
    lb = 0
    ub = np.pi
    X = np.linspace(lb, ub, N_f).reshape(-1, 1)

    # Prepare training data (collocation and boundary points)
    collocation_points, boundary_points, boundary_values = prepare_training_data(N_u, N_f, lb, ub)

    # Visualize training data
    visualize_training_data_1D(collocation_points, boundary_points, boundary_values)

    # Visualize the potential
    X_test = np.linspace(lb, ub, N_f).reshape(-1, 1)  # Test points along the 1D interval
    potential = model.compute_potential(torch.tensor(X_test, dtype=torch.float32).to(device)).detach().cpu().numpy()
    plot_potential_1D(X_test, potential)

    # Convert data to PyTorch tensors and move to device
    collocation_points_tensor = torch.tensor(collocation_points, dtype=torch.float32, requires_grad=True).to(device)
    boundary_points_tensor = torch.tensor(boundary_points, dtype=torch.float32).to(device)
    boundary_values_tensor = torch.tensor(boundary_values, dtype=torch.float32).to(device)

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Calculate the total loss (boundary, Riesz energy, and PDE losses)
        loss = model.total_loss(collocation_points_tensor, boundary_points_tensor, boundary_values_tensor)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        # Adjust learning rate if the loss plateaus
        #scheduler.step(loss)

        # Every 400 epochs, print loss and plot the predicted solution
        if epoch % 400 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
            pde_loss, _, lambda_pde = model.pde_loss(collocation_points_tensor, model.forward(collocation_points_tensor))
            plot_solution_1D(model, X_test, epoch=epoch, lambda_pde=lambda_pde.item())

    return model


def plot_solution_1D(model, X_test, epoch=0, lambda_pde=0):
    """
    Plot the predicted solution of the 1D Gross-Pitaevskii equation.

    Parameters
    ----------
    model : GrossPitaevskiiPINN
        The trained PINN model.
    X_test : np.ndarray
        The test points where the predicted solution is computed.
    epoch : int, optional
        The current training epoch, used in the plot title (default is 0).
    lambda_pde : float, optional
        The smallest eigenvalue from the PDE loss, used in the plot title (default is 0).
    """

    # Predict the solution by the trained model
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    u_pred = model(X_test_tensor).detach().cpu().numpy()

    plt.figure(figsize=(8, 6))
    plt.plot(X_test, u_pred / np.max(u_pred), label=f'Epoch: {epoch}, λ: {lambda_pde:.4f}', color='b')
    plt.title(f'Predicted Solution of the 1D Gross-Pitaevskii Equation\nEpoch: {epoch}, λ: {lambda_pde:.4f}')
    plt.xlabel('$x$')
    plt.ylabel('$u_{pred}$ / max($u_{pred}$)')
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_potential_1D(X_test, potential):
    """
    Plot the 1D potential function.

    Parameters
    ----------
    X_test : np.ndarray
        The test points where the potential is computed.
    potential : np.ndarray
        The computed potential values at the test points.
    """
    plt.figure(figsize=(6, 5))

    # X_test is the x-values (positions) of the 1D potential
    plt.plot(X_test, potential, label='Potential $V(x)$', color='green')

    plt.title('Potential $V(x)$ in 1D')
    plt.xlabel('$x$')
    plt.ylabel('$V(x)$')
    plt.grid(True)
    plt.legend()
    plt.show()


def visualize_training_data_1D(collocation_points, boundary_points, boundary_values):
    """
    Visualize the boundary points and collocation points in 1D.

    Parameters
    ----------
    collocation_points : np.ndarray
        Collocation points to visualize.
    boundary_points : np.ndarray
        Boundary points to visualize.
    boundary_values : np.ndarray
        Corresponding boundary condition values.
    """
    plt.figure(figsize=(8, 5))

    # Plot boundary points
    plt.scatter(boundary_points, boundary_values, color='red', label='Boundary Points', alpha=0.6)

    # Plot collocation points
    plt.scatter(collocation_points, np.zeros_like(collocation_points), color='blue', label='Collocation Points', alpha=0.3)

    plt.title('Boundary and Collocation Points in 1D')
    plt.xlabel('$x$')
    plt.ylabel('$u(x)$')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    N_u = 20 # Number of boundary points (only needs to be 2 in this example)
    N_f = 1000 # Number of collocation points
    layers = [1, 200, 200, 200, 1] # Neural network architecture
    epochs = 2001 # Number of training epochs

    # Train the PINN
    model = train_pinn(N_u=N_u, N_f=N_f, layers=layers, epochs=epochs)
