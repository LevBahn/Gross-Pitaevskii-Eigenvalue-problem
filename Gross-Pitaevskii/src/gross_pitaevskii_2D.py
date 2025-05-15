import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import numpy as np
from pyDOE import lhs
import matplotlib.pyplot as plt

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import matplotlib
matplotlib.use('TkAgg')


class GrossPitaevskiiPINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) for solving the 2D Gross-Pitaevskii equation (GPE).
    This PINN learns the wave function 'u' and minimizes the energy functional.

    Attributes
    ----------
    layers : nn.ModuleList
        List of neural network layers.
    hbar : torch.nn.Parameter
        Planck's constant, scaled to 1.0.
    m : torch.nn.Parameter
        Particle mass, scaled to 1.0.
    g : torch.nn.Parameter
        Interaction strength, currently set to 100.
    """

    def __init__(self, layers, hbar=1.0, m=1.0, g=100.0):
        """
        Initializes the PINN model with given layer sizes and boundary conditions.

        Parameters
        ----------
        layers : list
            List of integers specifying the number of units in each layer.
        hbar : float, optional
            Planck's constant, in J⋅Hz^{−1}, scaled to 1.0.
        m : float
            Particle mass, scaled to 1.0.
        g : float
            Interaction strength, set as 100.
        """
        super().__init__()

        # Network
        self.layers = layers
        self.network = self.build_network()

        # Physics parameters
        self.hbar = hbar  # Planck's constant, fixed
        self.m = m  # Particle mass, fixed
        self.g = g  # Interaction strength

    def build_network(self):
        layers = []
        for i in range(len(self.layers) - 1):
            layers.append(nn.Linear(self.layers[i], self.layers[i + 1]))
            if i < len(self.layers) - 2:  # Apply activation for hidden layers
                layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network. Applies input scaling and passes through layers.

        Parameters
        ----------
        x : torch.Tensor
            Input spatial coordinates.

        Returns
        -------
        torch.Tensor
            Output spatial coordinates.
        """
        return self.network(x)

    def boundary_loss(self, x_bc, y_bc):
        """
        Computes the data loss (MSE) at the boundary.

        Parameters
        ----------
        x_bc : torch.Tensor
            Boundary condition input data.
        y_bc : torch.Tensor
            Boundary condition output (true) values.

        Returns
        -------
        torch.Tensor
            Scaled mean squared error (MSE) loss for boundary conditions.
        """

        # Predict values for boundary points
        u_pred = self.forward(x_bc)

        # Ensure boundary conditions are enforced by making y_bc a zero tensor
        y_bc = torch.zeros_like(u_pred)

        # Adaptive scaling for boundary loss
        bc_loss = torch.mean((u_pred - y_bc) ** 2) * 10

        return bc_loss


    def riesz_loss(self, predictions, inputs):
        """
        Computes the Riesz energy loss for the 2D Gross-Pitaevskii equation.

        Energy functional: E(u) = (1/2) ∫_Ω |∇u|² + V(x)|u|² + (η/2) |u|⁴ dx

        where:
        - u is the predicted solution,
        - V(x) is the potential function,
        - η is a constant parameter,
        - ∇u represents the gradient of u.

        Parameters
        ----------
        predictions : torch.Tensor
            Model predictions.
        inputs : torch.Tensor
            Input points.

        Returns
        -------
        torch.Tensor
            Riesz energy loss.
        """
        u = predictions

        if not inputs.requires_grad:
            inputs = inputs.clone().detach().requires_grad_(True)
        gradients = torch.autograd.grad(outputs=predictions, inputs=inputs,
                                        grad_outputs=torch.ones_like(predictions),
                                        create_graph=True, retain_graph=True)[0]

        laplacian_term = torch.sum(gradients ** 2)  # Kinetic term
        V = self.compute_potential(inputs).unsqueeze(1)
        potential_term = torch.sum(V * u ** 2)  # Potential term
        interaction_term = 0.5 * self.g * torch.sum(u ** 4)  # Interaction term

        riesz_energy = 0.5 * (laplacian_term + potential_term + interaction_term)

        return riesz_energy


    def pde_loss(self, inputs, predictions):
        """
        Computes the PDE loss for the 2D Gross-Pitaevskii equation:

        -∇²u + V(x)u + η|u|²u - λu = 0

        The loss is based on the residual of the equation after solving for `u` and computing
        the smallest eigenvalue `λ` using the energy functional.

        Parameters
        ----------
        inputs : torch.Tensor
            Input points (x, y).
        predictions : torch.Tensor
            Predicted output from the network, representing the wave function ψ.

        Returns
        -------
        pde_loss: torch.Tensor
            Constant representing the Gross-Pitaevskii PDE loss.
        pde_residual: torch.Tensor
            Tensor representing the residual of the Gross-Pitaevskii PDE.
        lambda_pde: torch.Tensor
            Constant representing the smallest eigenvalue of the Gross-Pitaevskii PDE.
        """

        u = predictions

        # Compute first and second derivatives with respect to x and y
        u_x = grad(u, inputs, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 0]
        u_y = grad(u, inputs, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 1]

        u_xx = grad(u_x, inputs, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0]
        u_yy = grad(u_y, inputs, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1]
        laplacian_u = u_xx + u_yy

        # Compute λ directly from the energy functional
        V = self.compute_potential(inputs)
        lambda_pde = torch.mean(u_x ** 2 + u_y ** 2 + V * u ** 2 + self.g * u ** 4) / torch.mean(u ** 2)

        # Residual of the PDE (Gross-Pitaevskii equation)
        pde_residual = -laplacian_u + V * u + self.g * torch.abs(u ** 2) * u - (lambda_pde * u)

        # Regularization: See https://arxiv.org/abs/2010.05075

        # Regularization term 1: L_f = 1 / (f(x, λ))^2, penalizes the network if the PDE residual is close to zero to
        # avoid trivial eigenfunctions
        L_f = 1 / (torch.mean(u ** 2) + 1e-2)  # Add small constant to avoid division by zero

        # Regularization term 2: L_λ = 1 / λ^2, penalizes small eigenvalues λ, ensuring non-trivial eigenvalues
        L_lambda = 1 / (lambda_pde ** 2 + 1e-6)

        # Regularization term 3: L_drive = e^(-λ + c), encourages λ to grow, preventing collapse to small values
        c = 1.0  # Tunable
        L_drive = torch.exp(-lambda_pde + c)

        # PDE loss as the residual
        pde_loss = torch.mean(pde_residual ** 2) + L_f + L_lambda

        return pde_loss, pde_residual, lambda_pde

    def loss(self, x, x_bc, u_bc):
        """
        Computes the total loss combining BC loss, PDE loss (Gross Pitaevskii), and Riesz loss.

        Parameters
        ----------
        x : torch.Tensor
            Boundary condition input data.
        x_bc : torch.Tensor
            Boundary condition true values.
        u_bc : torch.Tensor
            Input points for PDE training.

        Returns
        -------
        torch.Tensor
            Total loss combining data loss, pde loss, and the Riesz loss.
        """

        data_loss = self.boundary_loss(x_bc, u_bc)
        riesz_loss = self.riesz_loss(self.forward(x), x)
        pde_loss, _, _ = self.pde_loss(x, self.forward(x))

        alpha = 1.0
        beta = 1.0
        gamma = 1.0
        total_loss = alpha * data_loss + beta * pde_loss + gamma * riesz_loss
        return total_loss

    def compute_potential(self, inputs, V0=1.0, x0=np.pi/2, y0=np.pi/2, sigma=0.5):
        """
        Compute the Gaussian potential V(x,y) over the domain.

        V(x,y) = V0 * exp(-((x - x0)^2 + (y - y0)^2) / (2 * sigma^2))

        Parameters
        ----------
        inputs : torch.Tensor
            The input spatial coordinates (x, y) as a 2D tensor.
        V0 : float, optional
            Amplitude of the Gaussian potential (default is 1.0).
        x0 : float, optional
            Center of the Gaussian in the x-direction (default is π/2).
        y0 : float, optional
            Center of the Gaussian in the y-direction (default is π/2).
        sigma : float, optional
            Width (spread) of the Gaussian (default is 0.5).

        Returns
        -------
        V : torch.Tensor
            The potential evaluated at each input point.
        """
        x = inputs[:, 0]
        y = inputs[:, 1]

        # Gaussian potential
        V = V0 * torch.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

        return V


def prepare_training_data(N_u, N_f, center=(np.pi / 2, np.pi / 2), radius=np.pi / 2):
    # Generate boundary points along the circular edge
    theta = np.linspace(0, 2 * np.pi, N_u)
    circle_x = center[0] + radius * np.cos(theta)
    circle_y = center[1] + radius * np.sin(theta)
    X_u_train = np.column_stack((circle_x, circle_y))
    u_train = np.zeros((X_u_train.shape[0], 1))  # Boundary condition u=0

    # Generate collocation points within the effective circular region
    collocation_points = []
    while len(collocation_points) < N_f:
        random_angle = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0, radius)
        random_x = center[0] + r * np.cos(random_angle)
        random_y = center[1] + r * np.sin(random_angle)
        collocation_points.append([random_x, random_y])

    X_f_train = np.array(collocation_points)
    return X_f_train, X_u_train, u_train


def visualize_training_data(X_f_train, X_u_train, u_train, center=(np.pi/2, np.pi/2), radius=np.pi/2):
    """
    Visualizes the boundary points, collocation points, and the effective domain of the Gaussian potential.
    """
    plt.figure(figsize=(8, 8))

    # Plot boundary points
    plt.scatter(X_u_train[:, 0], X_u_train[:, 1], color='red', label='Boundary Points', alpha=0.6)

    # Plot collocation points
    plt.scatter(X_f_train[:, 0], X_f_train[:, 1], color='blue', label='Collocation Points', alpha=0.3)

    # Plot the effective region of the Gaussian potential as a circle
    circle = plt.Circle(center, radius, color='green', fill=False, label='Effective Region')
    plt.gca().add_artist(circle)

    plt.xlim([center[0] - radius - 0.2, center[0] + radius + 0.2])
    plt.ylim([center[1] - radius - 0.2, center[1] + radius + 0.2])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('$x$', fontsize=14)
    plt.ylabel('$y$', fontsize=14)
    plt.title('Boundary and Collocation Points within Effective Region')
    plt.legend()
    plt.grid()
    plt.show()


def train_pinn_hybrid(model, adam_optimizer, lbfgs_optimizer, scheduler, X_f_train, X_u_train, u_train, epochs_adam, epochs_lbfgs):
    """
    Hybrid training loop for the PINN model using Adam with mixed precision followed by LBFGS. Plots training error.

    Parameters
    ----------
    model : GrossPitaevskiiPINN
        The PINN model to be trained.
    adam_optimizer : torch.optim.Optimizer
        Adam optimizer for initial training.
    lbfgs_optimizer : torch.optim.Optimizer
        LBFGS optimizer for fine-tuning.
    scheduler : torch.optim.lr_scheduler
        Learning rate scheduler.
    X_f_train : torch.Tensor
        Collocation points from training data.
    X_u_train : torch.Tensor
        Boundary condition points from training data.
    u_train : torch.Tensor
        Boundary condition values to be enforced by PDE.
    epochs_adam : int
        Number of epochs for Adam optimization.
    epochs_lbfgs : int
        Number of epochs for LBFGS optimization.
    """

    scaler = torch.amp.GradScaler()  # Mixed precision scaler

    # Initialize lists to track progress
    train_losses = []
    test_losses = []
    test_metrics = []
    steps = []

    # Ensure input tensors have requires_grad=True
    X_f_train = X_f_train.clone().detach().requires_grad_(True)
    X_u_train = X_u_train.clone().detach().requires_grad_(True)
    u_train = u_train.clone().detach().requires_grad_(True)

    # Adam optimization phase with mixed precision
    for epoch in range(epochs_adam):
        model.train()
        adam_optimizer.zero_grad()

        # Calculate the total loss
        loss = model.loss(X_f_train, X_u_train, u_train)

        # Gradient clipping - Prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        # Backpropagation and optimization
        scaler.scale(loss).backward()
        scaler.step(adam_optimizer)
        scaler.update()

        scheduler.step(loss)

        # Center and parameters for Gaussian potential
        center = (np.pi / 2, np.pi / 2)  # Center of the Gaussian potential
        radius = np.pi / 2  # Effective radius
        sigma = 1.0  # Width of the Gaussian

        if epoch % 200 == 0:
            train_losses.append(loss.item())
            pde_loss, _, lambda_pde = model.pde_loss(X_f_train, model.forward(X_f_train))
            plot_solution(model, num_grid_pts=100, center=center, radius=radius, epoch=epoch,
                          lambda_pde=lambda_pde.item())

            # Evaluation on test data
            u_pred_test = model(X_u_test_tensor)  # Predictions from the model

            # Reshape predicted solution
            num_grid_pts = int(np.sqrt(X_u_test_tensor.shape[0]))
            u_pred_test_reshaped = u_pred_test.cpu().detach().numpy().reshape((num_grid_pts, num_grid_pts))

            # Ground Truth Calculation (Match the Gaussian potential)
            x_vals = np.linspace(center[0] - radius, center[0] + radius, num_grid_pts)
            y_vals = np.linspace(center[1] - radius, center[1] + radius, num_grid_pts)
            X, Y = np.meshgrid(x_vals, y_vals)

            # Compute the ground truth based on the Gaussian potential
            u_true_full_grid = np.exp(-((X - center[0]) ** 2 + (Y - center[1]) ** 2) / (2 * sigma ** 2))
            u_true_full_grid_reshaped = u_true_full_grid.reshape((num_grid_pts, num_grid_pts))

            # Compute the absolute error
            abs_error = np.abs(u_pred_test_reshaped - u_true_full_grid_reshaped)

            # Plot the predicted solution and absolute error in the current epoch
            plot_solution_and_error(X_u_test_tensor.detach().cpu().numpy(), u_pred_test_reshaped, abs_error, epoch,
                                    center=center, radius=radius)

            print(f'Epoch {epoch}/{epochs_adam}, Train Loss: {loss.item():.8f}')

    # LBFGS optimization
    def closure():
        lbfgs_optimizer.zero_grad()
        loss = model.loss(X_f_train, X_u_train, u_train)
        loss.backward()
        return loss

    for _ in range(epochs_lbfgs):
        lbfgs_optimizer.step(closure)

    print("Training complete.")

    # Plot the training progress after training is done
    plot_training_progress(train_losses, test_losses, test_metrics, steps)


def plot_training_progress(train_losses, test_losses, test_metrics, steps):
    """
    Plot the training and test losses, along with the test metric, over the course of training.

    Parameters
    ----------
    train_losses : list
        List of training loss values recorded at each training step.
    test_losses : list
        List of test loss values recorded at each evaluation step.
    test_metrics : list
        List of test metrics (such as error or accuracy) recorded during training.
    steps : list
        List of step numbers corresponding to the recorded losses and metrics.

    Returns
    -------
    None
    """

    plt.figure(figsize=(10, 6))

    # Plot training loss
    plt.plot(steps, train_losses, label='Train Loss', color='blue', linestyle='-', marker='o')

    # Plot test loss
    plt.plot(steps, test_losses, label='Test Loss', color='red', linestyle='--', marker='x')

    # Plot test error (or other metrics)
    plt.plot(steps, test_metrics, label='Test Error', color='green', linestyle='-.', marker='s')

    plt.xlabel('Steps')
    plt.ylabel('Loss/Error')
    plt.legend()
    plt.title('Training Progress')

    plt.tight_layout()
    plt.show()

    plt.savefig('training_progress_gpe_2.png')


def plot_solution_and_error(X_test, u_pred, abs_error, epoch, center=(np.pi/2, np.pi/2), radius=np.pi/2):
    """
    Plots the predicted solution and the absolute error for the Gross-Pitaevskii equation.

    Parameters
    ----------
    X_test : np.ndarray
        Test data (2D grid points) used for predictions.
    u_pred : np.ndarray
        Predicted solution u_pred from the neural network.
    abs_error : np.ndarray
        Absolute error between predicted solution and ground truth.
    epoch : int
        Current iteration of the optimizer.
    center : tuple
        Center of the Gaussian potential (default is (pi/2, pi/2)).
    radius : float
        Effective radius of the Gaussian potential (default is pi/2).
    """
    plt.figure(figsize=(16, 6))

    # Reshape X_test to 2D
    num_grid_pts = int(np.sqrt(X_test.shape[0]))
    X = X_test[:, 0].reshape((num_grid_pts, num_grid_pts))
    Y = X_test[:, 1].reshape((num_grid_pts, num_grid_pts))

    # Reshape u_pred and abs_error to 2D
    u_pred = u_pred.reshape((num_grid_pts, num_grid_pts))
    abs_error = abs_error.reshape((num_grid_pts, num_grid_pts))

    # Subplot 1: Predicted solution
    plt.subplot(1, 2, 1)
    plt.pcolor(X, Y, u_pred, cmap='viridis', shading='auto')
    plt.colorbar(label='u_pred')
    plt.title(f'Predicted Solution $u_{{pred}}$ - Iteration {epoch}')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.xlim([center[0] - radius, center[0] + radius])
    plt.ylim([center[1] - radius, center[1] + radius])

    # Subplot 2: Absolute error
    plt.subplot(1, 2, 2)
    plt.contourf(X, Y, abs_error, levels=50, cmap='inferno', alpha=0.9)
    plt.colorbar(label='Absolute Error')
    plt.title(f'Absolute Error - Iteration {epoch}')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.xlim([center[0] - radius, center[0] + radius])
    plt.ylim([center[1] - radius, center[1] + radius])
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_solution(model, num_grid_pts=100, center=(np.pi / 2, np.pi / 2), radius=np.pi / 2, epoch=0, lambda_pde=0):
    x_vals = np.linspace(center[0] - radius, center[0] + radius, num_grid_pts)
    y_vals = np.linspace(center[1] - radius, center[1] + radius, num_grid_pts)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Prepare test data
    X_test = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)  # Move to device

    # Predict the solution using the trained model
    u_pred = model(X_test_tensor).detach().cpu().numpy().reshape((num_grid_pts, num_grid_pts))

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.pcolor(X, Y, u_pred, shading='auto', cmap='viridis')
    plt.colorbar(label='Predicted Solution $u_{pred}$')
    plt.title(
        f'Predicted Solution of the Gross-Pitaevskii Equation\nEpoch: {epoch}, Smallest Eigenvalue: {lambda_pde:.4f}')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.xlim([center[0] - radius, center[0] + radius])
    plt.ylim([center[1] - radius, center[1] + radius])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(['Smallest Eigenvalue'])
    plt.show()


def plot_potential(X_test, potential):
    """
    Plot the potential.
    """
    plt.figure(figsize=(6, 5))
    X = X_test[:, 0].reshape((num_grid_pts, num_grid_pts))
    Y = X_test[:, 1].reshape((num_grid_pts, num_grid_pts))
    potential = potential.reshape((num_grid_pts, num_grid_pts))

    plt.contourf(X, Y, potential, levels=50, cmap='viridis')
    plt.colorbar()
    plt.title('Potential V(x, y)')
    plt.show()


def visualize_training_data(X_f_train, X_u_train, u_train, center=(np.pi/2, np.pi/2), radius=np.pi/2):
    """
    Visualizes the boundary points, collocation points, and the effective domain of the Gaussian potential.

    Parameters
    ----------
    X_f_train : np.ndarray
        Collocation points to visualize.
    X_u_train : np.ndarray
        Boundary points to visualize.
    u_train : np.ndarray
        Corresponding boundary condition values.
    center : tuple
        Center of the effective potential region (default is (pi/2, pi/2)).
    radius : float
        Radius of the effective region where the potential is significant.
    """
    plt.figure(figsize=(8, 8))

    # Plot boundary points
    plt.scatter(X_u_train[:, 0], X_u_train[:, 1], color='red', label='Boundary Points', alpha=0.6)

    # Plot collocation points
    plt.scatter(X_f_train[:, 0], X_f_train[:, 1], color='blue', label='Collocation Points', alpha=0.3)

    # Plot the effective region of the Gaussian potential as a circle
    circle = plt.Circle(center, radius, color='green', fill=False, label='Domain')
    plt.gca().add_artist(circle)

    # Set limits based on the domain
    plt.xlim([center[0] - radius - 0.2, center[0] + radius + 0.2])
    plt.ylim([center[1] - radius - 0.2, center[1] + radius + 0.2])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('$x$', fontsize=14)
    plt.ylabel('$y$', fontsize=14)
    plt.title('Boundary and Collocation Points within Domain')
    plt.legend()
    plt.grid()
    plt.show()


def initialize_weights(m):
    """Initialize weights for the neural network layers."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


if __name__ == "__main__":

    # Specify number of grid points and number of dimensions
    num_grid_pts = 512
    center = (np.pi / 2, np.pi / 2)
    radius = np.pi / 2

    x_vals = np.linspace(center[0] - radius, center[0] + radius, num_grid_pts)
    y_vals = np.linspace(center[1] - radius, center[1] + radius, num_grid_pts)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Prepare test data
    X_u_test = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))

    N_u = 500  # Number of boundary points
    N_f = 10000  # Number of collocation points
    X_f_train_np_array, X_u_train_np_array, u_train_np_array = prepare_training_data(N_u, N_f)

    # Convert numpy arrays to PyTorch tensors and move to GPU (if available)
    X_f_train = torch.from_numpy(X_f_train_np_array).float().to(device)  # Collocation points
    X_u_train = torch.from_numpy(X_u_train_np_array).float().to(device)  # Boundary condition points
    u_train = torch.from_numpy(u_train_np_array).float().to(device)  # Boundary condition values
    X_u_test_tensor = torch.from_numpy(X_u_test).float().to(device)  # Test data for boundary conditions

    # Model parameters
    layers = [2, 100, 100, 100, 1]  # Neural network layers
    epochs_adam = 1000
    epochs_lbfgs = 500

    # Initialize the model
    model = GrossPitaevskiiPINN(layers).to(device)
    model.apply(initialize_weights)  # Apply weight initialization

    # Print the neural network architecture
    print(model)

    # Calculate the potential
    X_test_tensor = torch.from_numpy(X_u_test).float().to(device)
    potential = model.compute_potential(X_test_tensor).cpu().detach().numpy()

    # Visualize the potential
    plot_potential(X_u_test, potential)

    # Visualize training data
    visualize_training_data(X_f_train_np_array, X_u_train_np_array, u_train_np_array)

    # Optimizers and scheduler
    adam_optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-6,
                                amsgrad=False)
    lbfgs_optimizer = optim.LBFGS(model.parameters(), max_iter=500, tolerance_grad=1e-5, tolerance_change=1e-9,
                                  history_size=100)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(adam_optimizer, patience=200, factor=0.5, verbose=True)

    # Train the model using the hybrid approach
    train_pinn_hybrid(model, adam_optimizer, lbfgs_optimizer, scheduler, X_f_train, X_u_train, u_train, epochs_adam,epochs_lbfgs)