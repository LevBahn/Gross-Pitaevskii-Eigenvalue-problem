import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import os
from pyDOE import lhs
import matplotlib.pyplot as plt

# Set default data type and random seeds
torch.set_default_dtype(torch.float32)
torch.manual_seed(1234)
np.random.seed(1234)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class HelmholtzPINN(nn.Module):
    """
    PINN for solving the 2D Helmholtz equation in a rectangular domain.

    Parameters
    ----------
    layers : list of int
        A list defining the number of nodes in each layer of the network.
    """

    def __init__(self, layers):
        super().__init__()
        self.activation = nn.LeakyReLU()
        self.loss_function = nn.MSELoss()
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])

        # Xavier normal initialization for weights
        for linear in self.linears:
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, 2), where N is the number of points.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, 1).
        """
        for i, linear in enumerate(self.linears[:-1]):
            x = self.activation(linear(x))
        return self.linears[-1](x)

    def loss_pde(self, x_f, k):
        """
        Computes the PDE loss (residual of the 2D Helmholtz equation) using automatic differentiation to calculate
        second-order derivatives.

        Parameters
        ----------
        x_f : torch.Tensor
            Collocation points for PDE training.
        k : float
            Wave number for the Helmholtz equation.

        Returns
        -------
        torch.Tensor
            PDE loss.
        """
        g = x_f.clone().requires_grad_(True)
        u = self.forward(g)

        u_x = grad(u, g, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 0]
        u_y = grad(u, g, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 1]

        u_xx = grad(u_x, g, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0]
        u_yy = grad(u_y, g, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1]
        laplacian_u = u_xx + u_yy

        # PDE residual
        q = k ** 2 * torch.sin(np.pi * g[:, [0]]) * torch.sin(np.pi * g[:, [1]])
        f = laplacian_u + k ** 2 * u - q

        return self.loss_function(f, torch.zeros_like(f))

    def loss_data(self, x_u, u_true):
        """
        Computes the boundary data loss (MSE between true and predicted boundary values).

        Parameters
        ----------
        x_u : torch.Tensor
            Boundary condition points.
        u_true : torch.Tensor
            True solution values at boundary points.

        Returns
        -------
        torch.Tensor
            Data loss.
        """
        u_pred = self.forward(x_u)
        return self.loss_function(u_pred, u_true)

    def compute_loss(self, x_f, x_u, u_true, k):
        """
        Computes the total loss as the sum of PDE loss and data loss.

        Parameters
        ----------
        x_f : torch.Tensor
            Collocation points.
        x_u : torch.Tensor
            Boundary condition points.
        u_true : torch.Tensor
            True boundary values.
        k : float
            Wavenumber.

        Returns
        -------
        torch.Tensor
            Total loss.
        """
        pde_loss = self.loss_pde(x_f, k)
        data_loss = self.loss_data(x_u, u_true)
        return pde_loss + data_loss


def create_grid(num_points=256):
    """
    Creates a 2D grid of points over the domain [0, pi] x [0, pi].

    Parameters
    ----------
    num_points : int, optional
        Number of points along each axis (default is 256).

    Returns
    -------
    X : np.ndarray
        Grid points in the x-dimension.
    Y : np.ndarray
        Grid points in the y-dimension.
    """
    x = np.linspace(0, np.pi, num_points)
    y = np.linspace(0, np.pi, num_points)
    X, Y = np.meshgrid(x, y)
    return X, Y


def prepare_training_data(N_u, N_f, lb, ub, usol, X, Y):
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
    usol : np.ndarray
        Analytical solution at grid points.
    X, Y : np.ndarray
        Grids of x and y points.

    Returns
    -------
    X_f_train : np.ndarray
        Collocation points.
    X_u_train : np.ndarray
        Boundary points.
    u_train : np.ndarray
        Boundary values.
    """
    # Boundary points (4 edges of square)
    boundary_points = np.vstack([
        np.column_stack((X[:, 0], Y[:, 0])),
        np.column_stack((X[:, -1], Y[:, -1])),
        np.column_stack((X[0, :], Y[0, :])),
        np.column_stack((X[-1, :], Y[-1, :]))
    ])
    boundary_values = (
        np.vstack([ usol[:, 0], usol[:, -1],
                    usol[0, :], usol[-1, :]]).reshape(-1, 1))

    # Randomly select N_u boundary points
    idx = np.random.choice(boundary_points.shape[0], N_u, replace=False)
    X_u_train = boundary_points[idx, :]
    u_train = boundary_values[idx, :]

    # Collocation points using Latin Hypercube Sampling
    X_f_train = lb + (ub - lb) * lhs(2, N_f)

    return X_f_train, X_u_train, u_train


def plot_solution(u_pred, usol, X, Y, epoch, figDir):
    """
    Plot the predicted solution, analytical solution, and absolute error.

    Parameters
    ----------
    u_pred : np.ndarray
        Predicted solution from the model.
    usol : np.ndarray
        Analytical solution.
    X, Y : np.ndarray
        Grids of x and y points.
    epoch : int
        Current training epoch.
    figDir : str
        Directory where the figures will be saved.
    """
    abs_error = np.abs(usol - u_pred)

    plt.figure(figsize=(18, 5))

    # Analytical solution
    plt.subplot(1, 3, 1)
    plt.pcolor(X, Y, usol, cmap='jet')
    plt.colorbar()
    plt.title('Analytical Solution')

    # Predicted solution
    plt.subplot(1, 3, 2)
    plt.pcolor(X, Y, u_pred, cmap='jet')
    plt.colorbar()
    plt.title(f'Predicted Solution at Epoch {epoch}')

    # Absolute error
    plt.subplot(1, 3, 3)
    plt.pcolor(X, Y, abs_error, cmap='jet')
    plt.colorbar()
    plt.title('Absolute Error')

    plt.tight_layout()

    # Create the figures directory if it doesn't exist
    os.makedirs(figures_dir, exist_ok=True)

    # Save the figure in the specified directory
    figure_path = os.path.join(figures_dir, f'Helmholtz_iter_{epoch}.png')
    plt.savefig(figure_path, dpi=500, bbox_inches='tight')

    plt.show()
    plt.pause(0.1)


def plot_training_domain(X_f_train, X_u_train):
    """
    Visualizes the boundary points and collocation points on a π × π square domain.

    Parameters
    ----------
    X_f_train : np.ndarray
        Collocation points to visualize.
    X_u_train : np.ndarray
        Boundary points to visualize.
    """
    plt.figure(figsize=(6, 6))

    # Plot boundary points
    plt.scatter(X_u_train[:, 0], X_u_train[:, 1], color='red', label='Boundary Points', alpha=0.6)

    # Plot collocation points
    plt.scatter(X_f_train[:, 0], X_f_train[:, 1], color='blue', label='Collocation Points', alpha=0.3)

    # Plot the π × π square
    square = plt.Rectangle((0, 0), np.pi, np.pi, edgecolor='green', fill=False, label='π-Square')
    plt.gca().add_artist(square)

    plt.xlim([0, np.pi])
    plt.ylim([0, np.pi])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('$x_1$', fontsize=14)
    plt.ylabel('$x_2$', fontsize=14)
    plt.title('Boundary and Collocation Points for Training')
    plt.legend()
    plt.grid()
    plt.show()


def plot_training_progress(train_losses, test_losses, train_maes, test_maes, steps, figDir):
    """
    Plot the training and test losses, along with the train and test MAEs, over the course of training.

    Parameters
    ----------
    train_losses : list
        List of training loss values recorded at each training step.
    test_losses : list
        List of test loss values recorded at each evaluation step.
    train_maes : list
        List of train MAE values recorded during training.
    test_maes : list
        List of test MAE values recorded during training.
    steps : list
        List of step numbers corresponding to the recorded losses and metrics.
    figDir : str
        Directory where the figures will be saved.

    Returns
    -------
    None
    """
    plt.figure(figsize=(12, 8))

    # Plot training loss
    plt.plot(steps, train_losses, label='Train Loss (MSE)', color='blue', linestyle='-', marker='o')

    # Plot test loss
    plt.plot(steps, test_losses, label='Test Loss (MSE)', color='red', linestyle='--', marker='x')

    # Plot train MAE
    plt.plot(steps, train_maes, label='Train MAE', color='green', linestyle='-.', marker='s')

    # Plot test MAE
    plt.plot(steps, test_maes, label='Test MAE', color='orange', linestyle=':', marker='d')

    plt.xlabel('Steps', fontsize="x-large")
    plt.ylabel('MSE / MAE', fontsize="x-large")
    plt.legend()
    plt.title('Training Progress: MSE and MAE')

    plt.tight_layout()
    plt.yscale('log')

    # Create the directory if it doesn't exist
    os.makedirs(figures_dir, exist_ok=True)

    # Save the figure in the specified directory
    figure_path = os.path.join(figures_dir, f'Helmholtz_Training_Progress.png')
    plt.savefig(figure_path, dpi=500, bbox_inches='tight')

    plt.savefig('training_progress.png', dpi=500, bbox_inches='tight')
    plt.show()
    plt.pause(0.1)


def train_pinn(N_u=500, N_f=10000, layers=[2, 200, 200, 200, 1], epochs=1000, lr=0.01, figDir=os.path.join(os.getcwd(), "Helmholtz_Figures")):
    """
    Train the Physics-Informed Neural Network for the Helmholtz equation.

    Parameters
    ----------
    N_u : int
        Number of boundary points.
    N_f : int
        Number of collocation points.
    layers : list of int
        Network architecture.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate.
    figDir : str
        Directory where the figures will be saved.

    Returns
    -------
    HelmholtzPINN
        Trained model.
    """
    # Generate grid and exact solution
    X, Y = create_grid()
    usol = np.sin(np.pi * X) * np.sin(np.pi * Y)
    u_true = usol.flatten()[:, None]
    lb = np.array([0, 0], dtype=np.float32)
    ub = np.array([np.pi, np.pi], dtype=np.float32)

    # Prepare training data
    X_f_train, X_u_train, u_train = prepare_training_data(N_u, N_f, lb, ub, usol, X, Y)

    # Split the boundary data into 80% training and 20% testing
    split_idx = int(0.8 * X_u_train.shape[0])
    X_u_train_split = X_u_train[:split_idx]
    u_train_split = u_train[:split_idx]
    X_u_test = X_u_train[split_idx:]
    u_test = u_train[split_idx:]

    # Convert to tensors
    X_f_train = torch.tensor(X_f_train, dtype=torch.float32).to(device)
    X_u_train = torch.tensor(X_u_train, dtype=torch.float32).to(device)
    X_u_train_split = torch.tensor(X_u_train_split, dtype=torch.float32).to(device)
    u_train_split = torch.tensor(u_train_split, dtype=torch.float32).to(device)
    X_u_test = torch.tensor(X_u_test, dtype=torch.float32).to(device)
    u_test = torch.tensor(u_test, dtype=torch.float32).to(device)

    # Visualize the training domain
    plot_training_domain(X_f_train.cpu().numpy(), X_u_train.cpu().numpy())

    # Create model
    model = HelmholtzPINN(layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Store training progress
    train_losses = []
    test_losses = []
    train_maes = []
    test_maes = []
    steps = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Compute train loss
        loss = model.compute_loss(X_f_train, X_u_train_split, u_train_split, k=np.sqrt(2) * np.pi)
        loss.backward()
        optimizer.step()

        # Record training loss and MAE
        train_loss = loss.item()
        u_pred_train = model(X_u_train_split)
        train_mae = nn.L1Loss()(u_pred_train, u_train_split).item()

        # Test the model on test data every 10 epochs
        if epoch % 10 == 0:
            with torch.no_grad():
                u_pred_test = model(X_u_test)
                test_loss = model.loss_data(X_u_test, u_test).item()  # Test loss (MSE)
                test_mae = nn.L1Loss()(u_pred_test, u_test).item()  # Test MAE

                # Store values for plotting
                steps.append(epoch)
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                train_maes.append(train_mae)
                test_maes.append(test_mae)

        # Print every 400 epochs
        if epoch % 400 == 0:
            print(f'Epoch {epoch}/{epochs}, Train Loss: {train_loss:.6f}, Train MAE: {train_mae:.6f}',
                  f'Test Loss: {test_loss:.6f}, Test MAE: {test_mae:.6f}')

            # Predict solution
            u_pred = model(torch.tensor(np.column_stack((X.flatten(), Y.flatten())), dtype=torch.float32).to(device))
            u_pred = u_pred.cpu().detach().numpy().reshape(X.shape)

            # Plot analytical, predicted solution, and absolute error
            plot_solution(u_pred, usol, X, Y, epoch, figDir)

    # Plot the training/test loss and mae over time
    plot_training_progress(train_losses, test_losses, train_maes, test_maes, steps, figDir)

    return model


if __name__ == "__main__":

    # Network Parameters
    N_u = 500  # Number of boundary points
    N_f = 10000  # Number of collocation points
    layers = [2, 200, 200, 200, 1]  # Neural network architecture
    epochs = 2001  # Number of training epochs
    lr = 0.01 # Learning rate

    # Create figures directory if it doesn't exist
    figures_dir = os.path.join(os.getcwd(), "Helmholtz_Figures")
    os.makedirs(figures_dir, exist_ok=True)

    # Train the PINN
    model = train_pinn(N_u=N_u, N_f=N_f, layers=layers, epochs=epochs, lr=lr, figDir=figures_dir)
