import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import os
from scipy.special import j0, jn
import matplotlib.pyplot as plt

# Set default data type and random seeds
torch.set_default_dtype(torch.float32)
torch.manual_seed(1234)
np.random.seed(1234)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class HelmholtzPINN(nn.Module):
    """
    PINN for solving the 2D Helmholtz equation in a circular domain.

    Parameters
    ----------
    layers : list of int
        A list defining the number of nodes in each layer of the network.
    """

    def __init__(self, layers):
        super().__init__()
        self.activation = nn.Tanh()
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

    def loss_pde(self, x_f, n, k):
        """
        Computes the PDE loss (residual of the 2D Helmholtz equation) using automatic differentiation to calculate
        second-order derivatives.

        Parameters
        ----------
        x_f : torch.Tensor
            Collocation points for PDE training.
        n : int
            Mode number for the Bessel function of the first kind.
        k : float
            Wave number for the Helmholtz equation.

        Returns
        -------
        torch.Tensor
            PDE loss.
        """
        g = x_f.clone().requires_grad_(True)  # Ensure gradients can be computed for x_f
        u = self.forward(g)  # Forward pass to get predicted u at collocation points

        # Convert to polar coordinates
        r = torch.sqrt(g[:, 0] ** 2 + g[:, 1] ** 2)  # Radius r = sqrt(x^2 + y^2)
        theta = torch.atan2(g[:, 1], g[:, 0])  # Angle theta = atan2(y, x)

        # First derivatives of u with respect to x and y
        u_x = grad(u, g, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 0]
        u_y = grad(u, g, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 1]

        # First derivatives of u with respect to r and theta using chain rule
        u_r = u_x * torch.cos(theta) + u_y * torch.sin(theta)
        u_theta = -u_x * torch.sin(theta) + u_y * torch.cos(theta)

        # Second derivatives of u with respect to x and y
        u_xx = grad(u_x, g, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0]
        u_yy = grad(u_y, g, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1]

        # Second derivatives of u with respect to r and theta
        u_rr = u_xx * torch.cos(theta) ** 2 + u_yy * torch.sin(theta) ** 2
        u_thetatheta = grad(u_theta, theta, grad_outputs=torch.ones_like(u_theta), create_graph=True)[0] / r

        # Laplacian in polar coordinates
        laplacian_u = u_rr + (1 / r) * u_r + (1 / r ** 2) * u_thetatheta

        # Source term
        q = k ** 2 * torch.special.bessel_j0(k * r) * torch.cos(n * theta)

        # PDE residual: Laplacian(u) + k^2 * u - q
        residual = laplacian_u + k ** 2 * u - q

        # Return the mean squared error between the residual and zero
        return self.loss_function(residual, torch.zeros_like(residual))

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

    def compute_loss(self, x_f, x_u, u_true, n, k):
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
        n : int
            Mode number.
        k : float
            Wave number.

        Returns
        -------
        torch.Tensor
            Total loss.
        """
        pde_loss = self.loss_pde(x_f, n, k)
        data_loss = self.loss_data(x_u, u_true)
        return pde_loss + data_loss


def create_circle_grid(num_points=256, radius=np.pi):
    """
    Create a 2D grid of points inside a circle of radius pi.

    Parameters
    ----------
    num_points : int
        Number of points to generate.
    radius : float
        Radius of the circle.

    Returns
    -------
    X : np.ndarray
        Grid points in the x-dimension.
    Y : np.ndarray
        Grid points in the y-dimension.
    mask : np.ndarray
        Mask indicating points inside the circle.
    """
    theta = np.linspace(0, 2 * np.pi, num_points)
    r = np.linspace(0, radius, num_points)
    R, Theta = np.meshgrid(r, theta)
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)

    # Filter points inside the circle (r <= radius)
    mask = R <= radius
    return X[mask], Y[mask]


def prepare_circle_training_data(N_u, N_f, radius, usol):
    """
    Prepare boundary and collocation points for a circular domain.

    Parameters
    ----------
    N_u : int
        Number of boundary points.
    N_f : int
        Number of collocation points.
    radius : float
        Radius of the circle.
    usol : np.ndarray
        Analytical solution at grid points.

    Returns
    -------
    X_f_train : np.ndarray
        Collocation points.
    X_u_train : np.ndarray
        Boundary points.
    u_train : np.ndarray
        Boundary values.
    """
    theta = np.linspace(0, 2 * np.pi, N_u)
    X_u_train = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])  # Boundary points on the circle
    u_train = usol(X_u_train[:, 0], X_u_train[:, 1]).reshape(-1, 1)  # Boundary values

    # Collocation points inside the circle
    X_f_train = np.random.uniform(-radius, radius, (N_f, 2))
    mask = np.linalg.norm(X_f_train, axis=1) <= radius
    X_f_train = X_f_train[mask]

    return X_f_train, X_u_train, u_train


def usol(x, y, n=1, k=np.sqrt(2) * np.pi):
    """
    Analytical solution for the Helmholtz equation inside a circular domain through the Bessel function.

    Parameters
    ----------
    x, y : float
        Coordinates.
    k : float
        Wavenumber.

    Returns
    -------
    u : float
        Analytical solution value.
    """
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.linspace(0, 2 * np.pi, len(x))
    return j0(k * r) * np.cos(n * theta)


def plot_solution(u_pred, usol, X, Y, epoch, figures_dir, radius=np.pi):
    """
    Plot the predicted solution, analytical solution, and absolute error inside a circular domain.

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
    figures_dir : str
        Directory where the figures will be saved.
    radius : float
        Radius of the circular domain.
    """
    r = np.sqrt(X ** 2 + Y ** 2)
    mask = r <= radius  # Only points inside the circle

    abs_error = np.abs(usol - u_pred)

    plt.figure(figsize=(18, 5))

    # Analytical solution (mask points outside the circle)
    plt.subplot(1, 3, 1)
    plt.scatter(X[mask], Y[mask], c=usol[mask], cmap='jet', s=1)
    plt.colorbar()
    plt.title('Analytical Solution')

    # Predicted solution
    plt.subplot(1, 3, 2)
    plt.scatter(X[mask], Y[mask], c=u_pred[mask], cmap='jet', s=1)
    plt.colorbar()
    plt.title(f'Predicted Solution at Epoch {epoch}')

    # Absolute error
    plt.subplot(1, 3, 3)
    plt.scatter(X[mask], Y[mask], c=abs_error[mask], cmap='jet', s=1)
    plt.colorbar()
    plt.title('Absolute Error')

    plt.tight_layout()

    # Create the directory if it doesn't exist
    os.makedirs(figures_dir, exist_ok=True)

    # Save the figure in the specified directory
    figure_path = os.path.join(figures_dir, f'Helmholtz_circle_iter_{epoch}.png')
    plt.savefig(figure_path, dpi=500, bbox_inches='tight')

    plt.show()
    plt.pause(0.1)


def plot_training_domain(X_f_train, X_u_train, radius=np.pi):
    """
    Visualizes the boundary points and collocation points on a circular domain.

    Parameters
    ----------
    X_f_train : np.ndarray
        Collocation points to visualize.
    X_u_train : np.ndarray
        Boundary points to visualize.
    radius : float
        Radius of the circular domain.
    """
    plt.figure(figsize=(6, 6))

    # Plot boundary points (red)
    plt.scatter(X_u_train[:, 0], X_u_train[:, 1], color='red', label='Boundary Points', alpha=0.6)

    # Plot collocation points (blue)
    plt.scatter(X_f_train[:, 0], X_f_train[:, 1], color='blue', label='Collocation Points', alpha=0.3)

    # Plot the circle
    circle = plt.Circle((0, 0), radius, color='green', fill=False, label=f'Circle with radius {radius}')
    plt.gca().add_artist(circle)

    plt.xlim([-radius, radius])
    plt.ylim([-radius, radius])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('$x$', fontsize=14)
    plt.ylabel('$y$', fontsize=14)
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
    figure_path = os.path.join(figures_dir, f'Helmholtz_Training_Progress_Circle.png')
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
    X, Y = create_circle_grid()
    usol_vals = usol(X, Y)
    radius = np.pi

    # Prepare training data
    X_f_train, X_u_train, u_train = prepare_circle_training_data(N_u, N_f, radius, usol)

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
    plot_training_domain(X_f_train.cpu().numpy(), X_u_train.cpu().numpy(), radius)

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
        loss = model.compute_loss(X_f_train, X_u_train_split, u_train_split, n=1, k=np.sqrt(2) * np.pi)
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
            plot_solution(u_pred, usol_vals, X, Y, epoch, figDir)

    # Plot the training/test loss and mae over time
    plot_training_progress(train_losses, test_losses, train_maes, test_maes, steps, figDir)

    return model


if __name__ == "__main__":

    # Network Parameters
    N_u = 500  # Number of boundary points
    N_f = 10000  # Number of collocation points
    layers = [2, 20, 20, 20, 1]  # Neural network architecture
    epochs = 1001  # Number of training epochs
    lr = 0.01 # Learning rate

    # Create figures directory if it doesn't exist
    figures_dir = os.path.join(os.getcwd(), "Helmholtz_Figures")
    os.makedirs(figures_dir, exist_ok=True)

    # Train the PINN
    model = train_pinn(N_u=N_u, N_f=N_f, layers=layers, epochs=epochs, lr=lr, figDir=figures_dir)
