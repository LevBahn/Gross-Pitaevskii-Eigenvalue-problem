import numpy as np
import torch
import torch.autograd as autograd         # Computation graph
from torch import Tensor                  # Tensor node in the computation graph
import torch.nn as nn                     # Neural networks
import torch.optim as optim               # Optimizers for gradient descent, ADAM, etc.
import time
from pyDOE import lhs                     # Latin Hypercube Sampling
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

plt.ion()  # Enable interactive mode

# Set default dtype to float32
torch.set_default_dtype(torch.float)

# PyTorch random number generator
torch.manual_seed(1234)

# NumPy random number generator
np.random.seed(1234)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_grid(num_grid_pts=256, n_dim=2):
    """
    Creates an n-dimensional grid of points as a NumPy array.

    Parameters
    ----------
    num_grid_pts : int, optional
        The number of grid points along each dimension (default is 256).
    n_dim : int, optional
        The number of dimensions (default is 2).

    Returns
    -------
    grid : np.ndarray
        n-dimensional grid points as a NumPy array.
    axis_points : list of np.ndarray
        List of 1D arrays of points for every dimension.
    """
    # Form 1D arrays for every dimension
    axis_points = [np.linspace(0, np.pi, num_grid_pts) for _ in range(n_dim)]

    # Generate a meshgrid up to n_dim
    grids = np.meshgrid(*axis_points, indexing='ij', sparse=False)

    return grids, axis_points


def prepare_test_data(X, Y):
    """
    Prepare test data by flattening the 2D grids and stacking them column-wise.

    Parameters
    ----------
    X : np.ndarray
        2D grid points in the x-dimension as a NumPy array.
    Y : np.ndarray
        2D grid points in the y-dimension as a NumPy array.

    Returns
    -------
    X_u_test : np.ndarray
        Test data prepared by stacking the flattened x and y grids.
    lb : np.ndarray
        Lower bound for the domain (boundary conditions).
    ub : np.ndarray
        Upper bound for the domain (boundary conditions).
    """
    # Flatten the grids and stack them into a 2D array
    X_u_test = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))

    # Domain bounds as NumPy arrays
    lb = np.array([0, 0], dtype=np.float32)
    ub = np.array([np.pi, np.pi], dtype=np.float32)

    return X_u_test, lb, ub


def prepare_training_data(N_u, N_f, lb, ub, usol, X, Y):
    """
    Prepare boundary condition data and collocation points for training.

    Parameters
    ----------
    N_u : int
        Number of boundary condition points to select.
    N_f : int
        Number of collocation points for the physics-informed model.
    lb : np.ndarray
        Lower bound for the domain.
    ub : np.ndarray
        Upper bound for the domain.
    usol : np.ndarray
        Analytical solution of the PDE.
    X : np.ndarray
        X grid of points.
    Y : np.ndarray
        Y grid of points.

    Returns
    -------
    X_f_train : np.ndarray
        Collocation points in the domain.
    X_u_train : np.ndarray
        Boundary condition points.
    u_train : np.ndarray
        Corresponding boundary condition values.
    """
    # Boundary points from all four edges
    boundary_points = np.vstack((
        np.hstack((X[:, 0][:, None], Y[:, 0][:, None])),
        np.hstack((X[:, -1][:, None], Y[:, -1][:, None])),
        np.hstack((X[0, :][:, None], Y[0, :][:, None])),
        np.hstack((X[-1, :][:, None], Y[-1, :][:, None]))
    ))

    # Boundary values (analytical solution at boundary points)
    boundary_values = np.vstack((
        usol[:, 0][:, None],
        usol[:, -1][:, None],
        usol[0, :][:, None],
        usol[-1, :][:, None]
    ))

    # Randomly select N_u boundary points
    idx = np.random.choice(boundary_points.shape[0], N_u, replace=False)
    X_u_train = boundary_points[idx, :]
    u_train = boundary_values[idx, :]

    # Generate N_f collocation points using Latin Hypercube Sampling
    X_f_train = lb + (ub - lb) * lhs(2, N_f)  # Collocation points in [lb, ub]

    return X_f_train, X_u_train, u_train


class HelmholtzPINN(nn.Module):
    """
    A neural network for solving data (boundary condition) loss and partial differential equation (residual) loss
    for the 2D Helmholtz equation in PyTorch.

    Parameters
    ----------
    layers : list
        A list defining the number of nodes in each layer of the network.
    """

    def __init__(self, layers):
        """
        Initializes the SequentialModel with the specified layers, activation function,
        loss function, and weight initialization.

        Parameters
        ----------
        layers : list
            A list of integers where each element defines the number of neurons
            in the respective layer.
        """
        super().__init__()

        # LeakyReLU activation function
        self.activation = nn.LeakyReLU()

        # Mean squared error (MSE) loss function
        self.loss_function = nn.MSELoss(reduction='mean')

        # L1 loss (mean absolute error) function
        self.l1loss_function = nn.L1Loss(reduction='mean')

        # Initialize the network as a list of linear layers
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])

        # Xavier normal initialization for weights and setting biases to zero
        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linears[i].bias.data)

    def forward(self, x):
        """
        Forward pass through the network. Scales the input features and passes them through
        the layers of the model, applying the activation function after each layer.

        Parameters
        ----------
        x : torch.Tensor or numpy array
            Input tensor or array to be processed.

        Returns
        -------
        torch.Tensor
            Output of the model.
        """
        # Convert numpy array to tensor if needed
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)

        # Convert lower and upper bounds to tensors
        u_b = torch.from_numpy(ub).float().to(device)
        l_b = torch.from_numpy(lb).float().to(device)

        # Feature scaling
        x = (x - l_b) / (u_b - l_b)

        # Convert input to float
        a = x.float()

        # Pass through each linear layer with activation
        for i in range(len(self.linears) - 1):
            z = self.linears[i](a)
            a = self.activation(z)

        # Final layer without activation
        a = self.linears[-1](a)

        return a

    def loss_PDE(self, x_to_train_f, k):
        """
        Computes the partial differential equation (PDE) loss using automatic differentiation
        to calculate the second-order derivatives.

        Parameters
        ----------
        x_to_train_f : torch.Tensor
            Input tensor for training PDE.
        k : int
            Wave number of the Helmholtz equation.

        Returns
        -------
        torch.Tensor
            Loss value for the PDE.
        """
        x_1_f = x_to_train_f[:, [0]]
        x_2_f = x_to_train_f[:, [1]]

        g = x_to_train_f.clone()
        g.requires_grad = True

        # Forward pass through the network
        u = self.forward(g)

        # Compute first-order derivatives
        u_x = autograd.grad(u, g, torch.ones([x_to_train_f.shape[0], 1]).to(device),
                            retain_graph=True, create_graph=True)[0]

        # Compute second-order derivatives
        u_xx = autograd.grad(u_x, g, torch.ones(x_to_train_f.shape).to(device),
                             create_graph=True)[0]
        u_xx_1 = u_xx[:, [0]]
        u_xx_2 = u_xx[:, [1]]

        # Define the PDE residual
        q = k**2 * torch.sin(n * x_1_f) * torch.sin(m * x_2_f)
        f = u_xx_1 + u_xx_2 + k**2 * u - q

        # PDE loss
        return self.loss_function(f, f_hat)

    def test(self, X_test, u_true):
        """
        Test the model on the test data and computes the relative L2 norm of the error and the mean absolute error.

        Parameters
        ----------
        X_test : torch.Tensor
            Test points across the domain.
        u_true : torch.Tensor
            True solution values (ground truth) for comparison.

        Returns
        -------
        error_vec : torch.Tensor
            The relative L2 norm of the error.
        u_pred : numpy.ndarray
            The predicted solution.
        mae : torch.Tensor
            Mean absolute error between the predictions and the true values.
        """
        # Model prediction
        u_pred = self.forward(X_test)

        # Compute the relative L2 norm of the error
        error_vec = torch.linalg.norm(u_true - u_pred, 2) / torch.linalg.norm(u_true, 2)

        # Compute the mean absolute error (MAE)
        mae = torch.mean(torch.abs(u_true - u_pred))

        # Reshape the predicted output
        u_pred = np.reshape(u_pred.cpu().detach().numpy(), (num_grid_pts, num_grid_pts), order='F')

        return error_vec, u_pred, mae


def solutionplot(u_pred, usol, x_1, x_2, index, X_f_train=None):
    """
    Plots the ground truth solution, predicted solution, and absolute error between them.
    Optionally includes the collocation points.

    Parameters
    ----------
    u_pred : numpy.ndarray
        Predicted solution values from the model.
    usol : numpy.ndarray
        Ground truth solution values to be plotted.
    x_1 : numpy.ndarray
        1D array of grid points in the x1-dimension.
    x_2 : numpy.ndarray
        1D array of grid points in the x2-dimension.
    index : int
        Current iteration of the optimizer.
    X_f_train : torch.Tensor, optional
        Collocation points used for PDE loss (optional, default is None).
    """

    plt.figure(figsize=(18, 5))

    # Plot ground truth solution
    plt.subplot(1, 3, 1)
    plt.pcolor(x_1, x_2, usol, cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x_1$', fontsize=18)
    plt.ylabel(r'$x_2$', fontsize=18)
    plt.title('Ground Truth $u(x_1,x_2)$', fontsize=15)

    # Plot predicted solution
    plt.subplot(1, 3, 2)
    plt.pcolor(x_1, x_2, u_pred, cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x_1$', fontsize=18)
    plt.ylabel(r'$x_2$', fontsize=18)
    plt.title(f'Predicted $\hat u(x_1,x_2)$ - Iteration {index}', fontsize=15)

    # Optionally plot collocation points
    if X_f_train is not None:
        X_f_train_np = X_f_train.cpu().numpy()
        plt.scatter(X_f_train_np[:, 0], X_f_train_np[:, 1], color='white', s=1, label="Collocation Points")
        plt.legend()

    # Plot absolute error
    plt.subplot(1, 3, 3)
    plt.pcolor(x_1, x_2, np.abs(usol - u_pred), cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x_1$', fontsize=18)
    plt.ylabel(r'$x_2$', fontsize=18)
    plt.title(r'Absolute error $|u(x_1,x_2)- \hat u(x_1,x_2)|$', fontsize=15)

    plt.tight_layout()
    plt.show()
    plt.pause(0.1)
    plt.savefig(f'Helmholtz_iter_{index}.png', dpi=500, bbox_inches='tight')


def LBFGS_training():
    """
    Computes the loss and its gradients for use with the LBFGS optimizer.

    Necessary for optimizers which require multiple evaluations of the function. It performs the following:
    - Resets gradients to zero.
    - Calculates the loss using the physics-informed neural network (PINN) model.
    - Backpropagates the gradients of the loss.

    Returns
    -------
    loss : torch.Tensor
        The computed loss value.
    """
    # Zero out the gradients of the optimizer before backpropagation
    lbfgs_optimizer.zero_grad()

    # Forward pass for boundary condition points
    u_pred_train = PINN.forward(X_u_train)

    # Data loss (MSE between predicted and true boundary condition values)
    data_loss = PINN.loss_function(u_pred_train, u_train)

    # Physics loss (PDE residuals)
    physics_loss = PINN.loss_PDE(X_f_train, k)

    # Total loss: sum of data loss and physics loss
    total_loss = data_loss + physics_loss

    # Perform backpropagation to compute the gradients of the loss
    total_loss.backward()

    # Return the total loss value to the optimizer
    return total_loss


def plot_training_progress(train_losses, test_losses, train_maes, test_maes, steps):
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
    plt.show()
    plt.pause(0.1)
    plt.savefig('training_progress.png', dpi=500, bbox_inches='tight')


if __name__ == "__main__":

    # Specify number of grid points and number of dimensions
    num_grid_pts = 256
    nDim = 2

    # Prepare test data
    grids, axis_points = create_grid(num_grid_pts=num_grid_pts, n_dim=nDim)
    X, Y = grids[0], grids[1]
    x_1, x_2 = axis_points[0], axis_points[1]
    X_u_test, lb, ub = prepare_test_data(X, Y)

    # Analytical solution of the PDE
    n = 1
    m = 1
    k = np.sqrt(np.pi ** 2 * (m ** 2 + n ** 2))
    usol = np.sin(n * X) * np.sin(m * Y)

    # Flatten the solution
    u_true = usol.flatten()[:, None]

    # Number of training points and collocation points
    N_u = 500  # Total number of data points for 'u', used to train the model on boundary conditions
    N_f = 10000  # Total number of collocation points for training the physics-informed part of the model in the domain

    # Prepare training data
    X_f_train_np_array, X_u_train_np_array, u_train_np_array = prepare_training_data(N_u, N_f, lb, ub, usol, X, Y)

    # Convert numpy arrays to PyTorch tensors and move to GPU (if available)
    X_f_train = torch.from_numpy(X_f_train_np_array).float().to(device)  # Collocation points
    X_u_train = torch.from_numpy(X_u_train_np_array).float().to(device)  # Boundary condition points
    u_train = torch.from_numpy(u_train_np_array).float().to(device)  # Boundary condition values
    X_u_test_tensor = torch.from_numpy(X_u_test).float().to(device)  # Test data for boundary conditions
    u = torch.from_numpy(u_true).float().to(device)  # True solution values (ground truth for testing)
    f_hat = torch.zeros(X_f_train.shape[0], 1).to(device)  # Zero tensor for the physics equation residual

    # Neural network architecture - Input layer with 2 nodes, 4 hidden layers with 200 nodes, and
    # an output layer with 1 node
    layers = np.array([2, 200, 200, 200, 200, 1])
    PINN = SequentialModel(layers)

    # Move the model to the GPU (if available)
    PINN.to(device)

    # Print the neural network architecture
    print(PINN)

    # Store the neural network parameters for optimization
    params = list(PINN.parameters())

    # Optimizer setup
    adam_optimizer = optim.Adam(PINN.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-6, amsgrad=False)
    lbfgs_optimizer = optim.LBFGS(PINN.parameters(), max_iter=500, tolerance_grad=1e-5, tolerance_change=1e-9, history_size=100)

    start_time = time.time()  # Start timer

    # Adam optimization loop
    adam_iter = 350

    # Store training progress
    train_losses = []
    test_losses = []
    train_maes = []
    test_maes = []
    steps = []

    for i in range(adam_iter):
        # Forward pass for boundary condition points (train data)
        u_pred_train = PINN.forward(X_u_train)

        # Data loss (MSE between predicted and true boundary condition values)
        data_loss = PINN.loss_function(u_pred_train, u_train)  # Boundary condition loss

        # Physics loss (PDE residuals computed inside the domain)
        physics_loss = PINN.loss_PDE(X_f_train, k)  # Physics loss at collocation points

        # Total loss: sum of data loss (boundary) and physics loss (domain)
        train_loss = data_loss + physics_loss

        # Train MAE (L1 loss between predicted and true solution on boundary points)
        train_mae = PINN.l1loss_function(u_pred_train, u_train)

        # Zero gradient buffers
        adam_optimizer.zero_grad()

        # Backpropagate gradients
        train_loss.backward()

        # Update model parameters
        adam_optimizer.step()

        if i % 10 == 0:
            # Test the model on validation data
            u_pred_test = PINN.forward(X_u_test_tensor)

            # Test loss (MSE between predicted and true solution on test data)
            test_loss = PINN.loss_function(u_pred_test, u)

            # Compute test MAE (L1 loss between predictions and true solution on test data)
            test_mae = PINN.l1loss_function(u_pred_test, u)

            # Print current iteration details
            #print(f"Iteration {i}: Train Loss {train_loss.item()}, Test Loss {test_loss.item()}, "
            #      f"Train MAE {train_mae.item()}, Test MAE {test_mae.item()}")

            # Append the current values to track progress
            steps.append(i)
            train_losses.append(train_loss.item())
            test_losses.append(test_loss.item())
            train_maes.append(train_mae.item())
            test_maes.append(test_mae.item())

        if i % 50 == 0:

            # Predict the solution
            _, u_pred, _ = PINN.test(X_u_test_tensor, u)

            # Visualize the current prediction
            solutionplot(u_pred, usol, x_1, x_2, i)

    # L-BFGS optimization
    lbfgs_optimizer.step(LBFGS_training)

    # Test after L-BFGS optimization
    error_vec, u_pred, _ = PINN.test(X_u_test_tensor, u)
    print(f'L-BFGS Test Error: {error_vec.item()}')

    # Total training time
    elapsed = time.time() - start_time
    print(f'Training time: {elapsed:.2f} seconds')

    # Plot training progress
    plot_training_progress(train_losses, test_losses, train_maes, test_maes, steps)

    # Final test accuracy
    error_vec, u_pred, _ = PINN.test(X_u_test_tensor, u)
    print(f'Test Error: {error_vec:.5f}')
