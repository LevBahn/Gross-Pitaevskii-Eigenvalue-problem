import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define the PINN model
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        input_dim = 2
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                self.layers.append(nn.Tanh())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# Define the Helmholtz loss function
def helmholtz_loss(model, x, y, k):
    u = model(torch.cat([x, y], dim=1))
    u_x = torch.autograd.grad(outputs=u, inputs=x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(outputs=u, inputs=y, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    u_xx = torch.autograd.grad(outputs=u_x, inputs=x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(outputs=u_y, inputs=y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]

    laplacian = u_xx + u_yy
    return torch.mean((laplacian + k ** 2 * u) ** 2)


# Training the PINN
def train_pinn(model, optimizer, x_train, y_train, k, epochs):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = helmholtz_loss(model, x_train, y_train, k)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')


def visualize_solution(u_pred, x_train, y_train, u_truth):
    """
    Plots the ground truth solution, predicted solution, and absolute error between them.

    Parameters
    ----------
    u_pred : torch.Tensor
        Predicted solution values from the model. Should be a tensor of shape (N, 1).
    x_train : torch.Tensor
        Training points used for prediction. Should be a tensor of shape (N, 1).
    y_train : torch.Tensor
        Training points used for prediction. Should be a tensor of shape (N, 1).
    u_truth : numpy.ndarray
        Ground truth solution values on a grid. Should be a 2D numpy array.
    """
    # Convert tensors to numpy arrays
    x_train = x_train.detach().numpy().reshape(-1)
    y_train = y_train.detach().numpy().reshape(-1)

    # Create a grid for plotting
    x_unique = np.unique(x_train)
    y_unique = np.unique(y_train)
    X, Y = np.meshgrid(x_unique, y_unique)

    # Interpolate predicted values onto the grid
    U_pred_grid = np.zeros_like(u_truth)
    for i, x in enumerate(x_unique):
        for j, y in enumerate(y_unique):
            mask = (x_train == x) & (y_train == y)
            U_pred_grid[j, i] = np.mean(u_pred[mask])

    # Absolute error
    error = np.abs(u_truth - U_pred_grid)

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Ground truth solution
    c1 = axs[0].pcolor(X, Y, u_truth, cmap='jet')
    fig.colorbar(c1, ax=axs[0])
    axs[0].set_xlabel(r'$x$')
    axs[0].set_ylabel(r'$y$')
    axs[0].set_title('Ground Truth Solution')

    # Predicted solution
    c2 = axs[1].pcolor(X, Y, U_pred_grid, cmap='jet')
    fig.colorbar(c2, ax=axs[1])
    axs[1].set_xlabel(r'$x$')
    axs[1].set_ylabel(r'$y$')
    axs[1].set_title('Predicted Solution')

    # Absolute error
    c3 = axs[2].pcolor(X, Y, error, cmap='jet')
    fig.colorbar(c3, ax=axs[2])
    axs[2].set_xlabel(r'$x$')
    axs[2].set_ylabel(r'$y$')
    axs[2].set_title('Absolute Error')

    plt.tight_layout()
    plt.savefig('Helmholtz_solution_comparison.png', dpi=500, bbox_inches='tight')
    plt.show()


# Main function
def main():
    # Define parameters
    layers = [2, 20, 20, 20, 20, 1]  # Input layer, hidden layers, output layer
    k = 2 * np.pi  # Wavenumber
    epochs = 1000

    # Generate training data (e.g., uniform grid in the domain)
    x_train = torch.linspace(0.0, 1.0, 200).view(-1, 1).requires_grad_(True)
    y_train = torch.linspace(0.0, 1.0, 200).view(-1, 1).requires_grad_(True)
    x_train, y_train = torch.meshgrid(x_train.squeeze(), y_train.squeeze())
    x_train = x_train.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)

    # Initialize model, optimizer
    model = PINN(layers)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train the PINN
    train_pinn(model, optimizer, x_train, y_train, k, epochs)

    # Generate predictions
    model.eval()
    with torch.no_grad():
        u_pred = model(torch.cat([x_train, y_train], dim=1)).reshape(-1).numpy()

    # Generate or load the ground truth solution
    x_grid = x_train.detach().numpy().reshape(-1)
    y_grid = y_train.detach().numpy().reshape(-1)
    X_unique = np.unique(x_grid)
    Y_unique = np.unique(y_grid)
    X, Y = np.meshgrid(X_unique, Y_unique)
    u_truth = np.sin(np.pi * X) * np.sin(np.pi * Y)  # Ground truth

    # Visualize solution
    visualize_solution(u_pred, x_train, y_train, u_truth)

if __name__ == "__main__":
    main()
