import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.checkpoint import checkpoint

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) for solving PDEs using Riesz energy minimization.

    Attributes
    ----------
    layers : nn.ModuleList
        List of neural network layers.
    ub : torch.Tensor
        Upper bound for input normalization.
    lb : torch.Tensor
        Lower bound for input normalization.
    adaptive_bc_scale : torch.nn.Parameter
        Learnable scaling factor for boundary condition loss.
    k : torch.nn.Parameter
        Learnable parameter for the wave number.
    """
    def __init__(self, layers, ub, lb):
        """
        Initializes the PINN model with given layer sizes and boundary conditions.

        Parameters
        ----------
        layers : list
            List of integers specifying the number of units in each layer.
        ub : list or numpy.ndarray
            Upper bounds of the input domain for feature normalization.
        lb : list or numpy.ndarray
            Lower bounds of the input domain for feature normalization.
        """
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        self.ub = torch.tensor(ub, dtype=torch.float32, device=device)
        self.lb = torch.tensor(lb, dtype=torch.float32, device=device)
        self.adaptive_bc_scale = nn.Parameter(torch.tensor(10.0, device=device))  # Adaptive weighting for BC loss (lambda)
        self.k = nn.Parameter(torch.tensor(np.pi, device=device))  # Initialize k as a learnable parameter

        # Define network layers
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                self.layers.append(nn.LeakyReLU())

        self.activation = nn.LeakyReLU()
        self.init_weights()

    def init_weights(self):
        """ Initialize weights using Xavier initialization. """
        for m in self.layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        """
        Forward pass through the neural network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor for the neural network.

        Returns
        -------
        torch.Tensor
            Output of the neural network after forward pass.
        """

        # Ensure lb and ub are broadcastable to the shape of x
        lb = self.lb.view(1, -1)  # Ensure lb shape is (1, num_features)
        ub = self.ub.view(1, -1)  # Ensure ub shape is (1, num_features)

        # Normalize the inputs
        x = (x - lb) / (ub - lb)

        # Use gradient checkpointing for layers to save memory
        for i, layer in enumerate(self.layers):
            if i % 2 == 0:  # Apply checkpointing only to Linear layers
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

        return x

    def loss_BC(self, x_bc, y_bc):
        """
        Computes the boundary condition (BC) loss.

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
        u_pred = self.forward(x_bc)
        adaptive_scale = self.adaptive_bc_scale  # Adaptive scaling
        bc_loss = adaptive_scale * torch.mean((u_pred - y_bc) ** 2)
        return bc_loss

    def riesz_loss(self, predictions, inputs):
        """
        Computes the Riesz energy loss for regularization.

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
        if not inputs.requires_grad:
            inputs = inputs.clone().detach().requires_grad_(True)
        gradients = torch.autograd.grad(outputs=predictions, inputs=inputs,
                                        grad_outputs=torch.ones_like(predictions),
                                        create_graph=True, retain_graph=True)[0]
        riesz_energy = torch.sum(gradients ** 2)
        return riesz_energy

    def pde_loss(self, inputs, predictions):
        """
        Computes the Helmholtz PDE loss u_xx + u_yy - k^2 * u = 0.
        Uses the learnable parameter k.

        Parameters
        ----------
        inputs : torch.Tensor
            Input points (x, y).
        predictions : torch.Tensor
            Predicted output from the network.

        Returns
        -------
        torch.Tensor
            Helmholtz PDE loss.
        """
        u_x = torch.autograd.grad(outputs=predictions, inputs=inputs,
                                  grad_outputs=torch.ones_like(predictions),
                                  create_graph=True)[0]

        u_xx = torch.autograd.grad(outputs=u_x[:, 0], inputs=inputs,
                                   grad_outputs=torch.ones_like(u_x[:, 0]),
                                   create_graph=True)[0][:, 0]

        u_yy = torch.autograd.grad(outputs=u_x[:, 1], inputs=inputs,
                                   grad_outputs=torch.ones_like(u_x[:, 1]),
                                   create_graph=True)[0][:, 1]

        # Helmholtz equation u_xx + u_yy - k^2 * u = 0 with learnable k
        pde_residual = u_xx + u_yy - self.k ** 2 * predictions
        pde_loss = torch.mean(pde_residual ** 2)

        return pde_loss

    def loss(self, x_bc, y_bc, x_to_train_f):
        """
        Computes the total loss combining BC loss, PDE loss (Helmholtz), and Riesz loss.

        Parameters
        ----------
        x_bc : torch.Tensor
            Boundary condition input data.
        y_bc : torch.Tensor
            Boundary condition true values.
        x_to_train_f : torch.Tensor
            Input points for PDE training.

        Returns
        -------
        torch.Tensor
            Total loss combining BC, PDE, and Riesz losses.
        """
        loss_u = self.adaptive_bc_scale * self.loss_BC(x_bc, y_bc)  # BC loss
        predictions = self.forward(x_to_train_f)

        # PDE loss (Helmholtz equation) with learnable k
        loss_pde = self.pde_loss(x_to_train_f, predictions)

        # Riesz energy loss for smoothness
        loss_k = self.riesz_loss(predictions, x_to_train_f)

        total_loss = loss_u + loss_pde + loss_k
        return total_loss


# Training function
def train_pinn(model, optimizer, scheduler, x_bc, y_bc, x_to_train_f, epochs):
    """
    Training loop for the PINN model.

    Parameters
    ----------
    model : PINN
        PINN model to be trained.
    optimizer : torch.optim.Optimizer
        Optimizer for training.
    scheduler : torch.optim.lr_scheduler
        Learning rate scheduler.
    x_bc : torch.Tensor
        Boundary condition input data.
    y_bc : torch.Tensor
        Boundary condition output data.
    x_to_train_f : torch.Tensor
        Input points for PDE training.
    epochs : int
        Number of training epochs.
    """

    scaler = torch.amp.GradScaler('cuda')  # Mixed precision training

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):  # Use mixed precision for forward pass
            loss = model.loss(x_bc, y_bc, x_to_train_f)

        scaler.scale(loss).backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step(loss)

        if epoch % 100 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}, Learned Adaptive BC: {model.adaptive_bc_scale.item()}, '
                  f'Learned k: {model.k.item()}')

def visualize_solution(model, x_train, y_train):
    """
    Visualizes the predicted solution from the Physics-Informed Neural Network (PINN).

    The plot shows the network's predicted solution over the 2D spatial domain.

    Parameters
    ----------
    model : PINN
        The trained PINN model used for predictions.
    x_train : torch.Tensor
        Training points used for prediction. Should be a tensor of shape (N, 1).
    y_train : torch.Tensor
        Training points used for prediction. Should be a tensor of shape (N, 1).
    """
    # Flatten the grid of x_train and y_train
    x_flat = x_train.reshape(-1, 1)
    y_flat = y_train.reshape(-1, 1)

    # Generate predictions using the model
    with torch.no_grad():  # Don't track gradients for visualization
        inputs = torch.cat([x_flat, y_flat], dim=1)  # Combine x and y into a 2D tensor
        u_pred = model(inputs)  # Get the model's predictions

    # Detach the predictions and convert to numpy
    x_train = x_train.detach().cpu().numpy()
    y_train = y_train.detach().cpu().numpy()
    u_pred = u_pred.detach().cpu().numpy().reshape(x_train.shape)

    # Create a grid
    x_unique = np.unique(x_train)
    y_unique = np.unique(y_train)
    X, Y = np.meshgrid(x_unique, y_unique)

    # Plotting the predicted solution
    fig, ax = plt.subplots(figsize=(8, 6))

    # Surface plot of predicted solution
    c = ax.pcolor(X, Y, u_pred, cmap='jet')
    fig.colorbar(c, ax=ax)

    ax.set_xlabel(r'$x$ (Spatial Coordinate)', fontsize=12)
    ax.set_ylabel(r'$y$ (Spatial Coordinate)', fontsize=12)

    ax.set_title('Predicted Solution from PINN after Training\n(Prediction over the Spatial Domain)', fontsize=14)
    plt.tight_layout()
    plt.show()


# Model initialization and training
if __name__ == "__main__":

    # Input parameters
    ub = np.array([1, 1])
    lb = np.array([0.0, 0.0])
    layers = [2, 40, 40, 40, 40, 1]  # Neural network layers
    epochs = 1000

    # Training data (boundary condition points)
    x_train = torch.linspace(0.0, 1.0, 100, device=device).view(-1, 1).requires_grad_(True)
    y_train = torch.linspace(0.0, 1.0, 100, device=device).view(-1, 1).requires_grad_(True)
    x_train, y_train = torch.meshgrid(x_train.squeeze(), y_train.squeeze(), indexing='ij')
    x_bc, y_bc = x_train.reshape(-1, 1), y_train.reshape(-1, 1)

    # Model initialization
    model = PINN(layers, ub=ub, lb=lb).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, factor=0.5, verbose=True)

    # Train the model
    train_pinn(model, optimizer, scheduler, x_bc, y_bc, torch.cat([x_bc, y_bc], dim=1), epochs)

    # Visualize solution
    visualize_solution(model, x_train, y_train)
