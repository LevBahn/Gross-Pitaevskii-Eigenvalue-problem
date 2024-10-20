import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0, jn


# Define the neural network
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.hidden1 = nn.Linear(2, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.hidden3 = nn.Linear(128, 128)
        self.output = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.sin(self.hidden1(x))
        x = torch.sin(self.hidden2(x))
        x = torch.sin(self.hidden3(x))
        return self.output(x)


# Analytical solution
def analytical_solution(r, theta, k, n):
    return j0(k * r) * np.cos(n * theta)


# Define the Helmholtz residual
def helmholtz_residual(model, r, theta, k):
    r.requires_grad = True
    theta.requires_grad = True
    u = model(torch.cat((r, theta), dim=1))

    # Calculate gradients
    u_r = torch.autograd.grad(u, r, torch.ones_like(u), create_graph=True)[0]
    u_rr = torch.autograd.grad(u_r, r, torch.ones_like(u_r), create_graph=True)[0]
    u_theta = torch.autograd.grad(u, theta, torch.ones_like(u), create_graph=True)[0]
    u_theta_theta = torch.autograd.grad(u_theta, theta, torch.ones_like(u_theta), create_graph=True)[0]

    # Laplacian in polar coordinates
    laplacian = u_rr + (1 / r) * u_r + (1 / (r ** 2)) * u_theta_theta

    q = k ** 2 * torch.special.bessel_j0(k * r) * torch.cos(n * theta)

    # Helmholtz equation residual
    residual = laplacian + k ** 2 * u - q
    return residual


# Define the loss function
def loss_function(model, r, theta, k):
    res = helmholtz_residual(model, r, theta, k)
    return torch.mean(res ** 2)


# Enhanced training with adaptive sampling
def train_pinn(model, k, n, num_epochs=10000, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        r = torch.rand(1000, 1) * 1.0
        theta = torch.rand(1000, 1) * 2 * np.pi

        optimizer.zero_grad()
        loss = loss_function(model, r, theta, k)
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

            # Visualization of predictions vs analytical solution
            r_test = torch.linspace(0, 1, 100).view(-1, 1)
            theta_test = torch.linspace(0, 2 * np.pi, 100).view(-1, 1)
            r_grid, theta_grid = torch.meshgrid(r_test.squeeze(), theta_test.squeeze())
            inputs = torch.cat((r_grid.flatten().view(-1, 1), theta_grid.flatten().view(-1, 1)), dim=1)

            with torch.no_grad():
                u_pred = model(inputs).numpy().reshape(r_grid.shape)
                u_analytical = analytical_solution(r_grid.numpy(), theta_grid.numpy(), k, n)
                abs_error = np.abs(u_pred - u_analytical)

            # Create subplots for the predicted solution, analytical solution, and absolute error
            plt.figure(figsize=(18, 6))

            # Predicted solution
            plt.subplot(1, 3, 1)
            plt.contourf(r_grid.numpy(), theta_grid.numpy(), u_pred, levels=50)
            plt.colorbar()
            plt.title('Predicted Solution')
            plt.xlabel('r')
            plt.ylabel('theta')

            # Analytical solution
            plt.subplot(1, 3, 2)
            plt.contourf(r_grid.numpy(), theta_grid.numpy(), u_analytical, levels=50)
            plt.colorbar()
            plt.title('Analytical Solution')
            plt.xlabel('r')
            plt.ylabel('theta')

            # Absolute error
            plt.subplot(1, 3, 3)
            plt.contourf(r_grid.numpy(), theta_grid.numpy(), abs_error, levels=50)
            plt.colorbar()
            plt.title('Absolute Error')
            plt.xlabel('r')
            plt.ylabel('theta')

            plt.show()


# Instantiate and train the model
model = PINN()
k = 1.0  # Example wave number
n = 0  # Mode number
train_pinn(model, k, n)
