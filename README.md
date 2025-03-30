# Gross Pitaevskii and Helmholtz Equation Solver using PINNs

This repository implements a Physics-Informed Neural Network (PINN) to solve both the Helmholtz equation and the the Gross–Pitaevskii equation. The method combines traditional neural network training with physical constraints enforced by the governing equation.

## Features

- **Physics-Informed Neural Networks (PINNs)**: The neural network is trained not only to minimize boundary condition errors but also to respect the underlying physics of the Helmholtz and Gross–Pitaevskii equation through the PDE loss. The Gross–Pitaevskii PINN class contains three additional loss terms including the Riesz energy loss, normalization loss, and symmetry loss.
- **Adaptive Optimizers**: The code uses both the Adam and L-BFGS optimizers to effectively minimize the loss and improve convergence. It also contains [Distributed Shampoo optimizer](https://github.com/facebookresearch/optimizers/tree/main) for better performance.
- **Tanh Activation Function**: The model employs the hyperbolic tangent (Tanh) activation function, which maps input values to a range between -1 and 1. This zero-centered output can facilitate faster convergence during training by promoting more balanced gradients.
- **Custom Loss Functions**: The loss function combines both boundary condition losses and PDE residual losses to drive the network training.
- **Data Scaling**: Input features are scaled between 0 and 1 to improve convergence and stability during training.

## Prerequisites

The following Python libraries are required:

- `torch` (PyTorch for building and training neural networks)
- `numpy` (for matrix operations)
- `pyDOE` (for Latin Hypercube Sampling of collocation points)
- `matplotlib` (for visualizing the results)
- `scipy` (for special functions)

Install these dependencies using:

```bash
pip install torch numpy pyDOE matplotlib scipy
```

The PyTorch Distributed Shampoo optimizer can be installed by

```bash
pip install pytorch-optimizer
git clone https://github.com/facebookresearch/optimizers.git
cd optimizers
pip install .
cd ..
```

## How It Works

The network architecture consists of fully connected layers with customizable sizes. Both boundary condition data and collocation points generated via Latin Hypercube Sampling are fed into the network.

1. **Training Data**: The network is trained on boundary conditions and a set of random collocation points within the domain.
2. **Loss Function**: The loss function is a combination of boundary condition loss and the PDE loss. The Gross–Pitaevskii equation has five loss terms:
   - **Boundary Condition Loss**: Mean squared error between predicted and known boundary values.
   - **PDE Loss**: Residual of the Gross–Pitaevskii equation computed via automatic differentiation.
   - **Riesz Energy Loss**: The energy loss from the energy functional (for regularization).
   - **Normalization Loss**: The $L^2$ norm of the solution should be equal to $1$.
   - **Symmetry Loss**: A symmetry potential should generate a symmetric solution.
3. **Optimizers**: Training is initially performed using the Adam optimizer, followed by fine-tuning with the L-BFGS optimizer and variants of the Shampoo optimizers for higher precision.

## Tanh Activation Function

The Tanh function is used as the activation function, which is particularly useful for nonlinear PDEs.

## Usage

1. **Training**: Modify the parameters such as `num_grid_pts`, `N_u`, `N_f`, and network architecture (`layers`) in the code as needed.
2. **Run the Solver**:

```bash
python helmholtz.py
```
3. **Results**: The solution will be visualized in terms of:
   - Ground truth solution
   - Predicted solution
   - Absolute error between the two

The final test error will be printed and plotted.

## Example Output

The solver will generate a plot showing:

- **Ground Truth Solution**: The known solution for the Helmholtz equation.
- **Predicted Solution**: The solution predicted by the neural network.
- **Absolute Error**: The difference between the true solution and the predicted solution.

## Future Improvements

- Add more flexible handling of boundary conditions.
- Extend the solver to more complex forms of different PDEs.
- Explore the use of different activation functions and optimizers for further improvement.
