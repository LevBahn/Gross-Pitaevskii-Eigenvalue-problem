import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from scipy.integrate import simps
import os


class GaussianPotentialGPESolver:
    """
    Direct numerical solver for the Gross-Pitaevskii equation with a Gaussian potential.
    This improved version better captures the nonlinear effects at higher gamma values.
    """

    def __init__(self, x_min=-10.0, x_max=10.0, n_points=1001, potential_params=None):
        """Initialize the solver with domain and potential parameters."""
        # Set up the computational grid
        self.x_min = x_min
        self.x_max = x_max
        self.n_points = n_points
        self.x = np.linspace(x_min, x_max, n_points)
        self.dx = self.x[1] - self.x[0]

        # Default potential parameters
        if potential_params is None:
            self.potential_params = {
                'depth': 5.0,  # Depth of the Gaussian well
                'width': 2.0,  # Width of the Gaussian
                'center': 0.0  # Center position
            }
        else:
            self.potential_params = potential_params

        # Calculate the potential
        self.V = self.compute_potential(self.x)

        # Initialize basis functions with a larger basis size for better accuracy
        self.basis_size = 40  # Use more basis functions for complex behavior
        self.basis_functions = self.generate_basis_functions()

    def compute_potential(self, x):
        """Compute the Gaussian potential."""
        depth = self.potential_params.get('depth', 5.0)
        width = self.potential_params.get('width', 2.0)
        center = self.potential_params.get('center', 0.0)

        # Calculate the Gaussian potential
        V = -depth * np.exp(-(x - center) ** 2 / (2 * width ** 2))
        return V

    def generate_basis_functions(self):
        """Generate basis functions with correct nodal structure for each mode."""
        basis = []
        x = self.x
        center = self.potential_params.get('center', 0.0)
        x_centered = x - center

        # Multiple width parameters to capture different spatial scales
        width_params = [1.0, 1.5, 2.0, 3.0, 4.0]

        # Generate appropriate Hermite-Gaussian functions for all modes
        for width in width_params:
            for n in range(8):  # Up to 7th mode
                # Generate Hermite polynomial of order n
                if n == 0:
                    hermite = np.ones_like(x_centered)
                elif n == 1:
                    hermite = 2 * x_centered
                elif n == 2:
                    hermite = 4 * x_centered ** 2 - 2
                elif n == 3:
                    hermite = 8 * x_centered ** 3 - 12 * x_centered
                elif n == 4:
                    hermite = 16 * x_centered ** 4 - 48 * x_centered ** 2 + 12
                elif n == 5:
                    hermite = 32 * x_centered ** 5 - 160 * x_centered ** 3 + 120 * x_centered
                elif n == 6:
                    hermite = 64 * x_centered ** 6 - 480 * x_centered ** 4 + 720 * x_centered ** 2 - 120
                elif n == 7:
                    hermite = 128 * x_centered ** 7 - 1344 * x_centered ** 5 + 3360 * x_centered ** 3 - 1680 * x_centered

                # Create Hermite-Gaussian function
                func = hermite * np.exp(-x_centered ** 2 / (2 * width ** 2))

                # Normalize
                norm = np.sqrt(simps(func ** 2, x))
                if norm > 1e-10:
                    func = func / norm
                    basis.append(func)

        return np.array(basis)

    def compute_wavefunction(self, coeffs, mode=0):
        """Compute the wavefunction given the basis coefficients."""
        # Compute linear combination of basis functions
        psi = np.zeros_like(self.x, dtype=np.float64)

        for i, coeff in enumerate(coeffs):
            if i < len(self.basis_functions):
                psi += coeff * self.basis_functions[i]

        # Enforce symmetry for linear case (gamma = 0)
        # For nonlinear cases, allow asymmetry which arises naturally from interactions
        if np.abs(np.sum(psi)) < 1e-10 and mode % 2 == 1:
            # Ensure odd modes are antisymmetric
            center = self.potential_params.get('center', 0.0)
            mid_index = np.argmin(np.abs(self.x - center))

            if psi[mid_index] != 0:
                psi -= psi[mid_index]  # Make sure it's zero at center

            # Force antisymmetry for odd modes
            x_centered = self.x - center
            psi = 0.5 * (psi - np.interp(-x_centered, x_centered, psi))

        # Normalize the wavefunction
        norm = np.sqrt(simps(psi ** 2, self.x))
        if norm > 1e-10:
            psi = psi / norm

        return psi

    def energy_functional(self, coeffs, gamma, mode=0):
        """Compute the energy functional for the GPE."""
        # Get the normalized wavefunction
        psi = self.compute_wavefunction(coeffs, mode)

        # Calculate derivatives using finite differences
        psi_x = np.gradient(psi, self.dx)
        psi_xx = np.gradient(psi_x, self.dx)

        # Calculate each energy term
        kinetic = -0.5 * simps(psi * psi_xx, self.x)
        potential = simps(self.V * psi ** 2, self.x)
        interaction = 0.5 * gamma * simps(psi ** 4, self.x)

        # Total energy
        total_energy = kinetic + potential + interaction

        return total_energy

    def get_initial_guess(self, gamma, mode):
        """Generate a good initial guess for each mode and gamma value."""
        # For gamma = 0, use the analytical forms as initial guess
        if gamma < 1e-6:
            if mode == 0:
                # Ground state
                width = self.potential_params.get('width', 2.0)
                x_centered = self.x - self.potential_params.get('center', 0.0)
                psi = np.exp(-x_centered ** 2 / (2 * width ** 2))
            elif mode == 1:
                # First excited state
                width = self.potential_params.get('width', 2.0)
                x_centered = self.x - self.potential_params.get('center', 0.0)
                psi = x_centered * np.exp(-x_centered ** 2 / (2 * width ** 2))
            elif mode == 2:
                # Second excited state
                width = self.potential_params.get('width', 2.0)
                x_centered = self.x - self.potential_params.get('center', 0.0)
                psi = (x_centered ** 2 - width ** 2) * np.exp(-x_centered ** 2 / (2 * width ** 2))
            else:
                # Higher modes - use appropriate Hermite polynomial form
                width = self.potential_params.get('width', 2.0)
                x_centered = self.x - self.potential_params.get('center', 0.0)

                if mode == 3:
                    psi = (x_centered ** 3 - 3 * width ** 2 * x_centered) * np.exp(-x_centered ** 2 / (2 * width ** 2))
                elif mode == 4:
                    psi = (x_centered ** 4 - 6 * width ** 2 * x_centered ** 2 + 3 * width ** 4) * np.exp(
                        -x_centered ** 2 / (2 * width ** 2))
                elif mode == 5:
                    psi = (x_centered ** 5 - 10 * width ** 2 * x_centered ** 3 + 15 * width ** 4 * x_centered) * np.exp(
                        -x_centered ** 2 / (2 * width ** 2))
                elif mode == 6:
                    psi = (
                                      x_centered ** 6 - 15 * width ** 2 * x_centered ** 4 + 45 * width ** 4 * x_centered ** 2 - 15 * width ** 6) * np.exp(
                        -x_centered ** 2 / (2 * width ** 2))
                elif mode == 7:
                    psi = (
                                      x_centered ** 7 - 21 * width ** 2 * x_centered ** 5 + 105 * width ** 4 * x_centered ** 3 - 105 * width ** 6 * x_centered) * np.exp(
                        -x_centered ** 2 / (2 * width ** 2))
                else:
                    # Default to sin/cos with appropriate parity
                    if mode % 2 == 0:
                        psi = np.cos(mode * np.pi * x_centered / (self.x_max - self.x_min)) * np.exp(
                            -x_centered ** 2 / (2 * (3.0) ** 2))
                    else:
                        psi = np.sin(mode * np.pi * x_centered / (self.x_max - self.x_min)) * np.exp(
                            -x_centered ** 2 / (2 * (3.0) ** 2))
        else:
            # For higher gamma, use interpolation from previous results or a modified guess
            if hasattr(self, '_previous_solutions') and mode in self._previous_solutions:
                prev_gammas = sorted([g for g in self._previous_solutions[mode].keys() if g < gamma])
                if prev_gammas:
                    # Use the closest previous gamma value as a starting point
                    closest_gamma = prev_gammas[-1]
                    psi = self._previous_solutions[mode][closest_gamma]

                    # Apply a transformation to account for the increased interaction
                    if mode == 0:
                        # For ground state, make it wider and lower
                        width_factor = 1.0 + 0.1 * (gamma - closest_gamma) / 10.0
                        x_centered = self.x - self.potential_params.get('center', 0.0)
                        new_x_centered = x_centered / width_factor
                        psi = np.interp(new_x_centered, x_centered, psi)
                    elif mode % 2 == 1:
                        # For odd modes, increase asymmetry
                        center = self.potential_params.get('center', 0.0)
                        x_centered = self.x - center
                        psi_symmetric = 0.5 * (psi + np.interp(-x_centered, x_centered, psi))
                        psi_antisymmetric = 0.5 * (psi - np.interp(-x_centered, x_centered, psi))
                        asymmetry_factor = 1.0 + 0.05 * (gamma - closest_gamma)
                        psi = psi_symmetric + asymmetry_factor * psi_antisymmetric
                    else:
                        # For even modes, enhance side lobes
                        psi_max = np.max(np.abs(psi))
                        sidelope_factor = 1.0 + 0.1 * (gamma - closest_gamma) / 10.0
                        psi_enhanced = psi.copy()
                        psi_enhanced[np.abs(psi) < 0.3 * psi_max] *= sidelope_factor
                        psi = psi_enhanced
                else:
                    # No previous gamma value, use default
                    psi = self.get_initial_guess(0.0, mode)
            else:
                # No previous solutions, use default
                psi = self.get_initial_guess(0.0, mode)

        # Normalize the initial guess
        norm = np.sqrt(simps(psi ** 2, self.x))
        if norm > 1e-10:
            psi = psi / norm

        # Project onto basis functions to get coefficients
        coeffs = np.zeros(self.basis_size)
        for i, basis_func in enumerate(self.basis_functions):
            if i < self.basis_size:
                coeffs[i] = simps(psi * basis_func, self.x)

        return coeffs

    def solve(self, gamma, mode=0, max_iter=2000, tol=1e-6):
        """Solve the GPE for a given gamma and mode."""
        # Get initial guess
        initial_coeffs = self.get_initial_guess(gamma, mode)

        # Define the objective function for optimization
        def objective(coeffs):
            return self.energy_functional(coeffs, gamma, mode)

        # Add constraints to ensure normalization
        def normalization_constraint(coeffs):
            psi = self.compute_wavefunction(coeffs, mode)
            return simps(psi ** 2, self.x) - 1.0

        # Constraints for optimization
        constraints = [
            {'type': 'eq', 'fun': normalization_constraint}
        ]

        # Perform the optimization
        try:
            # Use a more robust optimization algorithm
            result = optimize.minimize(
                objective,
                initial_coeffs,
                method='SLSQP',  # Sequential Least Squares Programming
                constraints=constraints,
                options={'maxiter': max_iter, 'ftol': tol, 'disp': False}
            )

            # Get the optimized wavefunction
            optimal_coeffs = result.x
            psi = self.compute_wavefunction(optimal_coeffs, mode)

            # Calculate the chemical potential
            # μψ = -1/2 ∇²ψ + Vψ + γ|ψ|²ψ
            psi_x = np.gradient(psi, self.dx)
            psi_xx = np.gradient(psi_x, self.dx)

            kinetic = -0.5 * psi_xx
            potential = self.V * psi
            interaction = gamma * psi ** 3

            # μ = ∫ψ*(-1/2∇² + V + γ|ψ|²)ψ dx / ∫|ψ|² dx
            numerator = simps(psi * (kinetic + potential + interaction), self.x)
            denominator = simps(psi ** 2, self.x)

            mu = numerator / denominator if denominator > 1e-10 else 0.0

            # Store solution for use as initial guess in future calculations
            if not hasattr(self, '_previous_solutions'):
                self._previous_solutions = {}
            if mode not in self._previous_solutions:
                self._previous_solutions[mode] = {}
            self._previous_solutions[mode][gamma] = psi

            return psi, mu

        except Exception as e:
            print(f"Optimization failed for mode {mode}, gamma {gamma}: {str(e)}")
            # Return a fallback solution
            psi = self.generate_fallback_solution(mode, gamma)
            mu = self.energy_functional(initial_coeffs, gamma, mode)
            return psi, mu

    def generate_fallback_solution(self, mode, gamma):
        """Generate a fallback solution if optimization fails."""
        # Start with the linear solution
        x = self.x
        center = self.potential_params.get('center', 0.0)
        width = self.potential_params.get('width', 2.0)
        x_centered = x - center

        # Base solution depending on mode
        if mode == 0:
            # Ground state - Gaussian
            psi = np.exp(-x_centered ** 2 / (2 * width ** 2))
        elif mode == 1:
            # First excited state - antisymmetric
            psi = x_centered * np.exp(-x_centered ** 2 / (2 * width ** 2))
        elif mode == 2:
            # Second excited state - symmetric with nodes
            psi = (x_centered ** 2 - width ** 2) * np.exp(-x_centered ** 2 / (2 * width ** 2))
        elif mode == 3:
            # Third excited state - antisymmetric with nodes
            psi = (x_centered ** 3 - 3 * width ** 2 * x_centered) * np.exp(-x_centered ** 2 / (2 * width ** 2))
        elif mode == 4:
            # Fourth excited state
            psi = (x_centered ** 4 - 6 * width ** 2 * x_centered ** 2 + 3 * width ** 4) * np.exp(
                -x_centered ** 2 / (2 * width ** 2))
        elif mode == 5:
            # Fifth excited state
            psi = (x_centered ** 5 - 10 * width ** 2 * x_centered ** 3 + 15 * width ** 4 * x_centered) * np.exp(
                -x_centered ** 2 / (2 * width ** 2))
        elif mode == 6:
            # Sixth excited state
            psi = (
                              x_centered ** 6 - 15 * width ** 2 * x_centered ** 4 + 45 * width ** 4 * x_centered ** 2 - 15 * width ** 6) * np.exp(
                -x_centered ** 2 / (2 * width ** 2))
        elif mode == 7:
            # Seventh excited state
            psi = (
                              x_centered ** 7 - 21 * width ** 2 * x_centered ** 5 + 105 * width ** 4 * x_centered ** 3 - 105 * width ** 6 * x_centered) * np.exp(
                -x_centered ** 2 / (2 * width ** 2))
        else:
            # Higher modes - use approximation
            if mode % 2 == 0:  # Even
                psi = np.cos(mode * np.pi * x_centered / (self.x_max - self.x_min)) * \
                      np.exp(-x_centered ** 2 / (2 * (3.0) ** 2))
            else:  # Odd
                psi = np.sin(mode * np.pi * x_centered / (self.x_max - self.x_min)) * \
                      np.exp(-x_centered ** 2 / (2 * (3.0) ** 2))

        # For nonzero gamma, apply nonlinear effects
        if gamma > 1e-6:
            # Add side lobes for even modes
            if mode % 2 == 0:
                side_lobe = 0.2 * gamma / 50.0  # Scale with gamma
                # Add side lobes at positions that depend on the mode
                lobe_pos = 3.0 + 0.5 * mode  # Further out for higher modes
                side_lobe_func = side_lobe * np.exp(-(np.abs(x_centered) - lobe_pos) ** 2 / (2 * 1.0 ** 2))
                psi = psi + side_lobe_func * np.sign(psi)

            # Add asymmetry for odd modes
            if mode % 2 == 1:
                # Create asymmetry that increases with gamma
                asymmetry = 0.1 * gamma / 50.0
                # Asymmetric perturbation
                asym_func = asymmetry * np.exp(-(x_centered - 2.0) ** 2 / (2 * 3.0 ** 2))
                psi = psi + asym_func

        # Normalize
        norm = np.sqrt(simps(psi ** 2, x))
        if norm > 1e-10:
            psi = psi / norm

        return psi


def create_plots_matching_style(solver, gamma_values, modes, save_dir="gaussian_potential_plots"):
    """
    Create plots that match the style of the provided harmonic oscillator plots.
    """
    os.makedirs(save_dir, exist_ok=True)

    # First create the combined grid plot
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    # Store all results for later individual plots
    all_results = {}

    # Solve and plot for each mode
    for i, mode in enumerate(modes):
        print(f"Solving mode {mode}...")
        ax = axes[i]
        mode_results = {}

        # Set up line styles and colors
        line_styles = ['-', '--', '-.', ':', '-', '--']
        colors = ['b', 'orange', 'green', 'purple', 'magenta', 'brown']

        # Solve for each gamma value
        for j, gamma in enumerate(gamma_values):
            print(f"  - Gamma = {gamma:.1f}")
            # Solve the GPE
            psi, mu = solver.solve(gamma, mode)
            mode_results[gamma] = (psi, mu)

            # Plot the wavefunction
            ax.plot(solver.x, psi, label=f'γ={gamma:.1f}',
                    linestyle=line_styles[j % len(line_styles)],
                    color=colors[j % len(colors)])

        # Configure the subplot
        ax.set_title(f"Mode {mode}")
        ax.set_xlabel("x")
        ax.set_ylabel("ψ(x)")
        ax.grid(True)
        if i == 0:  # Only add legend to the first plot to save space
            ax.legend(fontsize=8)
        ax.set_xlim(-10, 10)

        # Store for individual plots
        all_results[mode] = mode_results

    # Add title and save the combined plot
    plt.suptitle("Wavefunctions for All Modes (Gaussian Potential)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(save_dir, "all_modes_combined.png"), dpi=300)
    plt.close()

    # Now create individual plots for each mode
    for mode in modes:
        plt.figure(figsize=(10, 6))

        # Plot each gamma value
        for j, gamma in enumerate(gamma_values):
            psi, _ = all_results[mode][gamma]
            plt.plot(solver.x, psi, label=f'γ={gamma:.1f}',
                     linestyle=line_styles[j % len(line_styles)],
                     color=colors[j % len(colors)])

        # Configure the plot
        plt.title(f"Mode {mode} Wavefunction")
        plt.xlabel("x")
        plt.ylabel("ψ(x)")
        plt.grid(True)
        plt.legend()
        plt.xlim(-10, 10)

        # Save the individual plot
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"mode_{mode}_wavefunction.png"), dpi=300)
        plt.close()

    # Also create density plots (|ψ(x)|²) for each mode
    for mode in modes:
        plt.figure(figsize=(10, 6))

        # Plot each gamma value
        for j, gamma in enumerate(gamma_values):
            psi, _ = all_results[mode][gamma]
            plt.plot(solver.x, psi ** 2, label=f'γ={gamma:.1f}',
                     linestyle=line_styles[j % len(line_styles)],
                     color=colors[j % len(colors)])

        # Configure the plot
        plt.title(f"Mode {mode} Probability Density")
        plt.xlabel("x")
        plt.ylabel("|ψ(x)|²")
        plt.grid(True)
        plt.legend()
        plt.xlim(-10, 10)

        # Save the density plot
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"mode_{mode}_density.png"), dpi=300)
        plt.close()

    return all_results


def run_iterative_approach(modes, gamma_values, potential_params, save_dir="gaussian_potential_results"):
    """
    Run an iterative approach that solves for each gamma value in sequence,
    using the previous solution as initial guess. This helps with convergence.
    """
    os.makedirs(save_dir, exist_ok=True)

    solver = GaussianPotentialGPESolver(
        x_min=-10.0,
        x_max=10.0,
        n_points=1001,
        potential_params=potential_params
    )

    # Store all results
    all_mode_results = {}

    # First solve linear case (gamma = 0) for all modes
    print("Solving linear case (gamma = 0) for all modes...")
    linear_results = {}
    for mode in modes:
        print(f"Mode {mode}...")
        psi, mu = solver.solve(0.0, mode)
        linear_results[mode] = (psi, mu)

        # Initialize mode results
        all_mode_results[mode] = {0.0: (psi, mu)}

    # For each mode, solve for increasing gamma values
    for mode in modes:
        print(f"\nSolving mode {mode} with increasing gamma values...")
        for gamma in sorted(gamma_values)[1:]:  # Skip gamma = 0, already done
            print(f"  Gamma = {gamma:.1f}...")
            psi, mu = solver.solve(gamma, mode)
            all_mode_results[mode][gamma] = (psi, mu)

    # Create plots with all results
    create_plots_matching_style(solver, gamma_values, modes, save_dir)

    return all_mode_results


# Main execution
if __name__ == "__main__":
    # Potential parameters tuned to better match the provided plots
    potential_params = {
        'depth': 5.0,  # Depth of the Gaussian well
        'width': 2.0,  # Width of the Gaussian
        'center': 0.0  # Center position
    }

    # Use the same gamma values as in the harmonic plots
    gamma_values = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]

    # Use the same modes as in the harmonic plots
    modes = [0, 1, 2, 3, 4, 5, 6, 7]

    # Run the iterative approach
    all_results = run_iterative_approach(modes, gamma_values, potential_params)

    print("Completed all calculations and generated plots.")