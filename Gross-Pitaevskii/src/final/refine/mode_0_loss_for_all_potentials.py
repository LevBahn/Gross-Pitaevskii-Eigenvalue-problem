import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Disable LaTeX rendering
mpl.rcParams['text.usetex'] = False

# Set plot parameters
plot_params = {
    "figure.dpi": "300",
    "axes.labelsize": 20,
    "axes.linewidth": 1.5,
    "axes.titlesize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.title_fontsize": 14,
    "legend.fontsize": 16,
    "xtick.major.size": 3.5,
    "xtick.major.width": 1.5,
    "xtick.minor.size": 2.5,
    "xtick.minor.width": 1.5,
    "ytick.major.size": 3.5,
    "ytick.major.width": 1.5,
    "ytick.minor.size": 2.5,
    "ytick.minor.width": 1.5,
}

plt.rcParams.update(plot_params)


def moving_average(values, window_size=10):
    """Apply moving average smoothing to a list of values"""
    if len(values) < window_size:
        return values
    weights = np.ones(window_size) / window_size
    return np.convolve(values, weights, mode='valid')


def load_models(filename, save_dir):
    """Load training results from a file."""
    filepath = os.path.join(save_dir, filename)

    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return None, None, None, None, None

    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    print(f"Successfully loaded: {filepath}")

    return (data['models_state_dicts'],
            data['mu_table'],
            data['training_history'],
            data['constant_history'],
            data['epochs_history'])


def plot_all_potentials_mode0_comparison(potential_data, epochs, p, save_dir="potential_comparison"):
    """
    Creates a comparison plot of Mode 0 training progress across different potential types.
    Similar format to plot_improved_loss_visualization but comparing potentials instead of modes.

    Parameters:
    -----------
    potential_data : dict
        Dictionary with potential types as keys and their training_history as values
        Format: {'Box': training_history, 'Harmonic': training_history, ...}
    epochs : int
        Total number of training epochs
    p : int
        Nonlinearity power parameter
    save_dir : str
        Directory to save the plot
    """
    os.makedirs(save_dir, exist_ok=True)

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Define colors and labels for each potential type
    potential_colors = {
        'Box': '#1f77b4',  # Blue
        'Harmonic': '#ff7f0e',  # Orange
        'Gravity Well': '#2ca02c',  # Green
        'Harmonic (Negative)': '#d62728'  # Red
    }

    # Plot for each potential type
    for potential_name, training_history in potential_data.items():
        if training_history is None:
            print(f"Skipping {potential_name} - no data available")
            continue

        mode = 0  # Focus on mode 0
        gamma = 0.0  # Focus on η=0

        if mode in training_history and gamma in training_history[mode]:
            loss_history = training_history[mode][gamma]['loss']

            # Apply smoothing
            window_size = min(10, len(loss_history) // 5)
            if window_size > 1:
                smooth_loss = moving_average(loss_history, window_size)
                epoch_nums = np.linspace(0, epochs, len(smooth_loss))
                plt.semilogy(epoch_nums, smooth_loss,
                             color=potential_colors[potential_name],
                             linewidth=2.5,
                             label=potential_name)
            else:
                epoch_nums = np.linspace(0, epochs, len(loss_history))
                plt.semilogy(epoch_nums, loss_history,
                             color=potential_colors[potential_name],
                             linewidth=2.5,
                             label=potential_name)

            print(f"{potential_name}: Final loss = {loss_history[-1]:.2e}")
        else:
            print(f"Warning: Mode {mode} or gamma {gamma} not found in {potential_name} training history")

    # Configure plot
    plt.title(r"Training Progress Comparison: Mode 0, $\eta=0$", fontsize=22)
    plt.xlabel("Epochs", fontsize=18)
    plt.ylabel("Loss", fontsize=18)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend(fontsize=16, loc='best')
    plt.tight_layout()

    # Save figure
    output_filename = f"mode0_loss_comparison_all_potentials_p{p}_eta_0.png"
    output_path = os.path.join(save_dir, output_filename)
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    # Common parameters
    p = 3  # Nonlinearity power
    perturb_const = 0.01
    tol = 0.00001
    epochs = 5001

    # Define the four potential types and their file locations
    potential_configs = {
        'Box': {
            'filename': f"my_gpe_models_p{p}_box_pert_const_{perturb_const}_tol_{tol}.pkl",
            'save_dir': f"plots_p{p}_box_paper_test_pert_const_{perturb_const}_with_pretraining"
        },
        'Harmonic': {
            'filename': f"my_gpe_models_p{p}_harmonic_pert_const_{perturb_const}_tol_{tol}_with_pretraining.pkl",
            'save_dir': f"plots_p{p}_harmonic_paper_test_pert_const_{perturb_const}_with_pretraining"
        },
        'Gravity Well': {
            'filename': f"my_gpe_models_p{p}_gravity_well_pert_const_{perturb_const}_tol_{tol}_with_pretraining.pkl",
            'save_dir': f"plots_p{p}_gravity_well_paper_test_pert_const_{perturb_const}_with_pretraining"
        },
        'Harmonic (Negative)': {
            'filename': f"my_gpe_models_p{p}_harmonic_negative_interaction_strength_pert_const_1e-2_tol_{tol}_with_pretraining.pkl",
            'save_dir': f"plots_p{p}_harmonic_negative_interaction_strength_paper_test_with_pretraining"
        }
    }

    print("=" * 80)
    print("Loading training data from all four potential types...")
    print("=" * 80)

    # Load training histories for all potentials
    potential_data = {}
    for potential_name, config in potential_configs.items():
        print(f"\nLoading {potential_name}...")
        _, _, training_history, _, _ = load_models(
            config['filename'],
            config['save_dir']
        )
        potential_data[potential_name] = training_history

    print("\n" + "=" * 80)
    print("Creating comparison plot...")
    print("=" * 80)

    # Create the comparison plot
    output_dir = "potential_comparison_plots"
    plot_all_potentials_mode0_comparison(potential_data, epochs, p, output_dir)

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE!")
    print("=" * 80)
    print(f"Check '{output_dir}/' for the comparison plot")