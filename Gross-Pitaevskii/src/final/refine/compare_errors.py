import torch
import torch.nn as nn
import pickle
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import math
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import LogLocator
from scipy.special import hermite
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['text.usetex'] = False  # Disable LaTeX rendering

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


def plot_combined_potential_comparison(results_dict, save_dir="comparison_results_combined", include_curriculum=True):
    """
    Plot comparison between different PINN methods across multiple potential types.

    Parameters:
    -----------
    results_dict : dict
        Dictionary with potential types as keys and DataFrames as values
    save_dir : str
        Directory to save the plots
    include_curriculum : bool
        If True, includes Curriculum Training in the comparison.
        If False, only compares Vanilla PINN vs PL-PINN
    """
    import seaborn as sns
    os.makedirs(save_dir, exist_ok=True)

    palette = sns.color_palette("tab10")

    if include_curriculum:
        method_styles = {
            'Vanilla PINN': {'color': palette[0], 'marker': 'o'},
            'Curriculum Training': {'color': palette[1], 'marker': 's'},
            'PL-PINN': {'color': palette[2], 'marker': '^'}
        }
        legend_ncol = 3
    else:
        method_styles = {
            'Vanilla PINN': {'color': palette[0], 'marker': 'o'},
            'PL-PINN': {'color': palette[2], 'marker': '^'}
        }
        legend_ncol = 2

    potential_styles = {
        'box': {'linestyle': '-', 'linewidth': 2.5},
        'harmonic': {'linestyle': '--', 'linewidth': 2.5},
        'gravity_well': {'linestyle': '-.', 'linewidth': 2.5}
    }

    potential_labels = {
        'box': 'Box',
        'harmonic': 'Harmonic',
        'gravity_well': 'Gravity Well'
    }

    first_potential = list(results_dict.keys())[0]
    modes = sorted(results_dict[first_potential]['Mode'].unique())

    # Plot 1: Combined Absolute and Relative Error
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax1, ax2 = axes

    legend_elements = []

    for method in method_styles.keys():
        for potential_type, df in results_dict.items():
            method_data = df[df['Method'] == method]

            if method_data.empty:
                continue

            # Absolute Error
            mode_avg_abs = method_data.groupby('Mode')['Abs Error'].mean()
            line1, = ax1.semilogy(
                mode_avg_abs.index,
                mode_avg_abs.values,
                marker=method_styles[method]['marker'],
                color=method_styles[method]['color'],
                linestyle=potential_styles[potential_type]['linestyle'],
                linewidth=potential_styles[potential_type]['linewidth'],
                markersize=7,
                alpha=0.85,
                markeredgecolor='black'
            )

            # Relative Error
            mode_avg_rel = method_data.groupby('Mode')['Rel Error'].mean()
            ax2.semilogy(
                mode_avg_rel.index,
                mode_avg_rel.values,
                marker=method_styles[method]['marker'],
                color=method_styles[method]['color'],
                linestyle=potential_styles[potential_type]['linestyle'],
                linewidth=potential_styles[potential_type]['linewidth'],
                markersize=7,
                alpha=0.85,
                markeredgecolor='black'
            )

            legend_elements.append(plt.Line2D([0], [0],
                                              color=method_styles[method]['color'],
                                              marker=method_styles[method]['marker'],
                                              linestyle=potential_styles[potential_type]['linestyle'],
                                              linewidth=2.5,
                                              markersize=7,
                                              markeredgecolor='black',
                                              label=f"{method} ({potential_labels[potential_type]})"
                                              ))

    for ax, title, ylabel in zip(axes,
                                 ["Absolute Error Comparison", "Relative Error Comparison"],
                                 ["Absolute Error", "Relative Error (%)"]):
        ax.set_xlabel("Mode", fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_title(title, fontsize=18)
        ax.set_xticks(modes)
        ax.grid(True, alpha=0.3, which='both')
        ax.tick_params(labelsize=14)
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto'))

    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.18),
               ncol=legend_ncol, fontsize=12, frameon=True, fancybox=True, shadow=False)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(os.path.join(save_dir, "combined_error_comparison.png"), dpi=600, bbox_inches='tight')
    plt.close()

    potentials = list(results_dict.keys())
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    simple_colors = {
        'Vanilla PINN': '#E74C3C',  # Red
        'Curriculum Training': '#F39C12',  # Orange
        'PL-PINN': '#27AE60'  # Green
    }

    simple_markers = {
        'Vanilla PINN': 'o',
        'Curriculum Training': 's',
        'PL-PINN': '^'
    }

    # Top row: Error vs Mode for each potential
    for col, potential_type in enumerate(potentials):
        df = results_dict[potential_type]
        ax = axes[0, col]

        for method in method_styles.keys():
            method_data = df[df['Method'] == method]
            if not method_data.empty:
                mode_avg = method_data.groupby('Mode')['Abs Error'].mean()
                ax.semilogy(
                    mode_avg.index,
                    mode_avg.values,
                    marker=simple_markers[method],
                    color=simple_colors[method],
                    linewidth=2.5,
                    markersize=8,
                    label=method,
                    alpha=0.9,
                    markeredgecolor='black',
                    markeredgewidth=1
                )

        ax.set_xlabel("Mode", fontsize=14)
        ax.set_ylabel("Absolute Error", fontsize=14)
        ax.set_title(f"{potential_labels[potential_type]} - Error vs Mode at $\eta=0$", fontsize=16)
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=11, loc='best')
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto'))

    # Bottom row: Error vs Interaction Strength
    for col, potential_type in enumerate(potentials):
        df = results_dict[potential_type]
        ax = axes[1, col]

        for method in method_styles.keys():
            # Filter for ground state (Mode 0)
            method_data = df[(df['Method'] == method) & (df['Mode'] == 0)]
            if not method_data.empty:
                gamma_avg = method_data.groupby('Gamma')['Abs Error'].mean()
                ax.semilogy(
                    gamma_avg.index,
                    gamma_avg.values,
                    marker=simple_markers[method],
                    color=simple_colors[method],
                    linewidth=2.5,
                    markersize=8,
                    label=method,
                    alpha=0.9,
                    markeredgecolor='black',
                    markeredgewidth=1
                )

        ax.set_xlabel(r"$\eta$ (Interaction Strength)", fontsize=14)
        ax.set_ylabel("Absolute Error", fontsize=14)
        ax.set_title(f"{potential_labels[potential_type]} - Error vs Interaction Strength", fontsize=16)
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=11, loc='best')
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto'))

    #fig.suptitle("Performance Comparison Across Potential Types", fontsize=20, y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "combined_performance_by_interaction.png"), dpi=600, bbox_inches='tight')
    plt.close()

    # Plot 3: Average Performance Bar Chart
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    methods = list(method_styles.keys())
    potentials = list(potential_styles.keys())
    x = np.arange(len(methods))
    width = 0.25

    for i, metric in enumerate(['Abs Error', 'Rel Error']):
        ax = axes[i]
        for j, pot in enumerate(potentials):
            df = results_dict[pot]
            values = []
            for m in methods:
                method_data = df[df['Method'] == m]
                if not method_data.empty:
                    values.append(method_data[metric].mean())
                else:
                    values.append(np.nan)

            ax.bar(x + j * width, values, width,
                   label=potential_labels[pot],
                   alpha=0.8)

        ax.set_ylabel(f"Average {metric}", fontsize=14)
        ax.set_title(f"Average {metric} by Method and Potential", fontsize=16)
        ax.set_xticks(x + width)
        ax.set_xticklabels(methods, fontsize=12, rotation=15 if len(methods) > 2 else 0,
                           ha='right' if len(methods) > 2 else 'center')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "combined_average_performance_bars.png"), dpi=600, bbox_inches='tight')
    plt.close()

    comparison_type = "All methods" if include_curriculum else "Vanilla PINN vs PL-PINN only"
    print(f"Combined plots saved in: {save_dir}")
    print(f"Comparison type: {comparison_type}")


if __name__ == "__main__":
    # Load results from all three potentials
    results_dict = {
        'box': pd.read_csv("comparison_results_p3_box/raw_comparison_results.csv"),
        'harmonic': pd.read_csv("comparison_results_p3_harmonic/raw_comparison_results.csv"),
        'gravity_well': pd.read_csv("comparison_results_p3_gravity_well/raw_comparison_results.csv")
    }

    # Create plots with all methods (including curriculum learning)
    plot_combined_potential_comparison(
        results_dict,
        save_dir="comparison_results_combined_all_methods",
        include_curriculum=True
    )

    # Create plots with only Vanilla PINN vs PL-PINN
    plot_combined_potential_comparison(
        results_dict,
        save_dir="comparison_results_combined_vanilla_vs_plpinn",
        include_curriculum=False
    )