# These functions involve purely hypotheticcal plots that could be used in a presentation or a paper.

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from matplotlib import colormaps


def pred_hrf_uninformed(seed:(int | bool)=False):
    if seed:
        np.random.seed(seed)
        
    def _hrf_uninformed(t):
        """Hemodynamic response function with prestimulus undershoot"""
        return gamma.pdf(t, 6) - 0.35 * gamma.pdf(t, 16) - 0.1 * gamma.pdf(t, 4)


    # Generate time points
    t = np.linspace(0, 30, 1000)

    # Create a colormap that transitions from red to darkgrey to green
    cmap = LinearSegmentedColormap.from_list(
        "amplitude_to_color", ["grey", "white", "black", "darkgrey"]
    )

    # Plot the HRF waves
    plt.figure(figsize=(10, 6))

    for i in range(6):
        # Generate HRF wave at a random time with a random amplitude
        hrf_wave = (_hrf_uninformed(t - np.random.uniform(-1, 6))) * np.random.uniform(
            0.5, 1.5
        )

        # Plot the HRF wave
        plt.plot(
            t, hrf_wave, label=f"HRF at t={i+1}", color=cmap(i / 6), linewidth=10, alpha=0.8
        )

    plt.xlabel("Time", fontsize=28)
    plt.ylabel("Activation", fontsize=28)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def pred_hrf_informed(seed:(int | bool)=False):
    if seed:
        np.random.seed(seed)
        
    def _hrf_informed(t):
        """Hemodynamic response function with prestimulus undershoot"""
        return gamma.pdf(t, 6) - 0.35 * gamma.pdf(t, 16) - 0.1 * gamma.pdf(t, 4)


    # Generate time points
    t = np.linspace(0, 30, 1000)

    # Create a colormap that transitions from red to darkgrey to green
    cmap = LinearSegmentedColormap.from_list(
        "amplitude_to_color", ["red", "darkgrey", "green"]
    )

    # Generate HRF waves and their peak amplitudes
    hrf_waves = [
        (_hrf_informed(t - np.random.uniform(-1, 6))) * np.random.uniform(0.5, 1.5)
        for _ in range(6)
    ]
    peak_amplitudes = [np.max(wave) for wave in hrf_waves]

    # Normalize the peak amplitudes to the range [0, 1]
    normalized_amplitudes = (peak_amplitudes - np.min(peak_amplitudes)) / (
        np.max(peak_amplitudes) - np.min(peak_amplitudes)
    )

    # Plot the HRF waves
    plt.figure(figsize=(10, 6))

    for i, hrf_wave in enumerate(hrf_waves):
        # Plot the HRF wave
        plt.plot(t, hrf_wave, color=cmap(normalized_amplitudes[i]), linewidth=10, alpha=0.8)

    # Create a legend
    red_patch = mpatches.Patch(color="red", label="Predictable")
    green_patch = mpatches.Patch(color="green", label="Unpredictable")
    plt.legend(handles=[red_patch, green_patch], fontsize=25, loc="upper right")

    plt.xlabel("Time", fontsize=28)
    plt.ylabel("Activation", fontsize=28)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def pred_bar_graph(seed:(int | bool)=False):
    if seed:
        np.random.seed(seed)
        
    # Create a colormap that transitions from green to red
    cmap = LinearSegmentedColormap.from_list("green_to_red", ["green", "black", "red"])

    # Generate random bar heights
    bar_heights = np.random.uniform(0.5, 3, 10)

    # Normalize the bar heights to the range [0, 1]
    normalized_heights = (bar_heights - bar_heights.min()) / (bar_heights.max() - bar_heights.min())

    # Plot the bar graph
    plt.figure(figsize=(10, 6))

    for i in range(8):
        # Plot the bar
        plt.bar(i, bar_heights[i], color=cmap(normalized_heights[i]), alpha=0.8)

    plt.xlabel('Image', fontsize=28)
    plt.ylabel('Predictability', fontsize=28)
    plt.xticks([])
    plt.yticks([])
    plt.show()
