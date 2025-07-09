import numpy as np
import os
from numpy import asarray as arr
import matplotlib.pyplot as plt
import seaborn as sns
from components.general_functions import poissonlike_noise
from mytools.my_seaborn import load_seaborn_prefs
from mytools.symbols import model_symbols, misc_symbols, lowercase_greek_alphabet, sub


# Plot singular tuning curve centred closest to 0
def plot_single_tuning(popcode, pref_ori=0, save_dir="", dpi=200,
                       custom_rc=None, figdims=(8, 4)):
    # Plot settings
    custom_rc_plot = {  # default values
        'axes.linewidth': 1,
        'xtick.major.width': 1, 'ytick.major.width': 1,
        'errorbar.capsize': 0.04, 'lines.linewidth': 2.4,
        'xtick.labelsize': 18, 'ytick.labelsize': 20,
        'axes.labelsize': 24, 'font.family': 'sans-serif'
    }
    if custom_rc is not None:  # replace default values if additional given
        for key in custom_rc:
            custom_rc_plot[key] = custom_rc[key]
    load_seaborn_prefs(custom_rc=custom_rc_plot)

    # Symbols
    misc_sym = misc_symbols()
    deg = misc_sym['deg']

    # Define plot parameters
    og_tunings = popcode.all_tunings
    noise = poissonlike_noise(og_tunings, scalar=popcode.param_distribs['noise_scale'])
    noisy_tunings = og_tunings + noise

    # Extract single tuning curve with preferred ori closest to target ori
    ori_idx = np.argmin(abs(popcode.all_prefs - pref_ori))
    tuning = og_tunings[ori_idx, :]
    noisy_tuning = noisy_tunings[ori_idx, :]

    # Plot noiseless
    x = popcode.orispace.tiling
    fig_tuning, ax_tuning = plt.subplots(figsize=figdims)
    ax_tuning.plot(x, tuning)

    # Plot noisy
    fig_noisy, ax_noisy = plt.subplots(figsize=figdims)
    ax_noisy.plot(x, noisy_tuning)

    for ax, fig, label in zip([ax_tuning, ax_noisy], [fig_tuning, fig_noisy], ["tuning", "noisy"]):

        # Axes settings
        ax.set(xlim=(-90, 90), xticks=[-90, -45, 0, 45, 90],
               ylim=(0, 90), yticks=[])
        # Leave room for oriented lines to be used
        ax.set_xlabel(f"Orientation ({deg})", labelpad=30, fontsize=20)
        ax.set_ylabel(f"Firing Rate\n(spikes / sec)", labelpad=12, fontsize=20)

        # Figure settings
        fig.set_dpi(dpi)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"singular_pref{pref_ori}_{label}.png"))


def plot_wrapped_tunings(popcode, legend=True, save_dir='', save_name='',
                         dpi=450, figdims=(8, 4), CvsO_colours=False,
                         custom_rc=None):
    # Params
    og_tunings = popcode.all_tunings
    noise = poissonlike_noise(og_tunings, scalar=popcode.param_distribs['noise_scale'])
    tunings = og_tunings + noise
    tiling = popcode.orispace.tiling

    # Plot settings
    custom_rc_plot = {  # default values
        'axes.linewidth': 1,
        'xtick.major.width': 1, 'ytick.major.width': 1,
        'errorbar.capsize': 0.04, 'lines.linewidth': 2.4,
        'xtick.labelsize': 18, 'ytick.labelsize': 20,
        'axes.labelsize': 24, 'font.family': 'sans-serif'
    }
    if custom_rc is not None:  # replace default values if additional given
        for key in custom_rc:
            custom_rc_plot[key] = custom_rc[key]
    load_seaborn_prefs(custom_rc=custom_rc_plot)

    # Symbols
    symbols = model_symbols()
    misc_sym = misc_symbols()
    low_sym = lowercase_greek_alphabet()
    deg = misc_sym['deg']

    # Step 1: Get preferred orientation index for each neuron
    preferred_idx = og_tunings.argmax(1)  # shape: (n_neurons,)
    preferred_oris = tiling[preferred_idx]  # actual orientation values in degrees

    # Step 2: Define cardinal check (modulo 180 for circular wraparound)
    def is_cardinal(ori, tol=22.5):
        ori_mod = abs(ori) % 180
        ori_offset = [abs(ori_mod - base) for base in [0, 90]]
        return any(_i <= tol for _i in ori_offset)

    # Step 3: Label each neuron's tuning
    ori_labels = np.array(['cardinal' if is_cardinal(ori) else 'oblique' for ori in preferred_oris])

    # Step 4: Generate colour map for plots
    palette = sns.color_palette('colorblind')
    # Mapping colour based on classification
    if CvsO_colours is True:
        colour_map = {'cardinal': palette[0], 'oblique': palette[1]}
        colours = [colour_map[label] for label in ori_labels]
    else:
        colours = sns.color_palette('deep', len(preferred_oris))

    # Step 5: Plot all tunings at once (lines as columns in .T)
    fig_line, ax_line = plt.subplots(figsize=figdims)
    for tuning_curve, colour in zip(tunings, colours):
        ax_line.plot(tiling, tuning_curve, color=colour, alpha=1)

    # LEGEND
    # Plot dummy lines (or re-plot one line) just to get Line2D handles for legend
    if CvsO_colours is True:
        line_cardinal, = ax_line.plot([], [], color=palette[0], label='Cardinal')
        line_oblique, = ax_line.plot([], [], color=palette[1], label='Oblique')
        # Use these lines for legend handles
        ax_line.legend(handles=[line_cardinal, line_oblique],
                       labels=[f"Cardinal", f"Oblique"], bbox_to_anchor=[1.02, 0.8])

    ax_line.set(xlim=(-90, 90), xticks=[-90, -45, 0, 45, 90],
                yticks=[])

    # Leave room for oriented lines to be used
    ax_line.set_xlabel(f"Orientation ({deg})", labelpad=30, fontsize=22)
    ax_line.set_ylabel(f"Firing Rate\n(spikes / sec)", labelpad=12, fontsize=22)

    # SAVING
    fig_line.set_dpi(dpi)
    fig_line.tight_layout()
    save_name = f"_CvsO_{save_name}"
    fig_line.savefig(os.path.join(save_dir, f"{save_name}_line.png"))


def thesis_plots():
    z = 0
