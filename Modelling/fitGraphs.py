"""Generate the main figures accompanying the manuscript.

Expected directory structure
----------------------------
JoV_fits/deN50_p12_i10_MS101/
    <fit directories>/
        *.csv
        indiv/<participant>/*.csv

Outputs are written to the ``graphs`` directory.
"""

from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from mytools.dframe_tools import (
    change_dframe_labels,
    custom_sort,
    index_dframe,
)
from mytools.image_tools import stitch_images
from mytools.my_seaborn import (
    init_markers,
    load_seaborn_prefs,
    plot_line,
    plot_parameter_solutions,
    plot_parameter_solutions_by_decoder,
)
from mytools.symbols import misc_symbols, model_symbols


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path.cwd() / "JoV_fits" / "deN50_p12_i10_MS101"
GRAPH_DIR = DATA_DIR / "graphs"

OUTPUT_DIRS = {
    "combined": GRAPH_DIR,
    "multiseed": GRAPH_DIR / "multiseed",
    "individual": GRAPH_DIR / "indiv",
    "thresholds": GRAPH_DIR / "thresholds",
    "both_tasks": GRAPH_DIR / "both",
    "best_seeds": GRAPH_DIR / "bestseeds",
}

for directory in OUTPUT_DIRS.values():
    directory.mkdir(parents=True, exist_ok=True)

DPI = 600
ERROR_COL = "MSE"
N_LABEL = "totalN"

TASKS = ("con_det", "ori_dis")
CONDITIONS = ("low", "mid", "high")
DECODER_ORDER = ("ML", "PV", "WTA", "UW", "MW", "PK")
PARAMETER_ORDER = (
    "n_fraction",
    "fwhm_fraction",
    "rmax_fraction",
    "noise_scale_fraction",
)

BEHAVIOURAL_OEI = {
    "con_det": [1.06, 1.39, 1.71],
    "ori_dis": [2.45, 3.25, 3.36],
}

BEHAVIOURAL_OEI_ERROR = {
    "con_det": [0.039, 0.210, 0.187],
    "ori_dis": [0.209, 0.287, 0.527],
}

BEHAVIOURAL_THRESHOLDS = {
    "con_det": [[0.459, 0.491], [1.123, 1.792], [5.154, 9.499]],
    "ori_dis": [[0.817, 2.033], [1.016, 3.261], [1.339, 4.049]],
}

BEHAVIOURAL_THRESHOLD_ERROR = {
    "con_det": [[0.051, 0.060], [0.129, 0.491], [0.901, 2.215]],
    "ori_dis": [[0.112, 0.320], [0.378, 1.066], [0.435, 0.937]],
}

OEI_LIMITS = {"con_det": (0, 2.5), "ori_dis": (0, 5)}
THRESHOLD_LIMITS = {"con_det": (0, 12), "ori_dis": (0, 8)}

PARTICIPANTS = ("rjl", "tl", "pvm", "kw", "yb", "re", "lj", "mw")
PARTICIPANT_LABELS = {
    name: f"P{index + 1}" for index, name in enumerate(PARTICIPANTS)
}

ORDERED_LABELS = {
    "task": list(TASKS),
    "decoder": list(DECODER_ORDER),
    "cond": list(CONDITIONS),
    "participant": list(PARTICIPANT_LABELS.values()),
}

CUSTOM_RC = {
    "axes.titlesize": 22.5,
    "axes.labelsize": 22.5,
    "axes.labelweight": "bold",
    "xtick.labelsize": 17.5,
    "ytick.labelsize": 17.5,
    "legend.title_fontsize": 17,
    "legend.fontsize": 15,
    "font.family": "sans-serif",
}

load_seaborn_prefs(custom_rc=CUSTOM_RC)

MARKERS = init_markers()
PALETTE = sns.color_palette("colorblind")
MODEL_SYMBOLS = model_symbols()
DEGREE = misc_symbols()["deg"]

ANISOTROPY_TICKS = [1 / 4, 1 / 3, 1 / 2, 1, 2, 3, 4]
ANISOTROPY_LABELS = ["1:4", "1:3", "1:2", "1:1", "2:1", "3:1", "4:1"]


# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def spatial_frequency_labels():
    """Return the three labels used on spatial-frequency axes."""
    values = zip((2, 8, 16), (28.1, 15.0, 11.0))
    return [
        f"{sf} c/{DEGREE}\n{MODEL_SYMBOLS['o_fwhm']}={fwhm}{DEGREE}"
        for sf, fwhm in values
    ]


def normalise_decoders(df):
    """Apply the abbreviated decoder labels used in the figures."""
    return change_dframe_labels(
        df,
        "decoder",
        old_labels=["sum", "mag", "peak", "SUM", "MAG", "PEAK"],
        new_labels=["UW", "MW", "PK", "UW", "MW", "PK"],
    )


def decoder_pair_label(df):
    """Create a stable label for a fit shared across both tasks."""
    present = set(normalise_decoders(df)["decoder"])
    return " & ".join(decoder for decoder in DECODER_ORDER if decoder in present)


def preprocess_fit(df):
    """Apply preprocessing shared by all fit files."""
    df = normalise_decoders(df.copy())
    df["nFree"] = sum(column.startswith("freeParam") for column in df)

    if df["task"].nunique() == 2:
        df["decoder"] = decoder_pair_label(df)
        df["BOTH_TASKS"] = True

    return df


def parse_reproduction_file(df, filename):
    """Recover task and decoder labels encoded in reproduction filenames."""
    parts = filename.split("'")

    if "con_det" in filename and "ori_dis" in filename:
        decoder = f"{parts[5].upper()} & {parts[7].upper()}"
        task = "both"
    else:
        task = parts[1]
        decoder = parts[3].upper()

    df[["decoder", "task"]] = [decoder, task]
    return normalise_decoders(df)


def load_fits(directory):
    """Load and classify all fit CSV files in a directory."""
    fits = []

    for path in sorted(directory.glob("*.csv")):
        if path.name.startswith(("summaryTopFits", "diffEvol")):
            continue

        df = pd.read_csv(path)
        df["filename"] = path.name

        if "topFits" in path.name:
            df["fitLabel"] = "topFits"
            df = preprocess_fit(df)
        elif "optFit" in path.name:
            df["fitLabel"] = "optFit"
            df = preprocess_fit(df)
        elif "repro" in path.name:
            df["fitLabel"] = "repro"
            df = parse_reproduction_file(df, path.name)
        else:
            continue

        fits.append(df)

    return fits


def combine_fits(fits, label):
    """Combine fit dataframes of one classification."""
    selected = [df for df in fits if df["fitLabel"].iloc[0] == label]
    return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()


def load_individual_fits(directory):
    """Load optimal fits for each participant."""
    individual_dir = directory / "indiv"
    frames = []

    if not individual_dir.exists():
        return pd.DataFrame()

    for participant_dir in sorted(individual_dir.iterdir()):
        if not participant_dir.is_dir():
            continue

        fits = load_fits(participant_dir)
        optimal = combine_fits(fits, "optFit")

        if not optimal.empty:
            optimal["participant"] = PARTICIPANT_LABELS.get(
                participant_dir.name,
                participant_dir.name,
            )
            frames.append(optimal)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def add_mse(df, group_cols):
    """Calculate the mean squared OEI error within each fitted solution."""
    df = df.copy()
    squared_error = (df["OEI"] - df["target"]) ** 2
    df[ERROR_COL] = squared_error.groupby(
        [df[column] for column in group_cols]
    ).transform("mean")
    return df


def add_task_targets(df, group_cols):
    """Add behavioural OEI targets and calculate fit error."""
    frames = []

    for task, task_df in df.groupby("task"):
        task_df = task_df.copy()
        task_df["target"] = task_df["cond"].map(
            dict(zip(CONDITIONS, BEHAVIOURAL_OEI[task]))
        )
        frames.append(add_mse(task_df, group_cols))

    return pd.concat(frames, ignore_index=True)


def free_parameters(df):
    """Return fitted anisotropy parameters in manuscript order."""
    n_free = sum(column.startswith("freeParam") for column in df)
    fitted = {
        df[f"freeParam_{index}"].dropna().iloc[0]
        for index in range(n_free)
    }
    return [parameter for parameter in PARAMETER_ORDER if parameter in fitted]


def fit_label(parameters):
    """Create a concise filename label from the free parameters."""
    replacements = {
        "n_fraction": "N",
        "fwhm_fraction": "h",
        "rmax_fraction": "g",
        "noise_scale_fraction": "F",
    }
    return "_".join(replacements[p] for p in parameters)


def sorted_fits(df):
    """Apply the plotting order used throughout the manuscript."""
    columns = ["task", "decoder", "cond"]
    return custom_sort(
        df,
        cols=columns,
        ordered_labels=[ORDERED_LABELS[column] for column in columns],
    )


# =============================================================================
# SHARED FIGURE FORMATTING
# =============================================================================

def format_anisotropy_axis(ax, parameters, xlabel="Parameter"):
    """Apply shared formatting to parameter-anisotropy panels."""
    ax.set_yscale("log")
    ax.set_ylim(min(ANISOTROPY_TICKS), max(ANISOTROPY_TICKS))
    ax.set_yticks(ANISOTROPY_TICKS)
    ax.set_yticks([], minor=True)
    ax.set_yticklabels(ANISOTROPY_LABELS)
    ax.set_xticklabels(
        [MODEL_SYMBOLS[p].replace(" ", "") for p in parameters]
    )
    ax.axhline(1, color="grey", linewidth=2, alpha=0.6, zorder=99)
    ax.set_xlabel(xlabel)


def add_behavioural_oei(ax, task):
    """Add behavioural OEI estimates and uncertainty."""
    ax.errorbar(
        range(3),
        BEHAVIOURAL_OEI[task],
        yerr=BEHAVIOURAL_OEI_ERROR[task],
        fmt="none",
        color="black",
        capsize=12,
    )
    ax.set_xticklabels(spatial_frequency_labels())


def best_seed(df):
    """Return the seed with the lowest MSE."""
    return (
        df.groupby("de_seed")[ERROR_COL]
        .first()
        .astype(float)
        .idxmin()
    )


# =============================================================================
# FIGURE 1: GROUP OEI FITS AND PARAMETER SOLUTIONS
# =============================================================================

def plot_group_figure(
    task_df,
    correlated_df,
    parameters,
    task,
    save_name,
):
    """Plot fitted OEIs and parameter solutions for each decoder."""
    decoders = task_df["decoder"].unique()
    width, height = 4.5, 4.2

    fig_oei, axes_oei = plt.subplots(
        1,
        len(decoders),
        figsize=(width * len(decoders), height),
        squeeze=False,
    )
    axes_oei = axes_oei[0]

    best_fits = []

    for index, decoder in enumerate(decoders):
        ax = axes_oei[index]
        decoder_df = index_dframe(task_df, ["decoder"], [decoder])
        selected_seed = best_seed(decoder_df)
        selected_fit = index_dframe(
            decoder_df,
            ["de_seed"],
            [selected_seed],
        )
        best_fits.append(selected_fit)

        seed_ids = decoder_df["de_seed"].unique()
        plot_line(
            dframe=decoder_df,
            x="cond",
            y="OEI",
            hue="de_seed",
            dodge=False,
            marker_size=10,
            linewidth=0.5,
            palette={seed: (0.7, 0.7, 0.7) for seed in seed_ids},
            title=decoder,
            y_label="OEI" if index == 0 else "",
            x_label="Spatial Frequency" if index == 1 else "",
            legend=False,
            y_lim=OEI_LIMITS[task],
            close_plot=False,
            ax=ax,
        )

        ax.plot(
            selected_fit["cond"],
            selected_fit["OEI"],
            color="black",
            linewidth=2,
            marker="o",
            markersize=8,
        )

        if correlated_df is not None:
            correlated_decoder = index_dframe(
                correlated_df,
                ["decoder"],
                [decoder],
            )
            correlated_fit = index_dframe(
                correlated_decoder,
                ["de_seed"],
                [best_seed(correlated_decoder)],
            )
            line, = ax.plot(
                correlated_fit["cond"],
                correlated_fit["OEI"],
                color="crimson",
                linewidth=2.1,
                linestyle="--",
                marker="x",
                markersize=11,
                markeredgewidth=2,
                alpha=0.8,
            )
            line.set_path_effects([
                pe.Stroke(linewidth=0.5, foreground="white"),
                pe.Normal(),
            ])

        add_behavioural_oei(ax, task)

        if index:
            ax.set_yticklabels([])

    axes_oei[-1].inset_axes([1.05, 0, 0.5, 1]).axis("off")
    fig_oei.tight_layout()
    fig_oei.subplots_adjust(wspace=0.25)

    oei_path = OUTPUT_DIRS["multiseed"] / f"{save_name}_{task}_OEI.png"
    fig_oei.savefig(oei_path, dpi=DPI)
    plt.close(fig_oei)

    best_fits_df = pd.concat(best_fits, ignore_index=True)
    best_fits_df.to_csv(
        OUTPUT_DIRS["best_seeds"] / f"{save_name}_{task}.csv",
        index=False,
    )

    fig_params, axes_params = plot_parameter_solutions_by_decoder(
        task_df,
        decoder_col="decoder",
        colour_by="bestfit",
        parameters=parameters,
        panel_dims=(width, height),
        solution_col="de_seed",
        error_col=ERROR_COL,
        colorbar_label="Fit Error (MSE)",
        show_title=False,
        wspace=0.25,
        cbar_labelpad=7,
    )

    for index, ax in enumerate(axes_params):
        format_anisotropy_axis(
            ax,
            parameters,
            xlabel="Parameter" if index == 1 else "",
        )
        ax.set_ylabel(
            "Anisotropy\n(Cardinal: Oblique)" if index == 0 else ""
        )

        if correlated_df is not None:
            decoder = decoders[index]
            correlated_decoder = index_dframe(
                correlated_df,
                ["decoder"],
                [decoder],
            )
            correlated_fit = index_dframe(
                correlated_decoder,
                ["de_seed"],
                [best_seed(correlated_decoder)],
            )
            ax.plot(
                ax.get_xticks(),
                correlated_fit[parameters].iloc[0],
                color="crimson",
                linewidth=2.1,
                linestyle="--",
                marker="x",
                markersize=11,
                markeredgewidth=2,
                alpha=0.8,
            )

    axes_params[-1].inset_axes([1.05, 0, 0.5, 1]).axis("off")
    fig_params.tight_layout()
    fig_params.subplots_adjust(wspace=0.25)

    params_path = (
        OUTPUT_DIRS["multiseed"] /
        f"{save_name}_{task}_PARAMETERS.png"
    )
    fig_params.savefig(params_path, dpi=DPI)
    plt.close(fig_params)

    stitch_images(
        [oei_path, params_path],
        output_file=(
            OUTPUT_DIRS["combined"] /
            f"{save_name}_{task}_GROUP_FIGURE.png"
        ),
        direction="vertical",
        separator=True,
        separator_size=15,
        separator_colour=(150, 150, 150),
        panel_gap=90,
        match_size=True,
        labels=["A.", "B."],
        label_pad=40,
    )


# =============================================================================
# FIGURE 2: INDIVIDUAL-PARTICIPANT PARAMETER SOLUTIONS
# =============================================================================

def plot_individual_figure(
    individual_df,
    parameters,
    task,
    save_name,
):
    """Plot the best parameter solution for each participant."""
    if individual_df.empty:
        return

    task_df = individual_df[individual_df["task"] == task].copy()
    task_df = add_task_targets(
        task_df,
        ["participant", "task", "decoder", "de_seed"],
    )
    task_df = custom_sort(
        task_df,
        cols=["participant", "task", "decoder", "cond"],
        ordered_labels=[
            ORDERED_LABELS["participant"],
            ORDERED_LABELS["task"],
            ORDERED_LABELS["decoder"],
            ORDERED_LABELS["cond"],
        ],
    )

    task_df["is_best"] = task_df[ERROR_COL].eq(
        task_df.groupby(["participant", "decoder"])[ERROR_COL]
        .transform("min")
    )
    best_df = task_df[task_df["is_best"]]

    fig, axes = plot_parameter_solutions_by_decoder(
        best_df,
        decoder_col="decoder",
        best=None,
        parameters=parameters,
        panel_dims=(4.5, 4.2),
        markers=MARKERS,
        colour_by="category",
        solution_col="participant",
        error_col=ERROR_COL,
        colorbar_label="Fit Error (MSE)",
        show_title=False,
        line_width=1.5,
        point_size=35,
        wspace=0.25,
    )

    for index, ax in enumerate(axes):
        format_anisotropy_axis(
            ax,
            parameters,
            xlabel="Parameter" if index == 1 else "",
        )
        ax.set_ylabel(
            "Anisotropy\n(Cardinal: Oblique)" if index == 0 else ""
        )

    axes[-1].inset_axes([1.05, 0, 0.05, 1]).set_visible(False)
    axes[-1].legend(
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        title="Participant",
        frameon=True,
    )

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.25)
    fig.savefig(
        OUTPUT_DIRS["individual"] /
        f"{save_name}_{task}_INDIVIDUAL_PARAMETERS.png",
        dpi=DPI,
    )
    plt.close(fig)


# =============================================================================
# FIGURE 3: CARDINAL AND OBLIQUE THRESHOLDS
# =============================================================================

def threshold_dataframe(df):
    """Convert cardinal and oblique threshold columns to long format."""
    result = df.melt(
        id_vars=[
            column
            for column in df
            if column not in ("cardinalT", "obliqueT")
        ],
        value_vars=["cardinalT", "obliqueT"],
        var_name="orientation",
        value_name="threshold",
    )
    return change_dframe_labels(
        result,
        "orientation",
        ["cardinalT", "obliqueT"],
        ["cardinal", "oblique"],
    )


def best_thresholds(df):
    """Select all rows belonging to the minimum-error fit."""
    return df[
        df[ERROR_COL].eq(
            df.groupby("decoder")[ERROR_COL].transform("min")
        )
    ]


def plot_threshold_figure(
    task_df,
    correlated_df,
    task,
    save_name,
):
    """Plot cardinal and oblique thresholds for the best fits."""
    independent = best_thresholds(threshold_dataframe(task_df))
    correlated = best_thresholds(threshold_dataframe(correlated_df))

    decoders = independent["decoder"].unique()
    fig, axes = plt.subplots(
        1,
        len(decoders),
        figsize=(5 * len(decoders), 5),
        squeeze=False,
    )
    axes = axes[0]

    cardinal = [values[0] for values in BEHAVIOURAL_THRESHOLDS[task]]
    oblique = [values[1] for values in BEHAVIOURAL_THRESHOLDS[task]]
    cardinal_error = [
        values[0] for values in BEHAVIOURAL_THRESHOLD_ERROR[task]
    ]
    oblique_error = [
        values[1] for values in BEHAVIOURAL_THRESHOLD_ERROR[task]
    ]

    for index, decoder in enumerate(decoders):
        ax = axes[index]
        decoder_df = index_dframe(independent, ["decoder"], [decoder])
        correlated_decoder = index_dframe(
            correlated,
            ["decoder"],
            [decoder],
        )

        plot_line(
            dframe=decoder_df,
            x="cond",
            y="threshold",
            hue="orientation",
            dodge=True,
            marker_size=10,
            markers=MARKERS,
            palette=PALETTE,
            title=decoder.upper(),
            y_label="Threshold" if index == 0 else "",
            x_label="Spatial Frequency",
            legend=False,
            y_lim=THRESHOLD_LIMITS[task],
            close_plot=False,
            ax=ax,
        )

        for orientation_index, orientation in enumerate(
            ("cardinal", "oblique")
        ):
            orientation_df = index_dframe(
                correlated_decoder,
                ["orientation"],
                [orientation],
            )
            ax.plot(
                orientation_df["cond"],
                orientation_df["threshold"],
                color=PALETTE[orientation_index],
                linestyle="--",
                marker="x",
                linewidth=1.6,
                markersize=8,
                alpha=0.6,
            )

        ax.errorbar(
            range(3),
            cardinal,
            yerr=cardinal_error,
            fmt="none",
            color="black",
            capsize=12,
        )
        ax.errorbar(
            range(3),
            oblique,
            yerr=oblique_error,
            fmt="none",
            color="black",
            capsize=12,
        )
        ax.set_xticklabels(spatial_frequency_labels())

        if index:
            ax.set_yticklabels([])

    legend_ax = axes[-1].inset_axes([1.05, 0, 0.5, 1])
    legend_ax.axis("off")

    orientation_handles = [
        Line2D([], [], color=PALETTE[0], marker="o", label="Cardinal"),
        Line2D([], [], color=PALETTE[1], marker="o", label="Oblique"),
    ]
    noise_handles = [
        Line2D([], [], color="black", linestyle="-", label="Independent"),
        Line2D(
            [],
            [],
            color="black",
            marker="x",
            linestyle="--",
            label="Correlated",
        ),
    ]

    orientation_legend = legend_ax.legend(
        handles=orientation_handles,
        title="Orientation",
        loc="upper left",
    )
    legend_ax.add_artist(orientation_legend)
    legend_ax.legend(
        handles=noise_handles,
        title="Noise type",
        loc="lower left",
    )

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.25)
    fig.savefig(
        OUTPUT_DIRS["thresholds"] /
        f"{save_name}_{task}_THRESHOLDS.png",
        dpi=DPI,
    )
    plt.close(fig)


# =============================================================================
# FIGURE 4: MODEL FIT SHARED ACROSS BOTH TASKS
# =============================================================================

def plot_shared_model_figure(
    optimal_fits,
    correlated_fits,
    parameters,
    save_name,
):
    """Plot OEI fits and parameters when one model fits both tasks."""
    fig = plt.figure(figsize=(14, 10))
    grid = fig.add_gridspec(
        2,
        2,
        hspace=0.75,
        wspace=0.25,
    )

    axes = {
        "con_det": fig.add_subplot(grid[0, 0]),
        "ori_dis": fig.add_subplot(grid[0, 1]),
        "parameters": fig.add_subplot(grid[1, :]),
    }

    task_titles = {
        "con_det": "Contrast Detection",
        "ori_dis": "Orientation Discrimination",
    }

    for task in TASKS:
        ax = axes[task]
        task_df = optimal_fits[optimal_fits["task"] == task]

        plot_line(
            dframe=task_df,
            x="cond",
            y="OEI",
            hue="decoder",
            dodge=True,
            marker_size=10,
            markers=MARKERS,
            palette=PALETTE,
            title=task_titles[task],
            title_fontweight="normal",
            y_label="OEI",
            x_label="Spatial Frequency",
            legend=False,
            y_lim=OEI_LIMITS[task],
            close_plot=False,
            ax=ax,
        )
        add_behavioural_oei(ax, task)

    if correlated_fits is not None:
        seed_errors = (
            correlated_fits.groupby(["de_seed", "decoder"])[ERROR_COL]
            .first()
            .astype(float)
        )
        selected_seed, selected_decoder = seed_errors.idxmin()

        selected_correlated = index_dframe(
            correlated_fits,
            ["de_seed", "decoder"],
            [selected_seed, selected_decoder],
        )

        for task in TASKS:
            task_fit = selected_correlated[
                selected_correlated["task"] == task
            ]
            line, = axes[task].plot(
                task_fit["cond"],
                task_fit["OEI"],
                color="black",
                linewidth=2.5,
                marker="X",
                markersize=8,
                linestyle="--",
                zorder=100,
            )
            line.set_path_effects([
                pe.Stroke(linewidth=4, foreground="white"),
                pe.Normal(),
            ])

    plot_parameter_solutions(
        optimal_fits,
        parameters=parameters,
        solution_col="decoder",
        best=None,
        deduplicate=True,
        title="Shared Model",
        title_fontweight="normal",
        y_label="Anisotropy\n(Cardinal: Oblique)",
        palette="colorblind",
        colour_by="category",
        line_alpha=0.7,
        line_width=1.5,
        point_size=40,
        close_plot=False,
        legend=False,
        ax=axes["parameters"],
    )

    if correlated_fits is not None:
        axes["parameters"].plot(
            axes["parameters"].get_xticks(),
            selected_correlated[parameters].iloc[0],
            color="black",
            linewidth=2.5,
            marker="X",
            markersize=8,
            linestyle="--",
            label=f"Correlated noise\n{selected_decoder}",
        )

    format_anisotropy_axis(axes["parameters"], parameters)

    handles, labels = axes["parameters"].get_legend_handles_labels()
    fig.subplots_adjust(right=0.77)
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.80, 0.5),
        frameon=True,
    )

    fig.text(0.05, 0.95, "A.", fontsize=36, fontweight="bold")
    fig.text(0.05, 0.45, "B.", fontsize=36, fontweight="bold")

    fig.savefig(
        OUTPUT_DIRS["both_tasks"] /
        f"{save_name}_SHARED_MODEL.png",
        dpi=DPI,
    )
    plt.close(fig)


# =============================================================================
# MAIN FIGURE GENERATION
# =============================================================================

def load_correlated_reference(fit_directories):
    """Load the correlated-noise reference fits once."""
    references = {"single": None, "both": None}

    for directory in fit_directories:
        if "cr0.3x22.5" not in directory.name:
            continue

        label = "both" if "_BOTH" in directory.name else "single"
        optimal = combine_fits(load_fits(directory), "optFit")

        if optimal.empty:
            continue

        groups = (
            ["decoder", "de_seed"]
            if label == "both"
            else ["task", "decoder", "de_seed"]
        )
        references[label] = add_task_targets(optimal, groups)

    return references


def main():
    """Generate the manuscript figures."""
    fit_directories = sorted(
        path
        for path in DATA_DIR.iterdir()
        if path.is_dir() and path.name != "graphs"
    )
    correlated = load_correlated_reference(fit_directories)

    for directory in fit_directories:
        # Correlated-noise directories are reference data, not separate outputs.
        if "cr0." in directory.name:
            continue

        fits = load_fits(directory)
        optimal = combine_fits(fits, "optFit")

        if optimal.empty:
            continue

        parameters = free_parameters(optimal)

        # The manuscript reports the model with four free anisotropies.
        if len(parameters) != 4:
            continue

        both_tasks = "BOTH" in directory.name
        save_name = fit_label(parameters)
        if both_tasks:
            save_name = f"BOTHTASKS_{save_name}"

        optimal = sorted_fits(optimal)

        if both_tasks:
            optimal = add_task_targets(
                optimal,
                ["decoder", "de_seed"],
            )
            plot_shared_model_figure(
                optimal,
                correlated["both"],
                parameters,
                save_name,
            )
            continue

        optimal = add_task_targets(
            optimal,
            ["task", "decoder", "de_seed"],
        )
        individual = load_individual_fits(directory)

        for task in TASKS:
            task_df = optimal[optimal["task"] == task]
            correlated_task = (
                correlated["single"][
                    correlated["single"]["task"] == task
                ]
                if correlated["single"] is not None
                else None
            )

            plot_group_figure(
                task_df,
                correlated_task,
                parameters,
                task,
                save_name,
            )
            plot_individual_figure(
                individual,
                parameters,
                task,
                save_name,
            )

            if correlated_task is not None:
                plot_threshold_figure(
                    task_df,
                    correlated_task,
                    task,
                    save_name,
                )

    plt.close("all")


if __name__ == "__main__":
    main()

