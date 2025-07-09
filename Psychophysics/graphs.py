import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mytools.dframe_tools import index_dframe, change_dframe_labels
from mytools.dict_tools import append_dicty
from mytools.misc import assign_participant_numbers
from mytools.my_seaborn import load_seaborn_prefs, plot_bar, plot_line, init_markers, multifig_formatter
from mytools.string_tools import misc_symbols

# Init
sym = misc_symbols()

# Paths
og_dir = os.getcwd()
exp_info = {
    'Contrast Detection': {
        'path': os.path.join(og_dir, 'Contrast Detection'),
        'ylabel': f"Threshold (%)",
        'ylim': (0, 30), 'log_ylim': (0.1, 30), 'mean_ylim': (0, 15),
        # 'ratio_ylim': (0, 4), 'ratio_log_ylim': (0.95, 10),
        'ratio_ylim': (0, 3), 'ratio_log_ylim': (0.6, 10),  # for thesis
    },
    'Orientation Discrimination': {
        'path': os.path.join(og_dir, 'Orientation Discrimination'),
        'ylabel': f"Threshold ({sym['deg']})",
        'ylim': (0, 15), 'log_ylim': (0.1, 20), 'mean_ylim': (0, 6),
        # 'ratio_ylim': (0, 4), 'ratio_log_ylim': (0.95, 10),
        'ratio_ylim': (0, 8), 'ratio_log_ylim': (0.6, 10),  # for thesis
    },
}

# Conditions
participants = [f"p{i + 1}" for i in range(8)]

# Init plot settings #
custom_rc = {
    'axes.titlesize': 26, 'axes.labelsize': 26, 'xtick.labelsize': 24, 'ytick.labelsize': 24,
    'legend.fontsize': 21, 'legend.title_fontsize': 21, 'legend.handletextpad': 0.25,
}

load_seaborn_prefs(custom_rc=custom_rc)

markers = init_markers()
# rgb_1 = (0, 0.5, 1)  # Blue
# rgb_2 = (1, 0.5, 0)  # Orange
# Thesis plots RGB
rgb_1 = (0.2, 0.5, 0.8)  # Blue
rgb_2 = (0.8, 0.5, 0.2)  # Orange
# Color palette
cblind = sns.color_palette('colorblind')  # set colors assoc'd w each ori
c_palette = [[0.2] * 3, [0.8] * 3, cblind[2], cblind[3]]  # set H and V to B&W (hashes unclear with smaller bars)
c_palette = [cblind[0], cblind[1], cblind[2], cblind[3]]  # set H and V to B&W (hashes unclear with smaller bars)
# Labels for orientation
old_ori_labels = ['horizontal', 'vertical', 'minus45', 'plus45']
# ori_labels = ['0$^\circ$', '90$^\circ$', '-45$^\circ$', '+45$^\circ$']  # Thesis
ori_labels = ['0$^\circ$', '90$^\circ$', '45$^\circ$', '135$^\circ$']  # (obliques labelled wrongly in exp file)

# Which plots
_plot_thresholds = True
_plot_ratios = True
_plot_indiv = False  # Plot graphs for each individual participant
_plot_multifig = True

# Plot settings for each experiment
plot_settings = {
    'line_alpha': 0.8, 'title_y': 1.02, 'xlabel': f"Spatial frequency ({sym['cpd']})",
    'figdims_title': (5.5, 4.3),
    'figdims': (5, 3.5), 'mean_figdims': (9, 5), 'ratio_figdims': (6, 5),
    'dpi': 600, 'legend': True,
    # ylims
    'ratio_ylim': (0, 4), 'ratio_log_ylim': (0.95, 10),
}
# Apply default setting if not set in exp_info
for i_exp in exp_info:
    for key in plot_settings:
        if exp_info[i_exp].get(key) is None:
            exp_info[i_exp][key] = plot_settings[key]

for i_exp in list(exp_info):
    # Settings for this experiment
    i_info = exp_info[i_exp]
    # Data paths
    exp_dir = i_info['path']
    raw_dir = os.path.join(exp_dir, 'raw')
    sum_dir = os.path.join(exp_dir, 'summary')
    ratio_dir = os.path.join(exp_dir, 'ratio')
    # Graph paths
    gph_dir = os.path.join(exp_dir, 'graphs')
    indiv_gph_dir = os.path.join(gph_dir, 'indiv')
    means_gph_dir = os.path.join(gph_dir, 'means')
    ratio_gph_dir = os.path.join(gph_dir, 'ratios')
    for i_dir in [gph_dir, indiv_gph_dir, means_gph_dir, ratio_gph_dir]:
        if not os.path.exists(i_dir):
            os.makedirs(i_dir)

    # Load data
    raw_data = pd.read_csv(os.path.join(raw_dir, 'raw_anonymised.csv'))
    sum_data = pd.read_csv(os.path.join(sum_dir, 'summary.csv'))
    ratio_data = pd.read_csv(os.path.join(ratio_dir, 'ratios.csv'))
    geomean_r_data = pd.read_csv(os.path.join(ratio_dir, 'geomean_ratios.csv'))

    # Change ori labels for each dataset
    raw_data, sum_data = [
        change_dframe_labels(_i, 'orientation', old_ori_labels, ori_labels) for _i in [
            raw_data, sum_data
        ]
    ]
    # Add empty data for participants who didn't perform 8 cyc/deg
    blank_data = {'participant': [], 'spatial frequency': [], 'orientation': [], 'threshold': []}
    for i_p in participants:
        for i_ori in raw_data['orientation'].unique():
            i_df = index_dframe(raw_data, ['participant', 'orientation'], [i_p, i_ori])
            if index_dframe(i_df, 'spatial frequency', 8).empty:
                append_dicty(blank_data, [i_p, 8, i_ori, np.nan])
                # append_dicty(blank_data, [i_p, 8, i_ori, 0])
    blank_data = pd.DataFrame(blank_data)
    raw_data = pd.concat([raw_data, blank_data])

    # Thesis - 2 vs 16 c/deg
    _thesis = False
    if _thesis is True:
        sf_list = [2, 16]
    else:
        sf_list = [2, 8, 16]
    raw_data = index_dframe(raw_data, 'spatial frequency', sf_list, 'or')
    sum_data = index_dframe(sum_data, 'spatial frequency', sf_list, 'or')
    ratio_data = index_dframe(ratio_data, 'spatial frequency', sf_list, 'or')
    geomean_r_data = index_dframe(geomean_r_data, 'spatial frequency', sf_list, 'or')

    # Indiv graphs - multifigure
    if _plot_thresholds is True and _plot_multifig is True:
        rc = {'axes.titlesize': 30, 'axes.labelsize': 28,
              'xtick.labelsize': 26, 'ytick.labelsize': 26,
              'legend.fontsize': 26, 'legend.title_fontsize': 27,
              'errorbar.capsize': 0.2, 'lines.linewidth': 0.4,
              }
        load_seaborn_prefs(custom_rc=rc)

        n_rows = 3
        n_cols = 3
        figsize = (17, 11)  # MS Word is 17 wide
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)
        wspace = 0.1
        hspace = 0.24
        marker_size = 15
        errwidth = 1
        capsize = 0.25

        axes = list(axes.flatten())  # flatten for easier iteration

        # Check whether to skip any ax indices
        p_list = [_p for _p in participants]  # copy
        if len(p_list) < len(axes):
            n_diff = len(axes) - len(p_list)
            skip_idcs = np.arange(n_cols - 1, len(axes), n_cols)[:n_diff]  # Which ax_idx to skip
        else:
            skip_idcs = []

        # Plot multifig
        for idx, ax in enumerate(axes):
            ax.set_clip_on(False)
            if any(_x == idx for _x in skip_idcs):
                # ax.remove()  # remove from figure
                ax.set_visible(False)  # remove from figure
                continue
            i_p = p_list.pop(0)
            p_dframe = raw_data[raw_data['participant'] == i_p]
            # This auto uses the arithmetic mean for thresholds
            # # # LINE PLOT # # #
            n_hues = len(p_dframe['orientation'].unique())
            _, ax = plot_line(p_dframe, x='spatial frequency', y='threshold', hue='orientation',
                              hue_order=ori_labels, palette=c_palette,
                              ci=68, title=f"{i_p}", title_y=1, legend=False,
                              x_label=i_info['xlabel'], y_label=i_info['ylabel'], y_lim=i_info['ylim'],
                              markers=markers[:n_hues], marker_size=225, dodge=0.4,
                              capsize=capsize, errwidth=errwidth,
                              ax=ax, fig=fig,
                              )
            # Formatting
            [_line.set_markersize(marker_size) for _line in ax.lines]
            ax.yaxis.grid(which='both', linestyle='-', alpha=0.4)

        # Format multifig
        fig = multifig_formatter(fig, x_label=i_info['xlabel'], y_label=i_info['ylabel'], fontsize=35)

        axes[n_cols - 2].legend(loc="center").set(title='Orientation', bbox_to_anchor=[1.4, 0.6])  # Legend
        fig.tight_layout()  # Spacing - must be after .legend()
        fig.subplots_adjust(wspace=wspace, hspace=hspace)  # Gaps between axes - must be after .tight_layout()
        # Save
        fig.savefig(os.path.join(indiv_gph_dir, f"all_lin.png"), dpi=i_info['dpi'])

        # Reload default plot settings
        load_seaborn_prefs(custom_rc=custom_rc)

    # PLOT MEANS #
    # Init plot settings for mean graps (smaller text as singular graph) #
    load_seaborn_prefs(custom_rc={
        'axes.titlesize': 33, 'axes.labelsize': 24, 'xtick.labelsize': 20, 'ytick.labelsize': 20,
        'legend.fontsize': 21, 'legend.title_fontsize': 21, 'lines.linewidth': 0.4,
    })

    # LINE PLOT #
    n_hues = len(sum_data['orientation'].unique())
    fig_means, ax_means = plot_line(sum_data, x='spatial frequency', y='threshold', hue='orientation',
                                    hue_order=ori_labels, palette=c_palette,
                                    ci=68, title_y=1, legend=False,
                                    x_label=i_info['xlabel'], y_label=i_info['ylabel'],
                                    y_lim=i_info['mean_ylim'],
                                    markers=markers[:n_hues], marker_size=10, dodge=0.4, line_width=1,
                                    capsize=capsize, errwidth=errwidth,
                                    fig_dims=(6, 4)
                                    )
    [_line.set_markersize(12) for _line in ax_means.lines]
    ax_means.yaxis.grid(which='both', linestyle='-', alpha=0.4)
    # fig_means.tight_layout()
    fig_means.savefig(os.path.join(means_gph_dir, f"threshold_means.png"), dpi=i_info['dpi'], )

    # Reload default plot settings
    load_seaborn_prefs(custom_rc=custom_rc)

    # PLOT RATIOS #
    load_seaborn_prefs(custom_rc={
        'axes.titlesize': 28, 'axes.labelsize': 28, 'xtick.labelsize': 24, 'ytick.labelsize': 24,
        'legend.fontsize': 24, 'legend.title_fontsize': 24, 'lines.linewidth': 0.8,
    })
    # Multifigure
    if _plot_ratios is True and _plot_multifig is True:
        wspace = 0.24
        # PLOT RATIOS #
        for i_ratio in ['OEI']:
            r_plot_data = index_dframe(ratio_data, ['ratio'], [i_ratio])  # Index OEI scores
            if r_plot_data.empty:
                continue
            # Multifigure for ratio and geomean ratios
            r_fig, r_axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5.5))
            n_hues = len(r_plot_data['participant'].unique())
            my_palette = sns.blend_palette([rgb_1, rgb_2], as_cmap=False, n_colors=n_hues)
            # Add missing data as NaN so that line is drawn
            for i_p in participants:
                for i_sf in sf_list:
                    i_df = index_dframe(r_plot_data, ['participant', 'spatial frequency'], [i_p, i_sf], 'and')
                    if i_df.empty:
                        r_plot_data.loc[len(r_plot_data)] = [i_p, i_sf, 'OEI', np.nan]
            # Line plot
            _, r_ax0 = plot_line(r_plot_data, x='spatial frequency', y='score', hue='participant',
                                 hue_order=participants, palette=my_palette, markers=markers[:n_hues], dodge=0.4, line_width=2,
                                 legend=False, legend_title='', title=None, y_label=i_ratio, x_label=i_info['xlabel'],
                                 y_lim=i_info['ratio_ylim'], ax=r_axes[0],
                                 )
            # Formatting
            [_line.set_markersize(15) for _line in r_ax0.lines]
            r_ax0.set_title("A.", loc="left", x=0, y=1.1, fontdict={'fontweight': 'bold',
                                                                    'fontsize': 40})
            r_ax0.yaxis.grid(which='both', linestyle='-', alpha=0.4)
            r_ax0.legend(title="Participant", fontsize=16, title_fontsize=18, bbox_to_anchor=[1.19, 1.15],
                         framealpha=0.9)  # Legend
            # Draw missing lines (NaN) - interpolate (linear or log)
            log_lines_ydata = []
            for idx_line, i_line in enumerate(r_ax0.lines):
                i_line_ydata = i_line.get_ydata()
                # if len(i_line_ydata) == 0:
                #     continue
                if len(i_line_ydata) == 0:
                    log_lines_ydata.append(i_line_ydata)  # no interpolation required
                elif not np.isnan(i_line_ydata[1]):
                    log_lines_ydata.append(i_line_ydata)  # no interpolation required
                else:
                    # Set linear ydata initially
                    linear_ydata = [_i for _i in i_line_ydata]  # copy
                    # linear_ydata[1] = np.nanmean(i_line_ydata)  # linear interpolation
                    linear_ydata[1] = None
                    i_line.set(ydata=linear_ydata, linestyle='--', alpha=1)
                    # Save log_ydata for later use
                    log_ydata = [_i for _i in i_line_ydata]  # copy
                    log_ydata[1] = np.exp(np.nanmean(np.log(i_line_ydata)))  # log interpolation
                    log_lines_ydata.append(log_ydata)

            # PLOT GEOMEAN RATIOS #
            ii_geomean_ratio_data = index_dframe(geomean_r_data, ['ratio'], [i_ratio])
            _, r_ax1 = plot_line(ii_geomean_ratio_data, x='spatial frequency', y='score', hue='ratio',
                                 legend=False, title=None, ci=None, palette=[[0.3] * 3],
                                 markers=["D"], line_width=2,
                                 y_label=i_ratio, ylabel_pad=6, y_lim=i_info['ratio_ylim'],
                                 x_label=f"Spatial frequency ({sym['cpd']})",
                                 x_type='cat', alpha=1,
                                 ax=r_axes[1]
                                 )
            [_line.set_markersize(15) for _line in r_ax1.lines]

            # Formatting
            r_ax1.set_title("B.", loc="left", x=0, y=1.1, fontdict={'fontweight': 'bold',
                                                                       'fontsize': 40})
            [_ax.yaxis.grid(which='both', linestyle='-', alpha=0.4) for _ax in r_axes]  # Set grid on both plots
            r_ax1.set(ylabel=" ", yticklabels=[])

            # Shared axis labels
            # y-axis
            r_fig.axes[0].set_ylabel(i_ratio, labelpad=14, fontdict={'fontweight': 'bold',
                                                                     'fontsize': 32})
            # x-axis
            r_fig.text(0.35, 0.05, i_info['xlabel'],
                       fontdict={'fontweight': 'bold',
                                 'fontsize': 29})
            [_ax.set_xlabel(' ') for _ax in r_fig.axes]  # After texts
            z = 0

            # SAVE LINEAR Y-AXIS
            r_fig.tight_layout()
            r_fig.subplots_adjust(wspace=wspace)  # Must be done after .tight_layout()
            r_fig.savefig(os.path.join(ratio_gph_dir, f"{i_ratio}_combinedplot_lin.png"), dpi=i_info['dpi'])
