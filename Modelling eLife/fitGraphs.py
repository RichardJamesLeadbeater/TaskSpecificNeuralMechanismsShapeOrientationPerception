import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Custom modules
from mytools.my_seaborn import (load_seaborn_prefs, plot_bar, plot_distribution, plot_line, init_markers)
from mytools.symbols import model_symbols, misc_symbols, lowercase_greek_alphabet, uppercase_greek_alphabet
from mytools.dframe_tools import (index_dframe, change_dframe_labels, custom_sort)


def build_label(sf, fwhm, n=None):
    label = f"{sf} c/{deg}\n{m_sym['o_fwhm']}={fwhm}{deg}"
    if n is not None:
        label += f"\nN={n}"

    return label


def get_sf_labels(sf_cpd, sf_fwhm, n_vals):
    # Generate labels for each sf condition
    sf_labels = []
    for i_cpd, i_sf, i_n in zip(sf_cpd, sf_fwhm, n_vals):
        i_lbl = build_label(i_cpd, i_sf, i_n)
        sf_labels.append(i_lbl)

    return sf_labels


def make_decoder_pair_label(df):
    df = change_dframe_labels(
        df,
        'decoder',
        ['sum', 'mag', 'peak', 'SUM', 'MAG', 'PEAK'],
        ['UW', 'MW', 'PK', 'UW', 'MW', 'PK']
    )

    present = list(df['decoder'].unique())
    decoder_order = ['ML', 'PV', 'WTA', 'UW', 'MW', 'PK']
    ordered = [d for d in decoder_order if d in present]

    return ' & '.join(ordered)


# PATHS
og_dir = os.getcwd()
# data_dir = os.path.join(og_dir, 'fits')
# data_dir = os.path.join(og_dir, 'n540_popsize12_maxiter10')
data_dir = os.path.join(og_dir, 'n540_popsize12_maxiter10')
graph_dir = os.path.join(data_dir, 'graphs')
params_dir = os.path.join(graph_dir, 'params')
bestfit_dir = os.path.join(graph_dir, 'bestfit')
isofit_dir = os.path.join(graph_dir, 'isolatedParams')
threshold_dir = os.path.join(graph_dir, 'thresholds')
fitError_dir = os.path.join(graph_dir, 'fitError')

for ipath in [graph_dir, params_dir, bestfit_dir, threshold_dir, fitError_dir, isofit_dir]:
    if not os.path.exists(ipath):
        os.makedirs(ipath)
fit_names = [_i for _i in os.listdir(data_dir) if os.path.exists(os.path.join(data_dir, _i))]

# Load symbol dicts
misc_sym = misc_symbols()
m_sym = model_symbols()
upper_greek = uppercase_greek_alphabet()
lower_greek = lowercase_greek_alphabet()
deg = misc_sym['deg']

# PLOT SETTINGS
beh_oei = {'con_det': [1.06, 1.39, 1.71], 'ori_dis': [2.45, 3.25, 3.36]}
beh_oei_error = {'con_det': [0.039, 0.21, 0.187], 'ori_dis': [0.209, 0.287, 0.527]}

beh_thresholds = {'con_det': [[0.459, 0.491], [1.123, 1.792], [5.154, 9.499]],
                  'ori_dis': [[0.817, 2.033], [1.016, 3.261], [1.339, 4.049]]}
beh_thresholds_error = {'con_det': [[0.051, 0.06], [0.129, 0.491], [0.901, 2.215]],
                        'ori_dis': [[0.112, 0.32], [0.378, 1.066], [0.435, 0.937]]}
markers = init_markers()

# Convert SE to 95CI
_convert_to_95CI = False
if _convert_to_95CI:
    from scipy.stats import t

    n_ps = [8, 6, 8]
    for k in beh_oei_error:
        for idx, _n in enumerate(n_ps):
            t_val = t.ppf(0.975, _n - 1)
            beh_oei_error[k][idx] *= t_val

# PLOT SETTINGS
custom_rc = {
    'axes.titlesize': 22,
    'axes.labelsize': 24, 'axes.labelweight': 'bold',
    'xtick.labelsize': 17, 'ytick.labelsize': 18,  # SF conds
    'legend.title_fontsize': 18, 'legend.fontsize': 14,
    'font.family': 'sans-serif'
}
load_seaborn_prefs(custom_rc=custom_rc)

ordered_labels = {'task': ['con_det', 'ori_dis'],
                  'decoder': ['Unweighted', 'Magnitude-weighted', 'Peak', 'ML', 'PV', 'WTA', 'UW', 'MW', 'PK',
                              'sum', 'mag', 'peak',
                              'Unweighted & ML', 'Unweighted & PV', 'Unweighted & WTA',
                              'Magnitude-weighted & ML', 'Magnitude-weighted & PV', 'Magnitude-weighted & WTA',
                              'Peak & ML', 'Peak & PV', 'Peak & WTA',
                              'ML & UW', 'ML & MW', 'ML & PK',
                              'PV & UW', 'PV & MW', 'PV & PK',
                              'WTA & UW', 'WTA & MW', 'WTA & PK',
                              ],
                  'cond': ['low', 'mid', 'high'],
                  'anisotropy': ['$N_{C}$ / $N_{O}$',
                                 '$h_{C}$ / $h_{O}$',
                                 '$g_{C}$ / $g_{O}$',
                                 '$F_{C}$ / $F_{O}$']
                  }

c_palette = sns.color_palette('colorblind')
ylim = {'con_det': (0, 2), 'ori_dis': (0, 6)}
ylim_thresholds = {'con_det': (0, 12), 'ori_dis': (0, 8)}
# FOLDERS
fit_cols = ['median', 'q25', 'q75', 'bobyqa', 'task', 'decoder']
allsims = []
for fit_name in fit_names:

    i_data_dir = os.path.join(data_dir, fit_name)

    both_tasks = True  # Toggle to show whether both tasks fit at once

    tasks_list = []
    fit_idx = 0
    iso_data = []  # Isolated conditions
    iso_params = []
    iso_opt_params = []
    raw_fits = []
    datalist = []
    opt_fits = []

    for filename in os.listdir(os.path.join(i_data_dir)):

        filepath = os.path.join(i_data_dir, filename)

        if os.path.isdir(os.path.join(i_data_dir, filename)):
            continue
        if not filename.endswith('.csv'):
            continue

        i_fit_df = pd.read_csv(filepath)

        i_fit_df['nFree'] = len([i for i in i_fit_df.columns if i.startswith('freeParam')])
        fit_idx += 1
        i_fit_df['fit_num'] = fit_idx

        if both_tasks == True:
            if i_fit_df['task'].nunique() == 2:
                i_fit_df = change_dframe_labels(i_fit_df, col='decoder',
                                                old_labels=['sum', 'mag', 'peak'],
                                                new_labels=['UW', 'MW', 'PK'])
                i_fit_df['decoder'] = make_decoder_pair_label(i_fit_df)
            else:
                continue

        # If not wanting both tasks but data file is both_tasks fit then skip
        else:
            if i_fit_df['task'].nunique() == 2:
                continue

        if 'TOPFITS' in filename.upper():

            # Adjust labels and sort order
            i_fit_df = change_dframe_labels(i_fit_df, col='decoder', old_labels=['sum', 'mag', 'peak'],
                                            new_labels=['UW', 'MW', 'PK'])

            tasks_list = [*tasks_list, *list(i_fit_df['task'].unique())]

            if filename.startswith('topFits'):
                raw_fits.append(i_fit_df)
            elif filename.startswith('summaryTopFits'):
                datalist.append(i_fit_df)

        elif 'OPTFIT' in filename.upper():
            opt_fits.append(i_fit_df)

    tasks_list = list(set(tasks_list))

    if not len(datalist):
        continue

    if not len(raw_fits):
        continue

    raw_fits = pd.concat(raw_fits)
    fits_df = pd.concat(datalist)
    opt_fits_df = pd.concat(opt_fits).reset_index()

    # SF labels
    # TODO include N and fwhm in original analysis to be able to generate custom sf labels
    nval = int(datalist[0]['total_n'].max())
    sf_labels = get_sf_labels(sf_cpd=[2, 8, 16], sf_fwhm=[28.1, 15.0, 11.0], n_vals=[nval, nval, nval])

    # Label
    n_free = len([i for i in opt_fits_df.columns if i.startswith('freeParam')])

    # Only plot fits with exactly 4 free parameters
    if n_free != 4:
        continue

    save_fit_name = f"BOTHTASKS_{fit_name}" if both_tasks is True else fit_name

    free_params = [opt_fits_df[f"freeParam_{_idx}"].unique()[0] for _idx in range(n_free)]

    label = "_".join(free_params).strip('_').upper()
    label = "_".join(label.split('_FRACTION'))
    label = ((label.replace("FWHM_", "h_")
             .replace("RMAX_", "g_"))
             .replace("NOISE_SCALE_", "F_")
             )

    opt_fits_df['fitname'] = label
    opt_fits_df['nFree'] = len(free_params)

    if both_tasks is False:
        allsims.append(opt_fits_df)

    # Loop through each task
    for idx_task, i_task in enumerate(tasks_list):
        if both_tasks is False:
            task_df = fits_df[fits_df['task'] == i_task]
            task_raw_df = raw_fits[raw_fits['task'] == i_task]
        else:
            task_df = fits_df
            task_raw_df = raw_fits

        # Filter out any outliers after re-evaluation
        task_raw_df = task_raw_df[
            task_raw_df['overall_fit_error'] <= task_raw_df.groupby(['decoder', 'cond'])['overall_fit_error'].transform(
                'median') * 2
            ].reset_index(drop=True)  # ungrouped, clean index
        #
        # Keep the top 50% (lowest errors) per decoder
        task_raw_df = task_raw_df[
            task_raw_df['overall_fit_error'] <=
            task_raw_df.groupby(['decoder', 'cond'])['overall_fit_error'].transform('quantile', 0.5)
            ].reset_index(drop=True)  # ungrouped, clean index

        task_raw_df = custom_sort(task_raw_df, cols=['task', 'decoder', 'cond'],
                                  ordered_labels=[ordered_labels[_col] for _col in ['task', 'decoder', 'cond']])

        task_df = custom_sort(task_df, cols=['task', 'decoder', 'cond'],
                              ordered_labels=[ordered_labels[_col] for _col in ['task', 'decoder', 'cond']])

        # bestfit_df = task_df[task_df['measure'] == 'bobyqa']
        bestfit_df = opt_fits_df[opt_fits_df['task'] == i_task]
        bestfit_df['measure'] = 'OPT_FIT'

        bestfit_df = change_dframe_labels(bestfit_df, 'decoder', ['mag', 'peak', 'sum'], ['MW', 'PK', 'UW'])
        bestfit_df = custom_sort(bestfit_df, cols=['task', 'decoder', 'cond'],
                                 ordered_labels=[ordered_labels[_col] for _col in ['task', 'decoder', 'cond']])

        ##########
        # PLOTS
        ##########
        # bestfit_df = task_raw_df[
        #     task_raw_df['overall_fit_error'] <= task_raw_df.groupby(['decoder', 'cond'])['overall_fit_error']
        #     .transform('min')
        #     ].reset_index(drop=True)  # ungrouped, clean index
        #
        # bestfit_df['measure'] = 'RAW'

        _PLOT_BESTFITS = True
        if _PLOT_BESTFITS is True:
            load_seaborn_prefs(custom_rc)
            # Settings for figure window
            ncols = 2
            nrows = 1
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
            figwidth = 7
            figheight = 4.9
            wspace = 0.2
            fig.set_figheight(figheight * nrows)  # scale height for nrows
            fig.set_figwidth(figwidth * ncols)  # assumes 2 cols
            fig, ax = plot_line(dframe=bestfit_df, x='cond', y='OEI',
                                hue='decoder',
                                dodge=True, marker_size=10, markers=markers,
                                palette=c_palette, title='', y_label="OEI", x_label='Spatial Frequency',
                                legend=True, y_lim=ylim[i_task], my_dpi=300, close_plot=False, custom_err=False,
                                custom_err_label='t_std', _log=False, tight_layout=True,
                                ax=axes[0]
                                )
            ax.set_xticklabels(sf_labels)

            ax.errorbar([0, 1, 2], beh_oei[i_task], yerr=beh_oei_error[i_task], fmt='o', markersize=0.1, color='k',
                        capsize=12, markerfacecolor='none', linestyle='none')

        # Plot anisotropies
        params_df = bestfit_df.copy()
        params_raw = task_raw_df.copy()
        # params_opt = opt_fits.copy()

        # Reformat table for anisotropies plot
        n_free = len([i for i in params_df.columns if i.startswith('freeParam')])
        free_param_labels = [bestfit_df[f"freeParam_{_idx}"].unique()[0] for _idx in range(n_free)]

        params_df = params_df.melt(
            id_vars=[c for c in params_df.columns if c not in free_param_labels],
            var_name='anisotropy', value_name='anisotropy_value'
        )
        params_raw = params_raw.melt(
            id_vars=[c for c in params_raw.columns if c not in free_param_labels],
            var_name='anisotropy', value_name='anisotropy_value'
        )
        # params_opt = params_opt.melt(
        #     id_vars=[c for c in params_raw.columns if c not in free_param_labels],
        #     var_name='anisotropy', value_name='anisotropy_value'
        # )

        params_df = params_df[['task', 'decoder', 'cond', 'measure', 'OEI', 'cardinalT', 'obliqueT',
                               'overall_fit_error', 'fit_num', 'anisotropy', 'anisotropy_value']]
        params_df = params_df.drop_duplicates()

        params_raw = params_raw[['task', 'decoder', 'cond', 'OEI', 'cardinalT', 'obliqueT',
                                 'overall_fit_error', 'fit_num', 'anisotropy', 'anisotropy_value']]
        params_raw = params_raw.drop_duplicates()

        # params_opt = params_opt[['task', 'decoder', 'cond', 'OEI', 'cardinalT', 'obliqueT',
        #                          'overall_fit_error', 'fit_num', 'anisotropy', 'anisotropy_value']]
        # params_opt = params_opt.drop_duplicates()

        # TABLE OF PARAMETER VALUES
        # Plot params of each fit as bar chart (x=fit_num, hue=param, y=bobyqa)
        # params_df['anisotropy_value'] = np.log(params_df['anisotropy_value'])
        # params_raw['anisotropy_value'] = np.log(params_raw['anisotropy_value'])
        # params_opt['anisotropy_value'] = np.log(params_opt['anisotropy_value'])

        params_df = change_dframe_labels(
            params_df, 'anisotropy',
            ['n_fraction', 'fwhm_fraction', 'rmax_fraction', 'noise_scale_fraction'],
            [m_sym[_p] for _p in ['n_fraction', 'fwhm_fraction', 'rmax_fraction', 'noise_fraction']]
        )
        params_raw = change_dframe_labels(
            params_raw, 'anisotropy',
            ['n_fraction', 'fwhm_fraction', 'rmax_fraction', 'noise_scale_fraction'],
            [m_sym[_p] for _p in ['n_fraction', 'fwhm_fraction', 'rmax_fraction', 'noise_fraction']]
        )
        # params_opt = change_dframe_labels(
        #     params_opt, 'anisotropy',
        #     ['n_fraction', 'fwhm_fraction', 'rmax_fraction', 'noise_scale_fraction'],
        #     [m_sym[_p] for _p in ['n_fraction', 'fwhm_fraction', 'rmax_fraction', 'noise_fraction']]
        # )

        # yticks = np.log([1/4, 1/3, 1/2, 1, 2, 3, 4])
        yticks = [1/4, 1/3, 1/2, 1, 2, 3, 4]
        ylim_edit = (np.min(yticks), np.max(yticks))

        # Adjust labels and sort order
        params_df = change_dframe_labels(params_df, 'decoder', ['SUM', 'MAG', 'PEAK'],
                                         ['Unweighted', 'Magnitude-weighted', 'Peak'])
        params_raw = change_dframe_labels(params_raw, 'decoder', ['SUM', 'MAG', 'PEAK'],
                                          ['Unweighted', 'Magnitude-weighted', 'Peak'])
        # params_opt = change_dframe_labels(params_opt, 'decoder', ['SUM', 'MAG', 'PEAK'],
        #                                   ['Unweighted', 'Magnitude-weighted', 'Peak'])

        params_df = custom_sort(params_df, cols=['decoder', 'anisotropy'],
                                ordered_labels=[ordered_labels[_col] for _col in ['decoder', 'anisotropy']])

        params_raw = custom_sort(params_raw, cols=['decoder', 'anisotropy'],
                                 ordered_labels=[ordered_labels[_col] for _col in ['decoder', 'anisotropy']])

        hue_order = [m_sym[_p] for _p in ['n_fraction', 'fwhm_fraction', 'rmax_fraction', 'noise_fraction']]

        # Bar plot
        _PLOT_ANISOTROPY = True
        if _PLOT_ANISOTROPY and _PLOT_BESTFITS:
            # Param anisotropies as independent plot (many combined conditions on x-axis)
            if both_tasks is True:
                fig.savefig(os.path.join(bestfit_dir, f"{save_fit_name}_{i_task}_multi.png"))
                fig, ax_params = plt.subplots(figsize=(12, 4.5))
            else:
                ax_params = axes[1]

            plot_distribution(
                dframe=params_raw,
                x='decoder',
                legend=True,
                y='anisotropy_value',
                hue='anisotropy',
                hue_order=hue_order,
                palette='colorblind',
                title='',
                y_label="Anisotropy\n(Cardinal:Oblique)",
                x_label='Decoder',
                y_lim=ylim_edit,
                width=0.4 if both_tasks is True else 0.7,  # Smaller values = larger spacing between each group on x-axis
                my_dpi=400,
                close_plot=False,
                strip_alpha=0.0,
                whis=0,  # IQR * whis (typically 1.5)
                ax=ax_params
            )
            if both_tasks is False:
                if i_task == 'con_det':
                    ax_params.set_xlabel('Pooling')
            else:
                ax_params.set_xlabel('Decoder & Pooling')
            ax_params.set_yticks(yticks)
            ax_params.set_yticklabels(['1:4', '1:3', '1:2', '1:1', '2:1', '3:1', '4:1'])
            ax_params.axhline(y=1, linestyle='-', linewidth=2, color='grey', alpha=0.6, zorder=99)

            ax_params.set_yscale('log')
            ax_params.set_yticks(yticks)
            ax_params.set_yticks([], minor=True)
            ax_params.set_yticklabels(['1:4', '1:3', '1:2', '1:1', '2:1', '3:1', '4:1'])

            # Add vertical lines between each group of boxes to make clearer separations
            for i in range(0, len(params_raw['decoder'].unique()) + 1):
                ax_params.axvline(
                    x=i - 0.5,
                    ymin=-0.1,  # extend 5% below axis
                    ymax=1,
                    color='grey',
                    linewidth=1,
                    alpha=0.4,
                    clip_on=False
                )
            ax_params.tick_params(axis='x', which='both', length=5)

            if both_tasks is True:
                ax_params.set_xticklabels(ax_params.get_xticklabels(), rotation=25,
                                          fontsize=15, fontweight='normal')
                fig.tight_layout()
            # fig.show()
            fig.savefig(os.path.join(bestfit_dir, f"{save_fit_name}_{i_task}_multi_params.png"))

        # THRESHOLD PLOTS
        _PLOT_THRESHOLDS = False
        if _PLOT_THRESHOLDS and both_tasks is False:
            load_seaborn_prefs(custom_rc=custom_rc)

            threshold_df = bestfit_df.melt(
                id_vars=[c for c in bestfit_df.columns if c not in ['cardinalT', 'obliqueT']],
                var_name='oriT', value_name='threshold'
            )
            threshold_df = change_dframe_labels(threshold_df, 'oriT', ['cardinalT', 'obliqueT'],
                                                ['cardinal', 'oblique']
                                                )
            nrows = 1
            ncols = 3
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
            figwidth = 6
            figheight = 5
            wspace = 0.2
            fig.set_figheight(figheight * nrows)  # scale height for nrows
            fig.set_figwidth(figwidth * ncols)  # assumes 2 cols

            for idx_decoder, i_decoder in enumerate(threshold_df['decoder'].unique()):
                i_threshold_df = threshold_df[threshold_df['decoder'] == i_decoder]
                fig, ax = plot_line(
                    dframe=i_threshold_df, x='cond', y='threshold', hue='oriT',
                    dodge=True, marker_size=10, markers=markers,
                    palette=c_palette, title='', y_label="Threshold", x_label='Spatial Frequency',
                    legend=False, y_lim=ylim_thresholds[i_task], my_dpi=300, close_plot=False, custom_err=False,
                    custom_err_label='t_std', _log=False, tight_layout=True,
                    ax=axes[idx_decoder]
                )

                ax.set_xticklabels(sf_labels)

                c_thresholds = [beh_thresholds[i_task][_idx][0] for _idx in range(len(beh_thresholds[i_task]))]
                c_err = [beh_thresholds_error[i_task][_idx][0] for _idx in range(len(beh_thresholds_error[i_task]))]
                ax.errorbar([0, 1, 2], c_thresholds, yerr=c_err, fmt='o', markersize=0.1,
                            color='k', capsize=12, markerfacecolor='none', linestyle='none')
                o_thresholds = [beh_thresholds[i_task][_idx][1] for _idx in range(len(beh_thresholds[i_task]))]
                o_err = [beh_thresholds_error[i_task][_idx][1] for _idx in range(len(beh_thresholds_error[i_task]))]
                ax.errorbar([0, 1, 2], o_thresholds, yerr=o_err, fmt='o', markersize=0.1, color='k',
                            capsize=12, markerfacecolor='none', linestyle='none')

                ax.set_title(i_decoder.upper())

            axes[-1].legend()
            fig.savefig(os.path.join(threshold_dir, f"{save_fit_name}_{i_task}_multi_thresholds.png"))

            for ax in axes:
                ax.set_yscale("log")
                ax.set_ylim(0.1, 100)
            fig.savefig(os.path.join(threshold_dir, f"{save_fit_name}_{i_task}_multi_thresholds_log.png"))

        _PLOT_MSE = False
        if _PLOT_MSE:
            # Plot IQR and distribution
            load_seaborn_prefs()
            # PLOT MSE FOR EACH PARTICIPANT AND EACH DECODER
            fig, axes = plt.subplots(ncols=2, figsize=(11, 5))
            ax1 = axes[0]
            ax2 = axes[1]

            # AX1
            ylim_mse = (0, task_raw_df['overall_fit_error'].max() * 1.1)
            plot_line(
                task_raw_df,
                x='decoder',
                y='overall_fit_error', hue='decoder',
                dodge=True, marker_size=8, linewidth=0.6, markers=markers,
                palette='colorblind', title='Mean MSE', y_label="Mean MSE", x_label='Decoder',
                legend=True, y_lim=ylim_mse, close_plot=False, tight_layout=True,
                ax=ax1
            )

            # AX2
            plot_distribution(
                dframe=params_raw,
                # x='fit_num' if both_tasks is True else 'decoder',
                x='decoder',
                legend=True,
                y='anisotropy_value', hue='anisotropy', hue_order=hue_order,
                palette='colorblind', title='', y_label="Anisotropy\n(Cardinal:Oblique)",
                x_label='Decoder' if i_task == 'ori_dis' else 'Pooling',
                y_lim=ylim_edit, my_dpi=400, close_plot=False,
                strip_alpha=0.05,
                strip_color=(0, 0, 0),
                whis=0,  # IQR * whis
                ax=ax2
            )
            ax2.set_yticks(yticks)
            ax2.set_yticklabels(['1:4', '1:3', '1:2', '1:1', '2:1', '3:1', '4:1'])
            if i_task == 'con_det':
                axes[1].set_xlabel('Pooling Method')

            # Grid
            # ax.yaxis.grid(which='both', linestyle='-', alpha=0.4)
            # Horizontal line at OEI = 1
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.4, linewidth=2, zorder=0)
            # fig.show()
            fig.savefig(os.path.join(fitError_dir, f"{save_fit_name}_{i_task}_MSE.png"))

        # Collect isolated params data for special plots
        if len(free_param_labels) == 1:
            iso_data.append(task_raw_df)
            iso_params.append(params_raw)

    z = 0

plt.close('all')

############################################
### PLOT ISOLATED PARAMETER FITS ###
############################################
custom_rc = {
    'axes.titlesize': 26,
    'axes.labelsize': 24, 'axes.labelweight': 'bold',
    'xtick.labelsize': 17, 'ytick.labelsize': 18,  # SF conds
    'legend.title_fontsize': 12, 'legend.fontsize': 16,
    'font.family': 'sans-serif'
}
load_seaborn_prefs(custom_rc=custom_rc)

_plot_iso = False
if _plot_iso:
    opt_fits_df = pd.concat(allsims)
    iso_df = opt_fits_df[opt_fits_df['nFree'] == 1]
    iso_df = change_dframe_labels(iso_df, 'decoder', ['SUM', 'MAG', 'PEAK', 'sum', 'mag', 'peak'],
                                  ['Unweighted', 'Magnitude-weighted', 'Peak',
                                   'Unweighted', 'Magnitude-weighted', 'Peak'])

    # Pull out anisotropy column and save assigned values (remove others)
    iso_df = iso_df.melt(id_vars=[c for c in iso_df.columns if c != 'freeParam_0'],
                         var_name='label', value_name='anisotropy')

    col_idx = iso_df.columns.get_indexer(iso_df['anisotropy'])
    iso_df['anisotropy_value'] = iso_df.to_numpy()[range(len(iso_df)), col_idx]

    iso_df = change_dframe_labels(
        iso_df, 'anisotropy', ['n_fraction', 'fwhm_fraction', 'rmax_fraction', 'noise_scale_fraction'],
        [m_sym[_p] for _p in ['n_fraction', 'fwhm_fraction', 'rmax_fraction', 'noise_fraction']]
    )

    for i_task in iso_df['task'].unique():
        i_df = iso_df[iso_df['task'] == i_task]

        fig, axes = plt.subplots(nrows=1, ncols=3)
        fig.set_figwidth(6 * 3)
        fig.set_figheight(7)

        i_df = custom_sort(i_df, ['cond', 'decoder', 'anisotropy'],
                           [ordered_labels[_col] for _col in ['cond', 'decoder', 'anisotropy']])

        # Loop through figure axes and plot
        axes = axes.flatten()
        for idx, i_decoder in enumerate(i_df['decoder'].unique()):

            plot_df = i_df[i_df['decoder'] == i_decoder]
            if plot_df.empty:
                continue

            # Update labels to include the best-fitting parameter estimate for each isolated condition
            param_labels = [f"{i} = {plot_df.loc[plot_df['anisotropy'] == i]['anisotropy_value'].to_list()[0]:.2f}"
                            for i in plot_df['anisotropy'].unique()]
            plot_df = change_dframe_labels(plot_df, 'anisotropy', plot_df['anisotropy'].unique(), param_labels)

            # plot_df = custom_sort(plot_df, ['cond', 'decoder', 'anisotropy'],
            #                       [ordered_labels[_col] for _col in ['cond', 'decoder', 'anisotropy']])

            ax = axes[idx]
            print(1)
            plot_line(dframe=plot_df, x='cond', y='OEI', hue='anisotropy',
                      dodge=True, marker_size=10, markers=markers,
                      palette=c_palette, title='',
                      # y_label=" ", x_label=' ',
                      y_label="OEI", x_label='Spatial Frequency',
                      legend=True, y_lim=ylim[i_task], my_dpi=300, close_plot=False, custom_err=False,
                      custom_err_label='t_std', _log=False, tight_layout=True,
                      ax=ax
                      )

            ax.set_xticklabels(sf_labels)
            ax.errorbar([0, 1, 2], beh_oei[i_task], yerr=beh_oei_error[i_task], fmt='o', markersize=0.1, color='k',
                        capsize=8, markerfacecolor='none', linestyle='none')
            ax.set_title(f"{i_decoder}")

            # Remove labels of axes objects not in first row or col
            if not ax.get_subplotspec().is_first_col():
                ax.set_ylabel(' ')
                ax.set_yticklabels([])
            if not ax.get_subplotspec().is_last_row():  # shared xaxis
                ax.set_xlabel(' ')
                ax.set_xticklabels([])
            print(2)
            # Grid
            ax.yaxis.grid(which='both', linestyle='-', alpha=0.4)
            # Horizontal line at OEI = 1
            ax.axhline(y=1, color='k', linestyle='--', alpha=0.4, linewidth=2)

            # Legend
            fig.set_clip_on(False)
            [_ax.set_clip_on(False) for _ax in axes]  # Allows legend to exceed boundaries of each plot

            # Add legend above this subplot
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.22))
            z = 0

        # Edit current axes labels
        print(3)
        for i_ax in axes:
            # Remove all axes labels
            i_ax.set_xlabel(' ')
            i_ax.set_ylabel(' ')
            # Remove yticklabels on all but first column
            if i_ax.get_subplotspec().is_first_col() is False:
                i_ax.set_yticklabels([])
            # Set shared labels
            if i_ax.get_subplotspec().colspan.start == 1:
                i_ax.set_xlabel("Spatial Frequency", labelpad=12,
                                fontdict={'fontsize': 24,
                                          'fontweight': 'bold'}
                                )
            if i_ax.get_subplotspec().is_first_col():
                i_ax.set_ylabel("OEI", labelpad=12,
                                fontdict={'fontsize': 24,
                                          'fontweight': 'bold'}
                                )

        fig.subplots_adjust(wspace=0.3)
        fig.savefig(os.path.join(isofit_dir, f"isolatedParams_{i_task}.png"), dpi=400)
    z = 0

################
# COMPARE MSE OF DIFFERENT MODEL FITS - REQUIRES INDIVIDUAL DATA
################
allsims = pd.concat(allsims)

allsims_labels = allsims['fitname'].to_list()
allsims_labels = [f"{lbl.strip('_').replace('__', ', ')}" for lbl in allsims_labels]

allsims['fitname'] = allsims_labels

allsims = change_dframe_labels(allsims, 'decoder', ['sum', 'mag', 'peak'], ['UW', 'MW', 'PK'])

unq_labels = allsims['fitname'].unique()
sorted_labels = sorted(unq_labels, key=lambda s: len(s.split(", ")))
allsims = custom_sort(allsims, cols=['fitname'], ordered_labels=[sorted_labels])

allsims = custom_sort(allsims, cols=['task', 'decoder'],
                      ordered_labels=[ordered_labels[_col] for _col in ['task', 'decoder']]
                      )

ncols = 3
nrows = 2

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))

# Ensure 2D indexing even if nrows=1
if nrows == 1:
    axes = np.array([axes])

for row, i_task in enumerate(allsims['task'].unique()):
    task_df = index_dframe(allsims, ['task'], [i_task])

    for col, i_decoder in enumerate(task_df['decoder'].unique()):  # only 3 per row
        ax = axes[row, col]

        plot_df = index_dframe(allsims, ['task', 'decoder'], [i_task, i_decoder])
        if plot_df.empty:
            ax.axis('off')
            continue

        plot_bar(plot_df, 'fitname', 'overall_fit_error',
                 hue='fitname',
                 palette='colorblind',
                 title=f"{i_decoder}",
                 _log=False,
                 y_label='MSE', x_label='Free Parameters',
                 title_y=1.01,
                 ax=ax)

        # Rotate x labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=67.5, fontsize=14)

        # ✅ Show ticks only on outer edges
        if row != nrows - 1:  # not bottom row
            ax.set_xticklabels([])
            ax.set_xlabel('')
        if col != 0:  # not first column
            ax.set_yticklabels([])
            ax.set_ylabel('')

fig.tight_layout()
fig.savefig(os.path.join(fitError_dir, 'allsims.png'), dpi=600)

z = 0

