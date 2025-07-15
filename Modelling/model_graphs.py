import pandas as pd
import os
import seaborn as sns
from mytools.my_seaborn import load_seaborn_prefs
from mytools.dframe_tools import to_csv_pkl, index_dframe, \
    change_dframe_colnames, change_dframe_labels
from mytools.symbols import model_symbols, misc_symbols, lowercase_greek_alphabet, uppercase_greek_alphabet
from mytools.os_tools import make_paths
import ast
import itertools as itls
from components.model_plots import alter_dframe, task_info, plot_multicond, plot_multicond_thresholds
from mytools.analysis import gen_oei_data, gen_cond_nums

pd.options.mode.chained_assignment = None  # default='warn'  # removes error messages


def get_sf_labels(use_adjusted_sf, vary_n, show_n=False):
    # Define SF values
    sf_vals = (28.1, 15.2, 22.2)

    # Define N values
    n_vals = (540, 540, 540) if vary_n is False else (1080, 540, 268)
    # n_vals = [60] * 3
    # n_vals = [1080, 540, 180]
    # n_vals = [40] * 3
    # n_vals = [32] * 3
    label_n_vals = n_vals if show_n is True else (None, None, None)

    # Generate labels for each sf condition
    sf_labels = []
    for i_cpd, i_sf, i_n in zip([2, 8, 16], sf_vals, label_n_vals):
        i_lbl = build_label(i_cpd, i_sf, i_n)
        sf_labels.append(i_lbl)

    return sf_labels, sf_vals, n_vals


def build_label(sf, hwhm, n=None):
    label = f"{sf} c/{deg}\n{m_sym['o_hwhm']}={hwhm}{deg}"
    if n is not None:
        label += f"\nN={n}"

    return label


# TODO - Improve data organisation (no need for separate folders with different tasks - just index with task)
# Load symbol dicts
misc_sym = misc_symbols()
m_sym = model_symbols()
upper_greek = uppercase_greek_alphabet()
lower_greek = lowercase_greek_alphabet()
deg = misc_sym['deg']

# Plot settings
ylim_oe = (0, 6)  # consistent across all graphs
oridis_ylim_t = (0.1, 100)  # consistent across same task
my_dpi = 100  # Low for tests - high for final copy
multicond_dpi = 450
legends = True

# Script settings
_analyse_data = False
_plot_oei = True
_plot_threshold = True

# Initialise settings
settings = {
    'ConDetSF':
        {'analysis': _analyse_data, 'label': 'con_det',
         'y_lim_oe': (0, 3), 'y_lim_oe_log': (0.6, 10), 'y_label_oei': 'OEI',
         'y_lim_t': (0, 10), 'y_lim_t_log': (0.1, 20), 'y_label_thresh': 'Threshold (%)',
         'dpi': my_dpi, 'legend': legends, 'tight': True, 'close': True, 'figsize': (5, 4),
         # List of labels to rename within the data {col: [old_labels, new_labels]}
         'rename': {'decoder': [['sum', 'mag', 'distance'], ['unweighted', 'magnitude-weighted', 'distance-weighted']],
                    'propn_cardinal': [[0.66], [0.67]]
                    },
         # Decoders to plot
         'decoder_list': ['unweighted', 'magnitude-weighted', 'peak'],
         # ['unweighted', 'magnitude-weighted', 'peak', 'distance-weighted']
         # ['unweighted', 'magnitude-weighted', 'peak', 'distance-weighted']

         'ignore': ['contrast',
                    'tmp_decoder', 'tmp_propn_cardinal', 'tmp_total_n', 'tmp_oblique_hwhm', 'tmp_hwhm_fraction',
                    'tmp_noise_fraction', 'tmp_cardinal_noise', 'tmp_baseline', 'tmp_con'],
         'behavioural_oei': [1.06, 1.39, 1.79],
         'behavioural_t': [[0.459, 1.123, 5.154], [0.491, 1.792, 9.499]],
         },

    'OriDisSF':
        {'analysis': _analyse_data, 'label': 'ori_dis',
         'y_lim_oe': (0, 6), 'y_lim_oe_log': (0.6, 10), 'y_label_oei': 'OEI', 'y_lim_t': (0, 7.5),
         'y_lim_t_log': (0.1, 20),
         'y_label_thresh': f"Threshold ({deg})",
         'dpi': my_dpi, 'legend': legends, 'tight': True, 'close': True, 'figsize': (5, 4),
         # List of labels to rename within the data {col: [old_labels, new_labels]}
         'rename': {'decoder': [['sum', 'mag', 'distance'], ['unweighted', 'magnitude-weighted', 'distance-weighted']],
                    'propn_cardinal': [[0.66], [0.67]]
                    },
         'decoder_list': ['ML', 'PV', 'WTA'],  # ['ML', 'PV', 'WTA']
         'contrast': 100,
         'ignore': ['tmp_decoder', 'tmp_propn_cardinal', 'tmp_total_n', 'tmp_oblique_hwhm', 'tmp_hwhm_fraction',
                    'tmp_noise_fraction', 'tmp_cardinal_noise', 'tmp_baseline', 'tmp_con'],
         'behavioural_oei': [2.45, 3.25, 3.36],
         'behavioural_t': [[0.817, 1.016, 1.339], [2.033, 3.261, 4.049]],
         },
}

# Global settings #
oblique_hwhm_filter = [70, 50, 30, 20]
hwhm_fraction_filter = [1, 0.8, 0.6, 0.4, 0.2]
noise_fraction_filter = [0.5, 1, 2, 3, 4, 5]
contrast_filter = None  # default (changed later)

# Conds for paper (SF) #
sf_labels, sf_vals, n_vals = get_sf_labels(use_adjusted_sf=True, vary_n=False, show_n=True)

# Define params to overwrite eveything else
conds_paper_sf = {}
for i_label, i_sf, i_n in zip(sf_labels, sf_vals, n_vals):
    conds_paper_sf[i_label] = {'oblique_hwhm': i_sf, 'total_n': i_n}

# Default param values - same across all graphs - overwritten by fixed_params and variable params
default_params = {'total_n': 540,  # 540, 30, 16
                  'propn_cardinal': 0.5,
                  'oblique_hwhm': 28.1, 'hwhm_fraction': 1,
                  'cardinal_noise': [1, 0.5], 'noise_fraction': 1,
                  'oblique_rmax': 60, 'rmax_fraction': 1,
                  'spont': '[3, 3]', 'crf_c50': 24, 'crf_exponent': 2,
                  # 'param_distrib': 'cos', 'pref_distrib': 'cos',
                  }

# Fixed param vals for each set of graphs - code will generate conditions based on these
fixed_params = {'propn_cardinal': [0.5, 0.67],
                'hwhm_fraction': [1, 0.5],
                'noise_fraction': [1, 0.5],
                'rmax_fraction': [1, 2],
                }

# Variable param values determining the hue (different lines) on each graph
variable_params = {'propn_cardinal': [1, 0.9, 0.8, 0.7, 0.6, 0.5],
                   'hwhm_fraction': hwhm_fraction_filter,
                   'noise_fraction': noise_fraction_filter,
                   'rmax_fraction': [1, 1.25, 1.67],
                   }

# Plot settings
custom_rc = {
    'axes.titlesize': 22,
    'axes.labelsize': 24, 'axes.labelweight': 'bold',
    'xtick.labelsize': 17, 'ytick.labelsize': 18,  # SF conds
    'legend.title_fontsize': 18, 'legend.fontsize': 18,
    'font.family': 'sans-serif'
}
load_seaborn_prefs(custom_rc=custom_rc)
# Run #
og_dir = os.getcwd()
for i_task in settings:  # loop through each task
    i_settings = settings[i_task]
    if legends is True:
        i_settings['figsize'] = list(i_settings['figsize'])
        i_settings['figsize'][0] = i_settings['figsize'][0] * 1.2

    # PATHS #
    data_path = os.path.join(og_dir, 'data')
    task_data_path = os.path.join(data_path, i_task)
    graph_path = os.path.join(og_dir, 'graphs')
    multicond_path = os.path.join(graph_path, 'multicond')
    task_graph_path = os.path.join(multicond_path, i_task)

    # Create directories if required
    make_paths([task_data_path, graph_path, multicond_path, task_graph_path])

    # LOAD DATA #
    thresh_data = []
    oei_data = []
    param_cols = [
        'contrast', 'total_n', 'propn_cardinal', 'oblique_hwhm', 'hwhm_fraction',
        'cardinal_noise', 'noise_fraction', 'rmax_fraction', 'oblique_rmax',
        'spont', 'crf_exponent', 'crf_c50', 'tuning_dist',
    ]
    if i_settings['analysis'] is False:
        thresh_data = pd.read_csv(os.path.join(task_data_path, 'preloaded_thresh_data.csv'))
        oei_data = pd.read_csv(os.path.join(task_data_path, 'preloaded_oei_data.csv'))
    else:
        thresh_data = []
        for i_dir in os.listdir(data_path):
            i_data_path = os.path.join(data_path, i_dir)
            if not os.path.exists(os.path.join(i_data_path, 'combined_datasets.csv')):
                continue
            i_thresh_dframe = pd.read_csv(os.path.join(i_data_path, 'combined_datasets.csv'))
            # Index the specified task
            if i_settings['label'] in i_thresh_dframe['task'].unique().tolist():
                i_thresh_dframe = i_thresh_dframe[i_thresh_dframe['task'] == i_settings['label']]
                thresh_data.append(i_thresh_dframe)
            else:
                continue

        # Combine data and reformat for analysis
        thresh_data = pd.concat(thresh_data)
        thresh_data = alter_dframe(thresh_data, i_task)  # Add columns for analysis
        thresh_data = change_dframe_colnames(thresh_data, ['threshold'], ['dv'])
        chosen_cols = ['design', 'ori_std', 'task', 'decoder', 'dv'] + param_cols
        thresh_data = thresh_data[chosen_cols]
        thresh_data = index_dframe(thresh_data, ['task'], [i_settings['label']])  # Index task
        to_csv_pkl(thresh_data, task_data_path, 'thresh_data', _pkl=False)

        # Calculate OEIs
        thresh_data = gen_cond_nums(thresh_data, ignore_list=['task', 'ori_std', 'decoder', 'contrast', 'dv'])
        oei_data = gen_oei_data(thresh_data, cond_list=['task', 'decoder', 'contrast', 'cond_num'],
                                ori_label='ori_std', dv_label='dv')
        to_csv_pkl(oei_data, task_data_path, 'oei_data', _pkl=False, rnd=4)

    # Change labels (decoders)
    for i_col in i_settings['rename']:
        oei_data = change_dframe_labels(oei_data, i_col, old_labels=i_settings['rename'][i_col][0],
                                        new_labels=i_settings['rename'][i_col][1])
        thresh_data = change_dframe_labels(thresh_data, i_col, old_labels=i_settings['rename'][i_col][0],
                                           new_labels=i_settings['rename'][i_col][1])

    # Von Mises
    oei_data = index_dframe(oei_data, ['tuning_dist'], ['von_mises'])
    thresh_data = index_dframe(thresh_data, ['tuning_dist'], ['von_mises'])
    print(f"Using Von Mises")

    # Alter proportion cardinal for easier analysis with same code
    pc_to_67 = [0.65, 0.66, 0.68, 0.69]
    oei_data = change_dframe_labels(oei_data, 'propn_cardinal',
                                    old_labels=pc_to_67, new_labels=[0.67] * len(pc_to_67))
    thresh_data = change_dframe_labels(thresh_data, 'propn_cardinal',
                                       old_labels=pc_to_67, new_labels=[0.67] * len(pc_to_67))

    print(f'{i_task} plots')
    # PLOTS #
    # Init figure info - all possible figures #
    default_params['contrast'] = i_settings.get('contrast')
    fig_info = task_info(fixed_params, variable_params, default_params,
                         conds=conds_paper_sf, decoder_list=i_settings['decoder_list'])

    # Plot specified hues (good for papers / summarising)
    # Setting conditions of plot - one specified parameter value and the rest set to default
    hue_params = [
        {
            'propn_cardinal': [0.67],
            'hwhm_fraction': [0.5],
            'noise_fraction': [0.5],
            'rmax_fraction': [2],
        },
    ]

    baseline_params = [
        {'propn_cardinal': 0.5},
        {'propn_cardinal': 0.67},
        # [{'propn_cardinal': 0.75},
        {'hwhm_fraction': 0.5},
        # [{'hwhm_fraction': 0.33},
        {'propn_cardinal': 0.67, 'hwhm_fraction': 0.5},
        {'rmax_fraction': 2, 'hwhm_fraction': 0.5},
    ]

    # Plots
    for i_design in ['mocs']:
        # mocs or staircase
        oei_df = index_dframe(oei_data, ['design'], [i_design])
        i_thresh = index_dframe(thresh_data, ['design'], [i_design])
        # Total N and check if specified in conditions
        if any(conds_paper_sf[_k].get('total_n') is not None for _k in conds_paper_sf):
            total_n = [conds_paper_sf[_k]['total_n'] for _k in conds_paper_sf]  # N from conds
        else:
            total_n = [540]  # Single N
            default_params['total_n'] = total_n[0]
        oei_df = index_dframe(oei_df, ['total_n'], total_n, 'or')
        i_thresh = index_dframe(i_thresh, ['total_n'], total_n, 'or')
        n_label = ("N" + "N".join(f"{_i}_" for _i in total_n)).rstrip("_")  # label for saving
        # Plots
        for i_hue in hue_params:
            if i_settings['label'] == 'con_det':
                i_path = os.path.join(task_graph_path, f"{i_design}_{n_label}")
                make_paths([i_path])
                if not oei_df.empty:
                    _plot_oei = False
                    if _plot_oei is True:
                        load_seaborn_prefs(custom_rc)
                        plot_multicond(oei_df, i_hue, baseline_params, default_params, conds_paper_sf,
                                       i_settings['decoder_list'],
                                       beh_oei=i_settings['behavioural_oei'],
                                       y_lim=i_settings['y_lim_oe'], y_lim_log=i_settings['y_lim_oe_log'],
                                       save_dir=i_path, dpi=multicond_dpi,
                                       )
                        z = 0
                if not i_thresh.empty:
                    if _plot_threshold is True:
                        # Thresholds for each baseline parameter
                        i_thresh_path = os.path.join(i_path, "thresholds")
                        make_paths([i_thresh_path])
                        load_seaborn_prefs(custom_rc)
                        plot_multicond_thresholds(i_thresh, baseline_params, default_params, conds_paper_sf,
                                                  i_settings['decoder_list'],
                                                  beh_thresh=i_settings['behavioural_t'],
                                                  y_lim=i_settings['y_lim_t'], y_lim_log=i_settings['y_lim_t_log'],
                                                  y_label=f"Threshold (%)",
                                                  save_dir=i_thresh_path, dpi=multicond_dpi)
            elif i_settings['label'] == 'ori_dis':
                for i_con in [100]:
                    i_path = os.path.join(task_graph_path, f"{i_design}_{n_label}_con{i_con}")
                    make_paths([i_path])
                    # Contrast
                    default_params['contrast'] = i_con
                    oei_df = index_dframe(oei_df, ['contrast'], [i_con])
                    if not oei_df.empty:
                        if _plot_oei is True:
                            load_seaborn_prefs(custom_rc)
                            plot_multicond(oei_df, i_hue, baseline_params, default_params, conds_paper_sf,
                                           i_settings['decoder_list'],
                                           beh_oei=i_settings['behavioural_oei'],
                                           y_lim=i_settings['y_lim_oe'], y_lim_log=i_settings['y_lim_oe_log'],
                                           palette=sns.color_palette("colorblind"),
                                           save_dir=i_path, dpi=multicond_dpi)
                    # Thresholds
                    i_thresh = index_dframe(i_thresh, ['contrast'], [i_con])
                    if not i_thresh.empty:
                        if _plot_threshold is True:
                            i_thresh_path = os.path.join(i_path, "thresholds")
                            make_paths([i_thresh_path])
                            load_seaborn_prefs(custom_rc)
                            plot_multicond_thresholds(i_thresh, baseline_params, default_params, conds_paper_sf,
                                                      i_settings['decoder_list'],
                                                      beh_thresh=i_settings['behavioural_t'],
                                                      y_lim=i_settings['y_lim_t'], y_lim_log=i_settings['y_lim_t_log'],
                                                      y_label=f"Threshold ({misc_sym['deg']})",
                                                      palette=sns.color_palette("colorblind"),
                                                      save_dir=i_thresh_path, dpi=multicond_dpi)
