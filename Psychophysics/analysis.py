import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools as itls
import scipy.stats as st
from mytools.analysis import get_mean_data, get_ratio_data
from mytools.dframe_tools import init_cols, to_csv_pkl, index_dframe, \
    change_dframe_labels, custom_sort, change_dframe_colnames, spss_format
from mytools.dict_tools import append_dicty
from mytools.misc import add_cardinal_and_oblique
from mytools.ratios import geomean
from mytools.file_tools import unpickle
from mytools.my_seaborn import barplot_err, load_seaborn_prefs, plot_bar, plot_iqr, plot_line, init_markers
from mytools.string_tools import misc_symbols


# Paths #
og_dir = os.getcwd()
paths = {
    'Contrast Detection': {
        'raw_dir': os.path.join(og_dir, '..', 'Contrast Detection', 'ThesisAnalysis', 'all_data'),
    },
    'Orientation Discrimination': {
        'raw_dir':  os.path.join(og_dir, '..', 'Orientation Discrimination', 'Spatial Frequency', 'ThesisAnalysis',
                                 'all_data'),
    },
}

# Analysis settings #
participants = ['rjl', 'tl', 'pvm', 'kw', 'yb', 're', 'lj', 'mw']
sf = [2, 8, 16]
_geomean_thresh = True  # threshold from reversals using geometric mean

# Analyse each experiment #
for i_exp in paths:
    all_raw_dir = paths[i_exp]['raw_dir']
    ori_labels = ['horizontal', 'vertical', 'minus45', 'plus45', 'cardinal', 'oblique']
    # Paths for analysis of i_exp
    exp_dir = os.path.join(og_dir, i_exp)
    raw_dir = os.path.join(exp_dir, 'raw')
    sum_dir = os.path.join(exp_dir, 'summary')
    ratio_dir = os.path.join(exp_dir, 'ratio')
    stats_dir = os.path.join(exp_dir, 'stats')
    cpd_ratio_dir = os.path.join(ratio_dir, 'cpd')
    revs_dir = os.path.join(exp_dir, 'graphs', 'reversals')
    for i_path in [exp_dir, raw_dir, sum_dir, ratio_dir, cpd_ratio_dir, revs_dir, stats_dir]:
        if not os.path.exists(i_path):
            os.makedirs(i_path)

    # TODO Include thresholds from staircase

    # Raw data
    raw_data = pd.read_csv(os.path.join(raw_dir, "raw_anonymised.csv"))
    raw_data['spatial frequency'] = raw_data['spatial frequency'].astype(str)

    # SUMMARY DATA
    sum_cols = ['participant', 'orientation', 'spatial frequency', 'n_runs', 'threshold', 'sem', 'ci95']
    sum_data = init_cols(sum_cols)

    condition_cols = ['participant', 'orientation', 'spatial frequency']
    for i_p, i_ori, i_sf in itls.product(*(raw_data[col].unique() for col in condition_cols)):
        i_dframe = index_dframe(raw_data, ['participant', 'orientation', 'spatial frequency'],
                                [i_p, i_ori, i_sf], 'and')
        if i_dframe.empty:
            continue
        append_dicty(sum_data, [i_p, i_ori, i_sf, len(i_dframe), i_dframe['threshold'].mean(),
                                i_dframe['threshold'].sem(),
                                st.t.interval(0.95, len(i_dframe) - 1, loc=i_dframe['threshold'].mean(),
                                              scale=i_dframe['threshold'].sem())  # bounds of 95CI
                                ])
    sum_data = pd.DataFrame(sum_data).sort_values(by=sum_cols)
    sum_data = custom_sort(sum_data, ['participant', 'spatial frequency', 'orientation'],
                           ordered_labels=[participants, ['2', '8', '16'], ori_labels])
    to_csv_pkl(sum_data, sum_dir, 'summary', rnd=3, _pkl=False)

    # MEAN THRESHOLDS
    mean_data = get_mean_data(sum_data, cond_list=['orientation', 'spatial frequency'], dv='threshold')
    mean_data = custom_sort(mean_data, ['spatial frequency', 'orientation'],
                            ordered_labels=[['2', '8', '16'], ori_labels])
    to_csv_pkl(mean_data, sum_dir, 'mean_thresholds', rnd=3, _pkl=False)

    # RATIOS #
    ratio_data, geomean_r_data = get_ratio_data(sum_data, cond_list=['participant', 'spatial frequency'],
                                                ori_label='orientation', ratio_list=['OEI', 'HVI'], dv='threshold')
    ratio_data = custom_sort(ratio_data, ['ratio', 'participant', 'spatial frequency', 'orientation'],
                             ordered_labels=[['OEI', 'HVI'], participants, ['2', '8', '16'], ori_labels])
    geomean_r_data = custom_sort(geomean_r_data, ['ratio', 'participant', 'spatial frequency', 'orientation'],
                                 ordered_labels=[['OEI', 'HVI'], participants, ['2', '8', '16'], ori_labels])
    to_csv_pkl(ratio_data, ratio_dir, 'ratios', rnd=3, _pkl=False)
    to_csv_pkl(geomean_r_data, ratio_dir, 'geomean_ratios', rnd=3, _pkl=False)

    # THRESHOLD INCREASE #
    # # # RATIOS BETWEEN 2 & 16 cpd # # #
    cpd_data = init_cols(['participant', 'orientation', 'score'])
    for i_p in sum_data['participant'].unique():
        for i_ori in sum_data['orientation'].unique():
            i_dframe = index_dframe(sum_data, ['participant', 'orientation'], [i_p, i_ori])
            if i_dframe.empty:
                continue
            i_dframe_2cpd = index_dframe(i_dframe, ['spatial frequency'], ['2'])
            i_dframe_16cpd = index_dframe(i_dframe, ['spatial frequency'], ['16'])
            ratio_val = i_dframe_16cpd['threshold'].sum() / i_dframe_2cpd['threshold'].sum()
            append_dicty(cpd_data, [i_p, i_ori, ratio_val])
    cpd_data = pd.DataFrame(cpd_data).sort_values(by=['participant', 'orientation'])
    to_csv_pkl(cpd_data, cpd_ratio_dir, '_CPD_ratios', rnd=3, _pkl=False)

    # geomean 2 & 16 cpd ratios #
    mean_cpd_data = init_cols(['ratio', 'orientation', 'score', 'SE'])
    for i_ori in cpd_data['orientation'].unique():
        i_dframe = index_dframe(cpd_data, ['orientation'], [i_ori])
        g_mean = geomean(i_dframe['score'])
        log_SD = np.std(np.log(i_dframe['score'].to_numpy()), ddof=0)
        g_SE = g_mean * (np.exp(log_SD / np.sqrt(len(i_dframe))) - 1)
        append_dicty(mean_cpd_data, ['16/2', i_ori, geomean(i_dframe['score']), g_SE])
    mean_cpd_data = pd.DataFrame(mean_cpd_data).sort_values(by=['ratio', 'orientation'])
    # Sort xlabels
    mean_cpd_data = custom_sort(mean_cpd_data, ['orientation'], [['2', '8', '16']])
    to_csv_pkl(mean_cpd_data, cpd_ratio_dir, '_CPD_geomean_ratios', rnd=3, _pkl=False)

    # Statistics
    # 2-way repeated measures anova
    anova_data = sum_data.copy()
    # Filter data to include conditions of interest
    anova_data = index_dframe(anova_data, ['orientation'], ['horizontal', 'vertical', 'minus45', 'plus45'],
                              index_filter='or')
    anova_data = index_dframe(anova_data, ['spatial frequency'], [2, 16], index_filter='or')
    to_csv_pkl(anova_data, stats_dir, "anova_data", rnd=3, _pkl=False)

    # Format for use with SPSS
    # Summary data
    sum_data_spss = index_dframe(sum_data, ['spatial frequency'], [2, 16], 'or')  # filter exp conditions
    sum_data_spss = spss_format(sum_data_spss, ['orientation', 'spatial frequency'],
                                id_label='participant', dv_label='threshold')
    to_csv_pkl(sum_data_spss, stats_dir, f"spss_{i_exp}", _pkl=False)
    # Ratio data
    ratio_data_spss = index_dframe(ratio_data, ['spatial frequency'], [2, 16], 'or')  # filter exp conditions
    ratio_data_spss = spss_format(ratio_data_spss, ['ratio', 'spatial frequency'],
                                  id_label='participant', dv_label='score')
    to_csv_pkl(ratio_data_spss, stats_dir, f"spss_{i_exp}_ratios", _pkl=False)
    # ^ #

