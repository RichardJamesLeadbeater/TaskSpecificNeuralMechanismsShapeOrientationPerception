import os
import pandas as pd
import numpy as np
import itertools as itls
import scipy.stats as st
import statsmodels.formula.api as smf

from mytools.analysis import get_mean_data, get_ratio_data
from mytools.dframe_tools import init_cols, to_csv_pkl, index_dframe, custom_sort, spss_format, change_dframe_colnames
from mytools.dict_tools import append_dicty
from mytools.ratios import geomean


# Paths #
og_dir = os.getcwd()

# Analysis settings #
participants = [f"P{i + 1}" for i in range(8)]
sf = [2, 8, 16]

# Analyse each experiment #
for i_exp in ['Contrast Detection', 'Orientation Discrimination']:

    # Labels consistent with modelling data
    task = ''
    if i_exp == 'Contrast Detection':
        task = 'con_det'
    elif i_exp == 'Orientation Discrimination':
        task = 'ori_dis'

    # -
    ori_labels = ['horizontal', 'vertical', 'minus45', 'plus45', 'cardinal', 'oblique']

    # Paths for analysis of i_exp
    exp_dir = os.path.join(og_dir, i_exp)

    analysis_dir = os.path.join(exp_dir, 'analysis')

    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)

    raw_dir = os.path.join(exp_dir, 'raw_data')

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

    # ===================================
    # SUMMARY DATA
    # ===================================
    sum_data = pd.DataFrame(sum_data).sort_values(by=sum_cols)
    sum_data = custom_sort(sum_data, ['participant', 'spatial frequency', 'orientation'],
                           ordered_labels=[participants, ['2', '8', '16'], ori_labels])

    to_csv_pkl(sum_data, analysis_dir, f'summary_{task}', rnd=3, _pkl=False)

    # MEAN THRESHOLDS
    mean_data = get_mean_data(sum_data, cond_list=['orientation', 'spatial frequency'], dv='threshold')
    mean_data = custom_sort(mean_data, ['spatial frequency', 'orientation'],
                            ordered_labels=[['2', '8', '16'], ori_labels])

    to_csv_pkl(mean_data, analysis_dir, f'mean_thresholds_{task}', rnd=3, _pkl=False)

    # ===================================
    # RATIOS #
    # ===================================
    ratio_data, geomean_r_data = get_ratio_data(sum_data, cond_list=['participant', 'spatial frequency'],
                                                ori_label='orientation', ratio_list=['OEI', 'HVI'], dv='threshold')

    ratio_data = custom_sort(ratio_data, ['ratio', 'participant', 'spatial frequency', 'orientation'],
                             ordered_labels=[['OEI', 'HVI'], participants, ['2', '8', '16'], ori_labels])

    geomean_r_data = custom_sort(geomean_r_data, ['ratio', 'participant', 'spatial frequency', 'orientation'],
                                 ordered_labels=[['OEI', 'HVI'], participants, ['2', '8', '16'], ori_labels])

    to_csv_pkl(ratio_data, analysis_dir, f'ratios_{task}', rnd=3, _pkl=False)

    to_csv_pkl(geomean_r_data, analysis_dir, f'geomean_ratios_{task}', rnd=3, _pkl=False)

    # ===================================
    # LINEAR MIXED EFFECTS ANALYSIS
    # Fit a linear mixed-effects model with participant as a random intercept
    # ===================================

    # Retain only cardinal and oblique orientations, including all SFs
    threshold_data = index_dframe(sum_data, ['orientation'], ['cardinal', 'oblique'], index_filter='or')
    threshold_data = change_dframe_colnames(threshold_data, ['spatial frequency'], ['SF'])

    for ref_SF in ['2', '8', '16']:
        # Fit LME with 2 c/° and cardinal orientation as reference levels
        print(f"\ntask={i_exp} refSF={ref_SF}\n")

        threshold_model = smf.mixedlm(
            f'threshold ~ C(orientation, Treatment(reference="cardinal")) * '
            f'C(SF, Treatment(reference="{ref_SF}"))',
            data=threshold_data,
            groups=threshold_data['participant']
        )
        threshold_results = threshold_model.fit()

        print('\nLinear Mixed-Effects Model - Thresholds')
        print(threshold_results.summary())

        # Save threshold LME results
        with open(os.path.join(analysis_dir, f"{i_exp}_refSF{ref_SF}_thresholds_lme.txt"), 'w') as file:
            file.write(threshold_results.summary().as_text())
