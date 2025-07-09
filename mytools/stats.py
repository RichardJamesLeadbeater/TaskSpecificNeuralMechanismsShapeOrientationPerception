import itertools as itls
import numpy as np
import pandas as pd
import scipy.stats as st
from mytools.dframe_tools import index_dframe
from mytools.dict_tools import init_cols, append_dicty


def descriptive_stats(dframe, cols, dv='threshold'):
    # input scores for each P on each cond
    # output descriptive statistics
    summary_stats = init_cols(cols + ['n', 'mean', 'stdev', 'sem', '95%CI', 'median', 'iqr', 'pnctl95', 'range'])
    # get combination of all possible conditions
    all_conds = list(itls.product(*[dframe[i].unique() for i in cols]))
    for i_condcombo in all_conds:  # loop through all cond combos
        i_data = index_dframe(dframe, cols, i_condcombo)
        # todo dv_label='threshold'
        ci95 = st.t.interval(0.95, len(i_data) - 1, loc=i_data[dv].mean(), scale=i_data[dv].sem())  # lower and upper bound of conf interval
        iqr = abs(np.subtract(*np.percentile(i_data[dv].to_numpy(), [25, 75])))
        pcntl95 = np.percentile(i_data[dv].to_numpy(), [2.5, 97.5])  # 95th percentile
        val_range = np.percentile(i_data[dv].to_numpy(), [0, 100])  # range of vals
        # append summary_stats dict, list of vals in same order as summary_stats
        d_stats = [len(i_data), i_data[dv].mean(), i_data[dv].std(), i_data[dv].sem(), ci95, i_data[dv].median(), iqr,
                   pcntl95, val_range]
        append_dicty(summary_stats, list(i_condcombo) + d_stats)
        z = 0
    summary_stats = pd.DataFrame(summary_stats)
    return summary_stats


def pairwise_comparisons(dframe, conds, dv):
    all_conds = list(itls.product(*[dframe[i].unique() for i in conds]))
    pairs = [[i, j] for i in all_conds for j in all_conds if i != j]  # remove repeats
    pairs = [i for idx, i in enumerate(pairs) for jdx, j in enumerate(pairs) if idx < jdx and i == j[::-1]]
    ttests = init_cols(['pair', 't', 'p', 'd_av'])
    for i_pair in pairs:
        data_1 = index_dframe(dframe, conds, i_pair[0])
        data_2 = index_dframe(dframe, conds, i_pair[1])
        paired_t = list(st.ttest_rel(data_1[dv], data_2[dv], alternative="two-sided"))
        t = paired_t[0]
        p = paired_t[1] * len(pairs)  # bonferroni correction
        if p > 1:  # p_vals cannot exceed 1
            p = 1
        elif p <= 0.001:
            p = '< 0.001'
        label = f"{''.join(i_pair[0])}_vs_{''.join(i_pair[1])}"  # cond1a_vs_cond1b
        cohens_d_av = paired_cohens_d(data_1[dv], data_2[dv])
        append_dicty(ttests, [label, t, p, cohens_d_av])
    ttests = pd.DataFrame(ttests)
    return ttests


def paired_cohens_d(x, y):
    # calcs d_av: see real-statistics.com/students-t-distribution/paired-sample-t-test/cohens-d-paired-samples/
    pooled_stdev = np.sqrt(np.var(x, ddof=1) + np.var(y, ddof=1) / 2)
    d_av = (np.mean(x) - np.mean(y)) / pooled_stdev
    return d_av
