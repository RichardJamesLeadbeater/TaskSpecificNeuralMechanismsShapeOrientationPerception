import numpy as np
import pandas as pd
from mytools.dframe_tools import index_dframe, get_cond_combos, change_dframe_labels
from mytools.dict_tools import init_cols, append_dicty
from mytools.ratios import get_ori_ratio, geomean

def get_mean_data(dframe, cond_list, dv):
    # Use groupby to calculate the statistics in one go
    grouped = dframe.groupby(cond_list)[dv]

    # Calculate n, mean, std, and sem for each group
    mean_data = grouped.agg(
        n='size',  # Count of rows in each group
        mean='mean',  # Mean of the dependent variable (dv)
        std='std',  # Standard deviation
        sem='sem'  # Standard error of the mean
    ).reset_index()

    # Sort the result by the condition columns
    mean_data = mean_data.sort_values(by=cond_list)

    return mean_data


def get_ratio_data(dframe, cond_list, ori_label, ratio_list=None, dv='threshold'):
    # Input mean_data / summary_data
    # Assumes that first value in cond_list is participant identifier
    if ratio_list is None:
        ratio_list = ['OEI']
    # INDIVIDUAL RATIOS #
    indiv_ratios = init_cols([*cond_list, 'ratio', 'score'])
    cond_combos = get_cond_combos(dframe, cond_list)
    for i_combo in cond_combos:
        i_dframe = index_dframe(dframe, cond_list, i_combo)
        if i_dframe.empty:
            continue
        for i_ratio in ratio_list:
            i_ratio_val = get_ori_ratio(i_dframe, i_ratio, dv, ori_label)
            append_dicty(indiv_ratios, [*i_combo, i_ratio, i_ratio_val])
    indiv_ratios = pd.DataFrame(indiv_ratios).sort_values(by=cond_list)
    # MEAN RATIOS #
    mean_cond_list = [*cond_list[1:], 'ratio']
    mean_ratios = init_cols([*mean_cond_list, 'n', 'score', 'se', 'CV'])
    cond_combos = get_cond_combos(indiv_ratios, mean_cond_list)
    for i_combo in cond_combos:
        i_dframe = index_dframe(indiv_ratios, mean_cond_list, i_combo)
        if i_dframe.empty:
            continue
        g_mean = geomean(i_dframe['score'])
        log_SD = np.std(np.log(i_dframe['score'].to_numpy()), ddof=0)
        g_SE = g_mean * (np.exp(log_SD / np.sqrt(len(i_dframe))) - 1)
        CV = (i_dframe['score'].std() / i_dframe['score'].mean()) * 100  # Coefficient of variation
        append_dicty(mean_ratios, [*i_combo, len(i_dframe), g_mean, g_SE, CV])
    mean_ratios = pd.DataFrame(mean_ratios).sort_values(by=mean_cond_list)
    return indiv_ratios, mean_ratios
