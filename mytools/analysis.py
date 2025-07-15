import numpy as np
import pandas as pd
from mytools.dframe_tools import index_dframe, get_cond_combos, change_dframe_labels, unique_combos
from mytools.dict_tools import init_cols, append_dicty
from mytools.list_tools import all_combinations
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


def get_oei_data(input_df, cond_list, ori_label, dv='threshold'):
    # Input mean_data / summary_data
    df = input_df.copy()
    df = df.fillna('NaN')
    # INDIVIDUAL RATIOS #
    oei_data = init_cols([*cond_list, 'score'])
    cond_combos = get_cond_combos(df, cond_list)
    for i_combo in cond_combos:
        i_df = index_dframe(df, cond_list, i_combo)
        if i_df.empty:
            continue
        cardinal_threshold = i_df[(i_df[ori_label] == 0)][dv].iloc[0]
        oblique_threshold = i_df[(i_df[ori_label] == 45)][dv].iloc[0]
        try:
            append_dicty(oei_data, [*i_combo, oblique_threshold / cardinal_threshold])
        except:
            append_dicty(oei_data, [*i_combo, 'NaN'])
    oei_df = pd.DataFrame(oei_data).sort_values(by=cond_list)

    return oei_df


def gen_oei_data(input_df, cond_list, ori_label, dv_label='threshold'):
    df = input_df.copy()
    df.fillna(0, inplace=True)

    cardinal_df = index_dframe(df, ['ori_std'], [0])
    oblique_df = index_dframe(df, ['ori_std'], [45])
    # Ensure cond_nums are aligned
    if not all(np.asarray(cardinal_df['cond_num']) == np.asarray(oblique_df['cond_num'])):
        print(f"WARNING - COND NUMS DO NOT ALIGN DURING OEI CALCULATION")
    oei_col = oblique_df['dv'].to_numpy() / cardinal_df['dv'].to_numpy()
    oei_dframe = cardinal_df.copy()
    oei_dframe['dv'] = oei_col

    return oei_dframe


def gen_cond_nums(input_df, ignore_list):
    df = input_df.copy()
    df.fillna(0, inplace=True)
    # Keep columns not in ignore_list
    cond_list = [col for col in df.columns if col not in ignore_list]
    # Get unique combinations as tuples
    cond_combos = unique_combos(df, cond_list)
    # Create lookup table: combo (as tuple) → index
    combo_LUT = {combo: idx for idx, combo in enumerate(cond_combos)}
    # Map each row's condition to its corresponding index
    df['cond_num'] = df[cond_list].apply(lambda row: combo_LUT[tuple(row)], axis=1)

    return df
