from copy import copy, deepcopy
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray as arr
import re
from mytools.my_seaborn import load_seaborn_prefs, plot_line, init_markers
from mytools.dframe_tools import init_cols, to_csv_pkl, index_dframe, unique_combos, \
    change_dframe_colnames, change_dframe_labels, custom_sort
from mytools.dict_tools import append_dicty
from mytools.list_tools import all_combinations, force_to_list
from mytools.symbols import model_symbols, misc_symbols, lowercase_greek_alphabet, uppercase_greek_alphabet, \
    italics, bold
from mytools.os_tools import make_paths
import ast
import itertools as itls

pd.options.mode.chained_assignment = None  # default='warn'  # removes error messages


def task_info(fixed_params, variable_params, default_params, conds=None, decoder_list=None, **kwargs):
    # String symbols
    misc_sym = misc_symbols()
    m_sym = model_symbols()
    upper_greek = uppercase_greek_alphabet()
    lower_greek = lowercase_greek_alphabet()

    # # Load filters
    # oblique_fwhm_filter = kwargs.get('oblique_fwhm_filter')
    # fwhm_fraction_filter = kwargs.get('fwhm_fraction_filter')
    # noise_fraction_filter = kwargs.get('noise_fraction_filter')
    # contrast_filter = kwargs.get('contrast_filter')

    # Converts for clearer saving
    filename_params = {'propn_cardinal': 'pC',
                       'fwhm_fraction': 'bwC',
                       'noise_fraction': 'noise0',
                       'rmax_fraction': 'rmaxC',
                       }
    # Plot info for all combinations of fixed params for each hue (variable params)
    all_info = []
    # Plot information for each possible combination of fixed params
    fixed_params_combos = all_combinations([fixed_params[_k] for _k in fixed_params])
    for i_fixed_combo in fixed_params_combos:  # Loop through all fixed param combos
        for i_hue in variable_params:  # Loop through each set of variable parameters to set hue
            i_info = {'folder': i_hue, 'plot_label': "", 'decoder_list': decoder_list,
                      'x': 'cond', 'y': 'dv', 'hue': i_hue, 'hue_order': None,
                      'x_label': "Spatial Frequency", 'legend_title': m_sym[i_hue],
                      'fixed': default_params.copy(),  # Default params constant across all plots
                      'filter': {i_hue: variable_params[i_hue]},  # Add iteration of variable params (i.e., hue)
                      'conds': conds,
                      }
            plot_label = f"HUE_{filename_params.get(i_hue, '')}"  # init
            # Add iteration of fixed params
            for idx, key_f in enumerate(list(fixed_params)):
                # If set within conds then set fixed param to None (so that it can vary)
                cond_keys = list(set([_i for _keys in i_info['conds'].values() for _i in _keys]))  # Unique conds
                for _k in cond_keys:
                    i_info['fixed'][_k] = None
                # If variable param then don't fix
                if key_f == i_hue:  # Set as None if param is also hue
                    i_info['fixed'][key_f] = None  # Replace default with None if key_f in variable values
                    continue
                # Replace with fixed values
                i_info['fixed'][key_f] = i_fixed_combo[idx]
                # Update plot_label and path (skip if set to default)
                if i_info['fixed'][key_f] != default_params[key_f]:
                    plot_label = f"{plot_label}_FIX_{filename_params[key_f]}{i_fixed_combo[idx]}"
            # Add plot_label and path
            i_info['plot_label'] = plot_label
            all_info.append(i_info)  # Add to list of fig_info dicts

    return all_info

def add_comparison_col(input_df, input_comparison_df, x, ignore_cols=None,
                       dv_label='comparison', split_col=None):
    std_df = input_df.copy()
    comp_df = input_comparison_df.copy()
    if ignore_cols is None:
        ignore_cols = []
    cond_list = [_i for _i in comp_df.columns if not any(_j == _i for _j in ignore_cols)]
    # Split dataframe according to certain condition
    if split_col is None:
        split_col = 'NoneSplit'
        std_df[split_col] = 0
        comp_df[split_col] = 0
    split_data = []
    for i_split in comp_df[split_col].unique():
        i_df = index_dframe(std_df, [split_col], [i_split])
        i_comp_df = index_dframe(comp_df, [split_col], [i_split])
        if len(unique_combos(i_comp_df, cond_list)) == 1:
            comp_col = [i_comp_df['dv'].mean()] * len(i_df)
        else:
            scores = {}  # If changing model params on x-axis
            for i_x, i_oei in zip(i_comp_df[x], i_comp_df['dv']):
                scores[i_x] = i_oei
            comp_col = []
            for idx_row in range(len(i_df)):
                x_val = i_df[x].iloc[idx_row]
                comp_col.append(scores.get(x_val, 0))  # use 0 if doesn't exist
        i_df[dv_label] = comp_col
        split_data.append(i_df)
    df = pd.concat(split_data)

    return df


def check_params(cond_info, param_list, ignore_list=None):
    # Check that all params have been accounted for before applying fixed params
    if ignore_list is None:
        ignore_list = []
    fixed_params = list(cond_info['fixed'])
    variable_params = [cond_info['x'], cond_info['hue']]
    filters = []
    if cond_info.get('filter') is not None:
        filters = list(cond_info['filter'])
    conds_listy = []
    if cond_info.get('conds') is not None:
        for key in cond_info['conds']:
            [conds_listy.append(_p) for _p in cond_info['conds'][key] if not conds_listy.count(_p)]
    plot_params_list = fixed_params + variable_params + filters + conds_listy  # list of all params
    unaccounted = []
    for _i in param_list:
        if ignore_list.count(_i):  # ignore certain params e.g. label
            continue
        if not plot_params_list.count(_i):
            unaccounted.append(_i)
    if unaccounted:
        print(f"Parameters unaccounted for {cond_info['plot_label']}:")
        [print(f"\t{_i}") for _i in unaccounted]
        return False
    else:
        return True


def apply_filters(input_df, info, baseline=False, ignore=None):
    if ignore is None:
        ignore = []
    df = input_df.copy()
    # If filtering baseline data change fixed values
    if baseline is True:
        df = index_dframe(df, [info['baseline']], [True])
        if not df.empty:
            baseline_fixed = {}
            cond_list = [_i for _i in info['fixed'] if _i.count('tmp_') == 0]  # not template params, as set to 0
            [baseline_fixed.update({_i: info['fixed'][_i]}) for _i in cond_list]
            fixed_dict = deepcopy(baseline_fixed)
        else:
            return pd.DataFrame({})  # return empty dframe
    else:
        fixed_dict = info['fixed']

    # Catch any mismatches (i.e., no data for a given key) - doesn't account for specific cond mismatches
    test_df = df.copy()
    for key in fixed_dict:
        if fixed_dict.get(key) is None:  # Ignore if set to None
            continue
        if any(_i == key for _i in ignore):  # Ignore certain columns
            continue
        if all(_k != fixed_dict[key] for _k in test_df[key].unique()):  # Check if no vals match desired fixed value
            print(key)
        test_df = index_dframe(test_df, [key], [fixed_dict[key]])
        if test_df.empty:
            print(key)
        else:
            save_test_df = test_df.copy()

    # Apply filters
    cols = [_k for _k in fixed_dict if not any(_i == _k for _i in ignore)]  # Filter out certain conds
    cols = [_k for _k in cols if fixed_dict[_k] is not None]  # Filter any parameters set to None (e.g., variable ps)
    df = index_dframe(df, cols, [fixed_dict[_k] for _k in cols])
    if baseline is False:
        if info.get('conds') is not None:  # Assign labels for specific conditions
            df = assign_conds(df, info['conds'])
        if info.get('filter') is not None:
            for _col in info['filter']:
                if info['filter'].get(_col) is None:
                    continue
                df = index_dframe(df, [_col], info['filter'][_col], andor='or')
    if df.empty:
        z = 0
    return df


def alter_dframe(input_df, task=''):
    df = input_df.copy()
    if df.get('template_rmax') is not None:
        df['template_rmax'].fillna("[50, 50]", inplace=True)  # replace missing data with default vals
    # df.fillna(0, inplace=True)  # replace any nan with 0
    # Change label format for fwhm, noise and spont
    # for _i_col in ['fwhm', 'noise_scale', 'spont']:
    for _i_col in df.columns.to_list():
        if any(_ignore == _i_col for _ignore in ['boundaries', 'template_boundaries', 'design']):
            continue
        for _i_val in df[_i_col].unique():
            og_label = _i_val
            if not isinstance(_i_val, str):
                continue
            if all(char in _i_val for char in ['[', '.']):
                if ',' not in _i_val:
                    # Replace multiple spaces with a single space using regex
                    _i_val = re.sub(r'\s+', ' ', _i_val)
                    # Remove spaces after '[' and before ']'
                    _i_val = _i_val.replace('[ ', '[').replace(' ]', ']')
                    # Replace remaining spaces with ', ' to fix missing commas in decimals list
                    _i_val = _i_val.replace(' ', ', ')

            # Check if all required substrings are present
            if all(substring in _i_val for substring in ['[', ', ', ']']):
                # Remove np.float64(...) wrappers safely
                _clean_val = re.sub(r'np\.float64\((.*?)\)', r'\1', _i_val)
                # Safely parse string to Python literal (list)
                _parsed_val = ast.literal_eval(_clean_val)
                # Update DataFrame labels with first two elements of the parsed list
                df = change_dframe_labels(df, _i_col, [og_label], [f"{_parsed_val[:2]}"])

    # TotalN & PropnCardinal
    n_prefs = df['n_prefs'].to_list()
    total_n = []
    propn_cardinal = []
    for n in n_prefs:
        i_n_prefs = ast.literal_eval(n)
        i_total_n = np.sum(i_n_prefs)
        total_n.append(i_total_n)
        i_propn_cardinal = i_n_prefs[0] / i_total_n
        propn_cardinal.append(i_propn_cardinal)
    df['total_n'] = np.asarray(total_n) * 2
    df['propn_cardinal'] = np.round(propn_cardinal, 2)

    # FWHM fraction and oblique fwhm #
    bw_fractions = []
    oblique_fwhm = []
    for i_bw in df['fwhm'].to_list():
        i_bw = ast.literal_eval(i_bw)
        i_bw_fraction = np.round(i_bw[0] / i_bw[1], 2)
        bw_fractions.append(i_bw_fraction)
        oblique_fwhm.append(i_bw[1])
    df['fwhm_fraction'] = bw_fractions
    df['oblique_fwhm'] = oblique_fwhm

    # Cardinal spont #
    cardinal_spont = []
    oblique_spont = []
    for i_spont in df['spont'].to_list():
        i_spont = ast.literal_eval(i_spont)
        cardinal_spont.append(i_spont[0])
        oblique_spont.append(i_spont[1])
    df['cardinal_spont'] = cardinal_spont
    df['oblique_spont'] = oblique_spont

    # Noise fraction and oblique noise
    noise_fractions = []
    oblique_noise = []
    cardinal_noise = []
    for i_noise in df['noise_scale'].to_list():
        i_noise = ast.literal_eval(i_noise)
        i_noise_fraction = np.round(i_noise[0] / i_noise[1], 2)  # cardinal / oblique
        # i_noise_fraction = np.round(i_noise[1] / i_noise[0], 2)  # oblique / cardinal
        noise_fractions.append(i_noise_fraction)
        cardinal_noise.append(i_noise[0])
        oblique_noise.append(i_noise[1])
    df['noise_fraction'] = noise_fractions
    df['cardinal_noise'] = cardinal_noise
    df['oblique_noise'] = oblique_noise

    # rmax fraction and oblique rmax
    rmax_fractions = []
    oblique_rmax = []
    cardinal_rmax = []
    for i_rmax in df['rmax'].to_list():
        i_rmax = ast.literal_eval(i_rmax)
        i_rmax_fraction = np.round(i_rmax[0] / i_rmax[1], 2)  # cardinal / oblique
        rmax_fractions.append(i_rmax_fraction)
        oblique_rmax.append(i_rmax[1])
        cardinal_rmax.append(i_rmax[0])
    df['rmax_fraction'] = rmax_fractions
    df['oblique_rmax'] = oblique_rmax
    df['cardinal_rmax'] = cardinal_rmax

    # CRF parameters
    df['crf_c50'] = [ast.literal_eval(_i)[0] for _i in df['crf_c50'].to_list()]
    df['crf_exponent'] = [ast.literal_eval(_i)[0] for _i in df['crf_exponent'].to_list()]

    # TEMPLATE PARAMS #
    # if df.get('template_decoder') is not None:
    #     df['tmp_decoder'] = df['template_decoder']
    #     df = change_dframe_labels(df, 'tmp_decoder', ["['ML', 'ML']", "['PV', 'PV']", "['WTA', 'WTA']"],
    #                               ['ML', 'PV', 'WTA'])
    # else:
    #     df['tmp_decoder'] = None

    if task == '6. OriId':
        df = change_dframe_colnames(df, ['template_decoder'], ['tmp_decoder'])
        df = change_dframe_labels(df, 'tmp_decoder', ["['ML', 'ML']", "['PV', 'PV']", "['WTA', 'WTA']"],
                                  ['ML', 'PV', 'WTA'])

        # TemplateTotalN & PropnCardinal
        tmp_n_prefs = df['template_n_prefs'].to_list()
        tmp_total_n = []
        tmp_propn_cardinal = []
        for n in tmp_n_prefs:
            if not pd.notna(n):  # i.e. if nan
                [_i.append(0) for _i in [tmp_total_n, tmp_propn_cardinal]]
            else:
                i_n_prefs = ast.literal_eval(n)
                i_total_n = np.sum(i_n_prefs)
                tmp_total_n.append(i_total_n)
                i_propn_cardinal = i_n_prefs[0] / i_total_n
                tmp_propn_cardinal.append(i_propn_cardinal)
        df['tmp_total_n'] = np.asarray(tmp_total_n) * 2
        df['tmp_propn_cardinal'] = np.round(tmp_propn_cardinal, 1)

        # FWHM fraction and oblique fwhm #
        tmp_bw_fractions = []
        tmp_oblique_fwhm = []
        for i_bw in df['template_fwhm'].to_list():
            if not pd.notna(i_bw):  # i.e. if nan
                [_i.append(0) for _i in [tmp_bw_fractions, tmp_oblique_fwhm]]
            else:
                i_bw = ast.literal_eval(i_bw)
                i_bw_fraction = np.round(i_bw[0] / i_bw[1], 1)
                tmp_bw_fractions.append(i_bw_fraction)
                tmp_oblique_fwhm.append(i_bw[1])
        df['tmp_fwhm_fraction'] = tmp_bw_fractions
        df['tmp_oblique_fwhm'] = tmp_oblique_fwhm

        # Noise fraction and oblique noise
        tmp_noise_fractions = []
        tmp_oblique_noise = []
        tmp_cardinal_noise = []
        for i_noise in df['template_noise_scale'].to_list():
            if not pd.notna(i_noise):
                [_i.append(0) for _i in [tmp_noise_fractions, tmp_cardinal_noise, tmp_oblique_noise]]
            else:
                i_noise = ast.literal_eval(i_noise)
                i_noise_fraction = np.round(i_noise[1] / i_noise[0], 2)  # oblique / cardinal
                tmp_noise_fractions.append(i_noise_fraction)
                tmp_cardinal_noise.append(i_noise[0])
                tmp_oblique_noise.append(i_noise[1])
        df['tmp_noise_fraction'] = tmp_noise_fractions
        df['tmp_cardinal_noise'] = tmp_cardinal_noise
        df['tmp_oblique_noise'] = tmp_oblique_noise

        # Template rmax_o and rmax_fraction
        tmp_rmax_fractions = []
        tmp_oblique_rmax = []
        tmp_cardinal_rmax = []
        for rmax in df['template_rmax'].to_list():
            if not pd.notna(rmax):
                [_i.append(0) for _i in [tmp_oblique_rmax, tmp_cardinal_rmax]]
            else:
                i_rmax = ast.literal_eval(rmax)
                tmp_cardinal_rmax.append(i_rmax[0])
                tmp_oblique_rmax.append(i_rmax[1])
                tmp_rmax_fractions.append(np.round(i_rmax[0] / i_rmax[1], 2))
        df['tmp_cardinal_rmax'] = tmp_cardinal_rmax
        df['tmp_oblique_rmax'] = tmp_oblique_rmax
        df['tmp_rmax_fraction'] = tmp_rmax_fractions

        # TMP CONTRAST
        if df.get('template_contrast') is None:
            df['tmp_con'] = 40  # default
        else:
            df = change_dframe_colnames(df, ['template_contrast'], ['tmp_con'])

        # BASELINE
        df = change_dframe_colnames(df, ['template_baseline'], ['tmp_baseline'])
        if not any(_i == 'tmp_baseline' for _i in df.columns):
            df['tmp_baseline'] = None

    return df


def assign_conds(df, conds_dict):
    df = df.copy()
    df_list = []
    for i_key in conds_dict:
        i_conds = conds_dict[i_key]
        i_n_df = index_dframe(df, list(i_conds), list(i_conds.values()))
        i_n_df['cond'] = i_key
        df_list.append(i_n_df)
    df = pd.concat(df_list)
    return df


def task_info(fixed_params, variable_params, default_params, conds=None, decoder_list=None, **kwargs):
    # String symbols
    misc_sym = misc_symbols()
    m_sym = model_symbols()
    upper_greek = uppercase_greek_alphabet()
    lower_greek = lowercase_greek_alphabet()

    # # Load filters
    # oblique_fwhm_filter = kwargs.get('oblique_fwhm_filter')
    # fwhm_fraction_filter = kwargs.get('fwhm_fraction_filter')
    # noise_fraction_filter = kwargs.get('noise_fraction_filter')
    # contrast_filter = kwargs.get('contrast_filter')

    # Converts for clearer saving
    filename_params = {'propn_cardinal': 'pC',
                       'fwhm_fraction': 'bwC',
                       'noise_fraction': 'noise0',
                       'rmax_fraction': 'rmaxC',
                       }
    # Plot info for all combinations of fixed params for each hue (variable params)
    all_info = []
    # Plot information for each possible combination of fixed params
    fixed_params_combos = all_combinations([fixed_params[_k] for _k in fixed_params])
    for i_fixed_combo in fixed_params_combos:  # Loop through all fixed param combos
        for i_hue in variable_params:  # Loop through each set of variable parameters to set hue
            i_info = {'folder': i_hue, 'plot_label': "", 'decoder_list': decoder_list,
                      'x': 'cond', 'y': 'dv', 'hue': i_hue, 'hue_order': None,
                      'x_label': "Spatial Frequency", 'legend_title': m_sym[i_hue],
                      'fixed': default_params.copy(),  # Default params constant across all plots
                      'filter': {i_hue: variable_params[i_hue]},  # Add iteration of variable params (i.e., hue)
                      'conds': conds,
                      }
            plot_label = f"HUE_{filename_params.get(i_hue, '')}"  # init
            # Add iteration of fixed params
            for idx, key_f in enumerate(list(fixed_params)):
                # If set within conds then set fixed param to None (so that it can vary)
                cond_keys = list(set([_i for _keys in i_info['conds'].values() for _i in _keys]))  # Unique conds
                for _k in cond_keys:
                    i_info['fixed'][_k] = None
                # If variable param then don't fix
                if key_f == i_hue:  # Set as None if param is also hue
                    i_info['fixed'][key_f] = None  # Replace default with None if key_f in variable values
                    continue
                # Replace with fixed values
                i_info['fixed'][key_f] = i_fixed_combo[idx]
                # Update plot_label and path (skip if set to default)
                if i_info['fixed'][key_f] != default_params[key_f]:
                    plot_label = f"{plot_label}_FIX_{filename_params[key_f]}{i_fixed_combo[idx]}"
            # Add plot_label and path
            i_info['plot_label'] = plot_label
            all_info.append(i_info)  # Add to list of fig_info dicts

    return all_info


def oridiscon_info(**kwargs):
    misc_sym = misc_symbols()
    m_sym = model_symbols()
    upper_greek = uppercase_greek_alphabet()
    lower_greek = lowercase_greek_alphabet()
    contrast_filter = [2.5, 5, 10, 20, 40, 80]
    fwhm_fraction_order = [1, 0.8, 0.6, 0.4, 0.2]
    noise_fraction_order = [1, 2, 3, 4, 5]
    crf_exponent_order = [1.5, 1.75, 2, 2.5, 3]
    crf_c50_order = [6, 12, 18, 24, 30]
    oblique_fwhm_filter = kwargs.get('oblique_fwhm_filter')
    fwhm_fraction_filter = kwargs.get('fwhm_fraction_filter')
    noise_fraction_filter = kwargs.get('noise_fraction_filter')
    xlabel_con = 'Contrast (%)'
    info = {
        'single_cond': {
            'folder': 'N540', 'x': 'contrast', 'y': 'dv', 'hue': 'propn_cardinal', 'iter': 'decoder',
            'fixed': {'propn_cardinal': 0.7, 'fwhm_fraction': 0.4, 'oblique_fwhm': 50, 'total_n': 540,
                      'noise_fraction': 2, 'cardinal_noise': 1, 'spont': '[0.03, 0.03]',
                      'crf_c50': 24, 'crf_exponent': 2, 'rmax_fraction': 1, 'oblique_rmax': 50},
            'hue_order': None, 'x_label': xlabel_con, 'legend_title': None, 'plots': ['OEI'],
            'filter': {'contrast': contrast_filter},
        },
        'N540_pC': {
            'folder': 'N540', 'x': 'contrast', 'y': 'dv', 'hue': 'propn_cardinal', 'iter': 'decoder',
            'fixed': {'fwhm_fraction': 0.4, 'oblique_fwhm': 50, 'total_n': 540,
                      'noise_fraction': 2, 'cardinal_noise': 1, 'spont': '[0.03, 0.03]',
                      'crf_c50': 24, 'crf_exponent': 2, 'rmax_fraction': 1, 'oblique_rmax': 50},
            'hue_order': None, 'x_label': xlabel_con, 'legend_title': None, 'plots': ['thresholds', 'OEI'],
            'filter': {'contrast': contrast_filter},
        },
        'N540_bwC': {
            'folder': 'N540', 'x': 'contrast', 'y': 'dv', 'hue': 'fwhm_fraction', 'iter': 'decoder',
            'fixed': {'propn_cardinal': 0.7, 'oblique_fwhm': 50, 'total_n': 540,
                      'noise_fraction': 2, 'cardinal_noise': 1, 'spont': '[0.03, 0.03]',
                      'crf_c50': 24, 'crf_exponent': 2, 'rmax_fraction': 1, 'oblique_rmax': 50},
            'hue_order': fwhm_fraction_order, 'x_label': xlabel_con, 'legend_title': None, 'plots': ['thresholds', 'OEI'],
            'filter': {'contrast': contrast_filter},
        },
        'N540_nzO': {
            'folder': 'N540', 'x': 'contrast', 'y': 'dv', 'hue': 'noise_fraction', 'iter': 'decoder',
            'fixed': {'propn_cardinal': 0.7, 'fwhm_fraction': 0.4, 'oblique_fwhm': 50, 'total_n': 540,
                      'cardinal_noise': 1, 'spont': '[0.03, 0.03]',
                      'crf_c50': 24, 'crf_exponent': 2, 'rmax_fraction': 1, 'oblique_rmax': 50},
            'hue_order': noise_fraction_order, 'x_label': xlabel_con, 'legend_title': None,
            'plots': ['OEI', 'thresholds'],
            'filter': {'contrast': contrast_filter, 'noise_fraction': noise_fraction_filter},
        },
        'N540_nz_lvl': {
            'folder': 'N540', 'label': None, 'x': 'contrast', 'y': 'dv', 'hue': 'cardinal_noise', 'iter': 'decoder',
            'fixed': {'propn_cardinal': 0.7, 'fwhm_fraction': 0.4, 'oblique_fwhm': 50, 'total_n': 540,
                      'noise_fraction': 2, 'spont': '[0.03, 0.03]',
                      'crf_c50': 24, 'crf_exponent': 2, 'rmax_fraction': 1, 'oblique_rmax': 50},
            'hue_order': None, 'x_label': None, 'legend_title': m_sym['noise_scale'], 'plots': ['thresholds', 'OEI'],
            'filter': {'contrast': contrast_filter}
            # , 'cardinal_noise': ['0.25', '0.5', '1.0', '2.0', '4.0']
        },
        'N540_rmax': {
            'folder': 'N540', 'label': None, 'x': 'contrast', 'y': 'dv', 'hue': 'rmax_fraction', 'iter': 'decoder',
            'fixed': {'propn_cardinal': 0.5, 'fwhm_fraction': 1, 'oblique_fwhm': 50, 'total_n': 540,
                      'cardinal_noise': 1, 'noise_fraction': 1, 'spont': '[0.03, 0.03]',
                      'crf_c50': 24, 'crf_exponent': 2},
            'hue_order': None, 'x_label': None, 'legend_title': m_sym['rmax_fraction'], 'plots': ['thresholds', 'OEI'],
            'filter': {'contrast': contrast_filter}
            # , 'cardinal_noise': ['0.25', '0.5', '1.0', '2.0', '4.0']
        },
        'reverse_rmax': {
            'folder': 'rmax', 'label': None, 'x': 'contrast', 'y': 'dv', 'hue': 'rmax_fraction', 'iter': 'decoder',
            'fixed': {'propn_cardinal': 0.5, 'fwhm_fraction': 1, 'oblique_fwhm': 50, 'total_n': 180,
                      'cardinal_noise': 1, 'noise_fraction': 1, 'spont': '[0.03, 0.03]',
                      'crf_c50': 24, 'crf_exponent': 2, 'oblique_rmax': 50},
            'hue_order': None, 'x_label': None, 'legend_title': m_sym['rmax_fraction'], 'plots': ['thresholds', 'OEI'],
            'filter': {'contrast': contrast_filter}, 'legend': True,
            # , 'cardinal_noise': ['0.25', '0.5', '1.0', '2.0', '4.0']
        },
        'fwhm': {
            'folder': 'fwhm', 'label': None, 'x': 'contrast', 'y': 'dv', 'hue': 'oblique_fwhm', 'iter': 'decoder',
            'fixed': {'propn_cardinal': 0.7, 'fwhm_fraction': 0.4, 'total_n': 180,
                      'cardinal_noise': 1, 'noise_fraction': 2, 'spont': '[0.03, 0.03]',
                      'crf_c50': 24, 'crf_exponent': 2, 'rmax_fraction': 1, 'oblique_rmax': 50},
            'hue_order': None, 'x_label': None, 'legend_title': m_sym['noise_scale'], 'plots': ['thresholds', 'OEI'],
            'legend': True,
            'filter': {'contrast': contrast_filter}
        },
        '1_': {
            'folder': 'pC', 'label': None, 'x': 'contrast', 'y': 'dv', 'hue': 'propn_cardinal', 'iter': 'decoder',
            'fixed': {'fwhm_fraction': 1, 'oblique_fwhm': 50, 'total_n': 180,
                      'noise_fraction': 1, 'cardinal_noise': 1, 'spont': '[0.03, 0.03]',
                      'crf_c50': 24, 'crf_exponent': 2, 'rmax_fraction': 1, 'oblique_rmax': 50},
            'hue_order': None, 'x_label': xlabel_con, 'legend_title': None, 'plots': ['OEI'],
            'filter': {'contrast': contrast_filter},
        },
        '1_bwC': {
            'folder': 'pC', 'label': None, 'x': 'contrast', 'y': 'dv', 'hue': 'propn_cardinal', 'iter': 'decoder',
            'fixed': {'fwhm_fraction': 0.4, 'oblique_fwhm': 50, 'total_n': 180,
                      'noise_fraction': 1, 'cardinal_noise': 1, 'spont': '[0.03, 0.03]',
                      'crf_c50': 24, 'crf_exponent': 2, 'rmax_fraction': 1, 'oblique_rmax': 50},
            'hue_order': None, 'x_label': xlabel_con, 'legend_title': None, 'plots': ['OEI'],
            'filter': {'contrast': contrast_filter},
        },
        '1_nzO': {
            'folder': 'pC', 'label': None, 'x': 'contrast', 'y': 'dv', 'hue': 'propn_cardinal', 'iter': 'decoder',
            'fixed': {'fwhm_fraction': 1, 'oblique_fwhm': 50, 'total_n': 180,
                      'noise_fraction': 2, 'cardinal_noise': 1, 'spont': '[0.03, 0.03]',
                      'crf_c50': 24, 'crf_exponent': 2, 'rmax_fraction': 1, 'oblique_rmax': 50},
            'hue_order': None, 'x_label': xlabel_con, 'legend_title': None, 'plots': ['OEI'],
            'filter': {'contrast': contrast_filter},
        },
        '1_all': {
            'folder': 'pC', 'label': None, 'x': 'contrast', 'y': 'dv', 'hue': 'propn_cardinal', 'iter': 'decoder',
            'fixed': {'fwhm_fraction': 0.4, 'oblique_fwhm': 50, 'total_n': 180,
                      'noise_fraction': 2, 'cardinal_noise': 1, 'spont': '[0.03, 0.03]',
                      'crf_c50': 24, 'crf_exponent': 2, 'rmax_fraction': 1, 'oblique_rmax': 50},
            'hue_order': None, 'x_label': xlabel_con, 'legend_title': None, 'plots': ['thresholds', 'OEI'],
            'filter': {'contrast': contrast_filter},
        },
        '2_': {
            'folder': 'bwC', 'label': None, 'x': 'contrast', 'y': 'dv', 'hue': 'fwhm_fraction', 'iter': 'decoder',
            'fixed': {'propn_cardinal': 0.5, 'oblique_fwhm': 50, 'total_n': 180,
                      'noise_fraction': 1, 'cardinal_noise': 1, 'spont': '[0.03, 0.03]',
                      'crf_c50': 24, 'crf_exponent': 2, 'rmax_fraction': 1, 'oblique_rmax': 50},
            'hue_order': fwhm_fraction_order, 'x_label': xlabel_con, 'legend_title': None, 'plots': ['OEI'],
            'filter': {'contrast': contrast_filter},
        },
        '2_pC': {
            'folder': 'bwC', 'label': None, 'x': 'contrast', 'y': 'dv', 'hue': 'fwhm_fraction', 'iter': 'decoder',
            'fixed': {'propn_cardinal': 0.7, 'oblique_fwhm': 50, 'total_n': 180,
                      'noise_fraction': 1, 'cardinal_noise': 1, 'spont': '[0.03, 0.03]',
                      'crf_c50': 24, 'crf_exponent': 2, 'rmax_fraction': 1, 'oblique_rmax': 50},
            'hue_order': fwhm_fraction_order, 'x_label': xlabel_con, 'legend_title': None, 'plots': ['OEI'],
            'filter': {'contrast': contrast_filter},
        },
        '2_nz0': {
            'folder': 'bwC', 'label': None, 'x': 'contrast', 'y': 'dv', 'hue': 'fwhm_fraction', 'iter': 'decoder',
            'fixed': {'propn_cardinal': 0.5, 'oblique_fwhm': 50, 'total_n': 180,
                      'noise_fraction': 2, 'cardinal_noise': 1, 'spont': '[0.03, 0.03]',
                      'crf_c50': 24, 'crf_exponent': 2, 'rmax_fraction': 1, 'oblique_rmax': 50},
            'hue_order': fwhm_fraction_order, 'x_label': xlabel_con, 'legend_title': None, 'plots': ['OEI'],
            'filter': {'contrast': contrast_filter},
        },
        '2_all': {
            'folder': 'bwC', 'label': None, 'x': 'contrast', 'y': 'dv', 'hue': 'fwhm_fraction', 'iter': 'decoder',
            'fixed': {'propn_cardinal': 0.7, 'oblique_fwhm': 50, 'total_n': 180,
                      'noise_fraction': 2, 'cardinal_noise': 1, 'spont': '[0.03, 0.03]',
                      'crf_c50': 24, 'crf_exponent': 2, 'rmax_fraction': 1, 'oblique_rmax': 50},
            'hue_order': fwhm_fraction_order, 'x_label': xlabel_con, 'legend_title': None, 'plots': ['thresholds', 'OEI'],
            'filter': {'contrast': contrast_filter},
        },
        '2_all_fwhm30': {
            'folder': 'bwC', 'label': None, 'x': 'contrast', 'y': 'dv', 'hue': 'fwhm_fraction', 'iter': 'decoder',
            'fixed': {'propn_cardinal': 0.7, 'oblique_fwhm': 30, 'total_n': 180,
                      'noise_fraction': 1, 'cardinal_noise': 1, 'spont': '[0.03, 0.03]',
                      'crf_c50': 24, 'crf_exponent': 2, 'rmax_fraction': 1, 'oblique_rmax': 50},
            'hue_order': fwhm_fraction_order, 'x_label': xlabel_con, 'legend_title': None, 'plots': ['thresholds', 'OEI'],
            'filter': {'contrast': contrast_filter},
        },
        '2_all_fwhm70': {
            'folder': 'bwC', 'label': None, 'x': 'contrast', 'y': 'dv', 'hue': 'fwhm_fraction', 'iter': 'decoder',
            'fixed': {'propn_cardinal': 0.7, 'oblique_fwhm': 70, 'total_n': 180,
                      'noise_fraction': 1, 'cardinal_noise': 1, 'spont': '[0.03, 0.03]',
                      'crf_c50': 24, 'crf_exponent': 2, 'rmax_fraction': 1, 'oblique_rmax': 50},
            'hue_order': fwhm_fraction_order, 'x_label': xlabel_con, 'legend_title': None, 'plots': ['thresholds', 'OEI'],
            'filter': {'contrast': contrast_filter},
        },
        '3_': {
            'folder': 'nzO', 'label': None, 'x': 'contrast', 'y': 'dv', 'hue': 'noise_fraction', 'iter': 'decoder',
            'fixed': {'propn_cardinal': 0.5, 'fwhm_fraction': 1, 'oblique_fwhm': 50, 'total_n': 180,
                      'cardinal_noise': 1, 'spont': '[0.03, 0.03]',
                      'crf_c50': 24, 'crf_exponent': 2, 'rmax_fraction': 1, 'oblique_rmax': 50},
            'hue_order': None, 'x_label': xlabel_con, 'legend_title': None, 'plots': ['OEI'],
            'filter': {'contrast': contrast_filter}
        },
        '3_pC': {
            'folder': 'nzO', 'label': None, 'x': 'contrast', 'y': 'dv', 'hue': 'noise_fraction', 'iter': 'decoder',
            'fixed': {'propn_cardinal': 0.7, 'fwhm_fraction': 1, 'oblique_fwhm': 50, 'total_n': 180,
                      'cardinal_noise': 1, 'spont': '[0.03, 0.03]',
                      'crf_c50': 24, 'crf_exponent': 2, 'rmax_fraction': 1, 'oblique_rmax': 50},
            'hue_order': None, 'x_label': xlabel_con, 'legend_title': None, 'plots': ['OEI'],
            'filter': {'contrast': contrast_filter}
        },
        '3_bwC': {
            'folder': 'nzO', 'label': None, 'x': 'contrast', 'y': 'dv', 'hue': 'noise_fraction', 'iter': 'decoder',
            'fixed': {'propn_cardinal': 0.5, 'fwhm_fraction': 0.4, 'oblique_fwhm': 50, 'total_n': 180,
                      'cardinal_noise': 1, 'spont': '[0.03, 0.03]',
                      'crf_c50': 24, 'crf_exponent': 2, 'rmax_fraction': 1, 'oblique_rmax': 50},
            'hue_order': None, 'x_label': xlabel_con, 'legend_title': None, 'plots': ['OEI'],
            'filter': {'contrast': contrast_filter}
        },
        '3_all': {
            'folder': 'nzO', 'label': None, 'x': 'contrast', 'y': 'dv', 'hue': 'noise_fraction', 'iter': 'decoder',
            'fixed': {'propn_cardinal': 0.7, 'fwhm_fraction': 0.4, 'oblique_fwhm': 50, 'total_n': 180,
                      'cardinal_noise': 1, 'spont': '[0.03, 0.03]',
                      'crf_c50': 24, 'crf_exponent': 2, 'rmax_fraction': 1, 'oblique_rmax': 50},
            'hue_order': noise_fraction_filter, 'x_label': xlabel_con,
            'legend_title': m_sym['noise_fraction'], 'plots': ['thresholds', 'OEI'],
            'filter': {'contrast': contrast_filter, 'noise_fraction': noise_fraction_filter}
        },
        '3_all_lowNoise': {
            'folder': 'nzO', 'label': None, 'x': 'contrast', 'y': 'dv', 'hue': 'noise_fraction', 'iter': 'decoder',
            'fixed': {'propn_cardinal': 0.7, 'fwhm_fraction': 0.4, 'oblique_fwhm': 50, 'total_n': 180,
                      'cardinal_noise': 0.25, 'spont': '[0.03, 0.03]',
                      'crf_c50': 24, 'crf_exponent': 2, 'rmax_fraction': 1, 'oblique_rmax': 50},
            'hue_order': noise_fraction_filter, 'x_label': None, 'legend_title': None, 'plots': ['thresholds', 'OEI'],
            'filter': {'contrast': contrast_filter, 'noise_fraction': noise_fraction_filter}
        },
        '3_all_highNoise': {
            'folder': 'nzO', 'label': None, 'x': 'contrast', 'y': 'dv', 'hue': 'noise_fraction', 'iter': 'decoder',
            'fixed': {'propn_cardinal': 0.7, 'fwhm_fraction': 0.4, 'oblique_fwhm': 50, 'total_n': 180,
                      'cardinal_noise': 2, 'spont': '[0.03, 0.03]',
                      'crf_c50': 24, 'crf_exponent': 2, 'rmax_fraction': 1, 'oblique_rmax': 50},
            'hue_order': noise_fraction_filter, 'x_label': None, 'legend_title': None, 'plots': ['thresholds', 'OEI'],
            'filter': {'contrast': contrast_filter, 'noise_fraction': noise_fraction_filter}
        },
        '3_unif': {
            'folder': 'nzO', 'label': None, 'x': 'contrast', 'y': 'dv', 'hue': 'cardinal_noise', 'iter': 'decoder',
            'fixed': {'propn_cardinal': 0.7, 'fwhm_fraction': 0.4, 'oblique_fwhm': 50, 'total_n': 180,
                      'noise_fraction': 1, 'spont': '[0.03, 0.03]',
                      'crf_c50': 24, 'crf_exponent': 2, 'rmax_fraction': 1, 'oblique_rmax': 50},
            'hue_order': None, 'x_label': None, 'legend_title': None, 'plots': ['OEI'],
            'filter': {'contrast': contrast_filter}
        },
        'nz_lvl': {
            'folder': 'nz_lvl', 'label': None, 'x': 'contrast', 'y': 'dv', 'hue': 'cardinal_noise', 'iter': 'decoder',
            'fixed': {'propn_cardinal': 0.7, 'fwhm_fraction': 0.4, 'oblique_fwhm': 50, 'total_n': 180,
                      'noise_fraction': 2, 'spont': '[0.03, 0.03]',
                      'crf_c50': 24, 'crf_exponent': 2, 'rmax_fraction': 1, 'oblique_rmax': 50},
            'hue_order': None, 'x_label': None, 'legend_title': m_sym['noise_scale'], 'plots': ['thresholds', 'OEI'],
            'filter': {'contrast': contrast_filter, 'cardinal_noise': [f"{_i}" for _i in [0.25, 0.5, 1, 2, 4]]}
        },
        'totalN_all': {
            'folder': 'totalN', 'label': None, 'x': 'contrast', 'y': 'dv', 'hue': 'total_n', 'iter': 'decoder',
            'fixed': {'propn_cardinal': 0.7, 'fwhm_fraction': 0.4, 'oblique_fwhm': 50,
                      'noise_fraction': 2, 'cardinal_noise': 1, 'spont': '[0.03, 0.03]',
                      'crf_c50': 24, 'crf_exponent': 2, 'rmax_fraction': 1, 'oblique_rmax': 50},
            'hue_order': None, 'x_label': None, 'legend_title': None, 'plots': ['thresholds', 'OEI'],
            'filter': {'contrast': contrast_filter},
        },
        'totalN_bwC_pC': {
            'folder': 'totalN', 'label': None, 'x': 'contrast', 'y': 'dv', 'hue': 'total_n', 'iter': 'decoder',
            'fixed': {'propn_cardinal': 0.7, 'fwhm_fraction': 0.4, 'oblique_fwhm': 50,
                      'noise_fraction': 1, 'cardinal_noise': 1, 'spont': '[0.03, 0.03]',
                      'crf_c50': 24, 'crf_exponent': 2, 'rmax_fraction': 1, 'oblique_rmax': 50},
            'hue_order': None, 'x_label': None, 'legend_title': None, 'plots': ['OEI'],
            'filter': {'contrast': contrast_filter},
        },
        'crf_exponent_lowcontrasts_n180': {
            'folder': 'crf', 'label': None, 'x': 'contrast', 'y': 'dv', 'hue': 'crf_exponent', 'iter': 'decoder',
            'fixed': {'propn_cardinal': 0.7, 'fwhm_fraction': 0.4, 'oblique_fwhm': 50, 'total_n': 180,
                      'noise_fraction': 2, 'cardinal_noise': 1, 'spont': '[0.03, 0.03]',
                      'crf_c50': 24, 'rmax_fraction': 1, 'oblique_rmax': 50},
            'hue_order': None, 'x_label': None, 'legend_title': None, 'plots': ['OEI'],
            'filter': {'contrast': [1.25 / 4, 1.25 / 2, 1.25, 2.5, 5, 10]}
        },
        'crf_exponent_lowcontrasts_n540': {
            'folder': 'crf', 'label': None, 'x': 'contrast', 'y': 'dv', 'hue': 'crf_exponent', 'iter': 'decoder',
            'fixed': {'propn_cardinal': 0.7, 'fwhm_fraction': 0.4, 'oblique_fwhm': 50, 'total_n': 540,
                      'noise_fraction': 2, 'cardinal_noise': 1, 'spont': '[0.03, 0.03]',
                      'crf_c50': 24, 'rmax_fraction': 1, 'oblique_rmax': 50},
            'hue_order': None, 'x_label': None, 'legend_title': None, 'plots': ['OEI'],
            'filter': {'contrast': [1.25 / 4, 1.25 / 2, 1.25, 2.5, 5, 10]}
        },
        'crf_exponent_solo': {
            'folder': 'crf', 'label': None, 'x': 'contrast', 'y': 'dv', 'hue': 'crf_exponent', 'iter': 'decoder',
            'fixed': {'propn_cardinal': 0.5, 'fwhm_fraction': 1, 'oblique_fwhm': 50, 'total_n': 180,
                      'noise_fraction': 1, 'cardinal_noise': 1, 'spont': '[0.03, 0.03]',
                      'crf_c50': 24, 'rmax_fraction': 1, 'oblique_rmax': 50},
            'hue_order': crf_exponent_order, 'x_label': None, 'legend_title': None, 'plots': ['thresholds'],
            'filter': {'contrast': contrast_filter}
        },
        'crf_exponent_all': {
            'folder': 'crf', 'label': None, 'x': 'contrast', 'y': 'dv', 'hue': 'crf_exponent', 'iter': 'decoder',
            'fixed': {'propn_cardinal': 0.7, 'fwhm_fraction': 0.4, 'oblique_fwhm': 50, 'total_n': 180,
                      'noise_fraction': 2, 'cardinal_noise': 1, 'spont': '[0.03, 0.03]',
                      'crf_c50': 24, 'rmax_fraction': 1, 'oblique_rmax': 50},
            'hue_order': crf_exponent_order, 'x_label': None, 'legend_title': None, 'plots': ['OEI'],
            'filter': {'contrast': contrast_filter}
        },
        'crf_exponent_all_N1260': {
            'folder': 'crf', 'label': None, 'x': 'contrast', 'y': 'dv', 'hue': 'crf_exponent', 'iter': 'decoder',
            'fixed': {'propn_cardinal': 0.7, 'fwhm_fraction': 0.4, 'oblique_fwhm': 50, 'total_n': 1260,
                      'noise_fraction': 2, 'cardinal_noise': 1, 'spont': '[0.03, 0.03]',
                      'crf_c50': 24, 'rmax_fraction': 1, 'oblique_rmax': 50},
            'hue_order': crf_exponent_order, 'x_label': None, 'legend_title': None, 'plots': ['thresholds', 'OEI'],
            'filter': {'contrast': contrast_filter}
        },
        'crf_exponent_bwC_pC': {
            'folder': 'crf', 'label': None, 'x': 'contrast', 'y': 'dv', 'hue': 'crf_exponent', 'iter': 'decoder',
            'fixed': {'propn_cardinal': 0.7, 'fwhm_fraction': 0.4, 'oblique_fwhm': 50, 'total_n': 180,
                      'noise_fraction': 1, 'cardinal_noise': 1, 'spont': '[0.03, 0.03]',
                      'crf_c50': 24, 'rmax_fraction': 1, 'oblique_rmax': 50},
            'hue_order': crf_exponent_order, 'x_label': None, 'legend_title': None, 'plots': ['thresholds', 'OEI'],
            'filter': {'contrast': contrast_filter}
        },
        'crf_c50_solo': {
            'folder': 'crf', 'label': None, 'x': 'contrast', 'y': 'dv', 'hue': 'crf_c50', 'iter': 'decoder',
            'fixed': {'propn_cardinal': 0.5, 'fwhm_fraction': 1, 'oblique_fwhm': 50, 'total_n': 180,
                      'noise_fraction': 1, 'cardinal_noise': 1, 'spont': '[0.03, 0.03]',
                      'crf_exponent': 2, 'rmax_fraction': 1, 'oblique_rmax': 50},
            'hue_order': None, 'x_label': None, 'legend_title': None, 'plots': ['thresholds'],
            'filter': {'contrast': contrast_filter}
        },
        'crf_c50_all': {
            'folder': 'crf', 'label': None, 'x': 'contrast', 'y': 'dv', 'hue': 'crf_c50', 'iter': 'decoder',
            'fixed': {'propn_cardinal': 0.7, 'fwhm_fraction': 0.4, 'oblique_fwhm': 50, 'total_n': 180,
                      'noise_fraction': 2, 'cardinal_noise': 1, 'spont': '[0.03, 0.03]',
                      'crf_exponent': 2, 'rmax_fraction': 1, 'oblique_rmax': 50},
            'hue_order': crf_c50_order, 'x_label': None, 'legend_title': None, 'plots': ['thresholds', 'OEI'],
            'filter': {'contrast': contrast_filter}
        },
        'crf_c50_N1260': {
            'folder': 'crf', 'label': None, 'x': 'contrast', 'y': 'dv', 'hue': 'crf_c50', 'iter': 'decoder',
            'fixed': {'propn_cardinal': 0.7, 'fwhm_fraction': 0.4, 'oblique_fwhm': 50, 'total_n': 1260,
                      'noise_fraction': 2, 'cardinal_noise': 1, 'spont': '[0.03, 0.03]',
                      'crf_exponent': 2, 'rmax_fraction': 1, 'oblique_rmax': 50},
            'hue_order': crf_c50_order, 'x_label': None, 'legend_title': None, 'plots': ['thresholds', 'OEI'],
            'filter': {'contrast': contrast_filter}
        },
        'crf_c50_bwC_pC': {
            'folder': 'crf', 'label': None, 'x': 'contrast', 'y': 'dv', 'hue': 'crf_c50', 'iter': 'decoder',
            'fixed': {'propn_cardinal': 0.7, 'fwhm_fraction': 0.4, 'oblique_fwhm': 50, 'total_n': 180,
                      'noise_fraction': 1, 'cardinal_noise': 1, 'spont': '[0.03, 0.03]',
                      'crf_exponent': 2, 'rmax_fraction': 1, 'oblique_rmax': 50},
            'hue_order': crf_c50_order, 'x_label': None, 'legend_title': None, 'plots': ['thresholds', 'OEI'],
            'filter': {'contrast': contrast_filter}
        },
        'rmax_all': {
            'folder': 'rmax', 'x': 'contrast', 'y': 'dv', 'hue': 'rmax', 'iter': 'decoder',
            'fixed': {'propn_cardinal': 0.7, 'fwhm_fraction': 0.4, 'oblique_fwhm': 50, 'total_n': 180,
                      'noise_fraction': 2, 'cardinal_noise': 1, 'spont': '[0.03, 0.03]',
                      'crf_c50': 24, 'crf_exponent': 2},
            'hue_order': None, 'x_label': None, 'legend_title': None, 'plots': ['thresholds', 'OEI'],
            'filter': {'contrast': contrast_filter}
        },
        'rmax_': {
            'folder': 'rmax', 'x': 'contrast', 'y': 'dv', 'hue': 'oblique_rmax', 'iter': 'decoder',
            'fixed': {'propn_cardinal': 0.5, 'fwhm_fraction': 1, 'oblique_fwhm': 50, 'total_n': 180,
                      'noise_fraction': 1, 'cardinal_noise': 1, 'spont': '[0.03, 0.03]',
                      'crf_c50': 24, 'crf_exponent': 2},
            'hue_order': [50, 40, 30, 20, 10], 'x_label': None, 'legend_title': m_sym['o_rmax'],
            'plots': ['thresholds', 'OEI'],
            'legend': True,
            'filter': {'contrast': contrast_filter, 'rmax_fraction': [1, 0.8, 0.6, 0.4, 0.2]}
        },
        'rmax_final': {
            'folder': 'rmax', 'x': 'contrast', 'y': 'dv', 'hue': 'rmax_fraction', 'iter': 'decoder',
            'fixed': {'propn_cardinal': 0.5, 'fwhm_fraction': 1, 'oblique_fwhm': 50, 'total_n': 180,
                      'noise_fraction': 1, 'cardinal_noise': 1, 'spont': '[0.03, 0.03]',
                      'crf_c50': 24, 'crf_exponent': 2},
            'hue_order': [1, 1.2, 1.4, 1.6, 1.8], 'x_label': None, 'legend_title': f"{m_sym['rmax_fraction']}", 'plots': ['thresholds', 'OEI'],
            'filter': {'contrast': contrast_filter}
        },
        # 'NEXT_N20': {
        #     'folder': 'Paper_N', 'x': 'contrast', 'y': 'dv', 'hue': 'fwhm_fraction', 'iter': 'decoder',
        #     'fixed': {'propn_cardinal': 0.5, 'total_n': 20,
        #               'cardinal_noise': 1, 'noise_fraction': 1, 'spont': '[0.03, 0.03]',
        #               'crf_c50': 24, 'crf_exponent': 2, 'rmax_fraction': 1, 'oblique_rmax': 50},
        #     'hue_order': None, 'legend_title': m_sym['fwhm_fraction'], 'title': '',
        #     'plots': ['OEI', 'thresholds'],
        #     'filter': {'oblique_fwhm': [50]},
        # },
        # 'NEXT_N180': {
        #     'folder': 'Paper_N', 'x': 'contrast', 'y': 'dv', 'hue': 'fwhm_fraction', 'iter': 'decoder',
        #     'fixed': {'propn_cardinal': 0.5, 'total_n': 180,
        #               'cardinal_noise': 1, 'noise_fraction': 1, 'spont': '[0.03, 0.03]',
        #               'crf_c50': 24, 'crf_exponent': 2, 'rmax_fraction': 1, 'oblique_rmax': 50},
        #     'hue_order': None, 'legend_title': m_sym['fwhm_fraction'], 'title': '',
        #     'plots': ['OEI', 'thresholds'],
        #     'filter': {'oblique_fwhm': [50]},
        # },
        'NEXT_N': {
            'folder': 'NEXT_N', 'x': 'contrast', 'y': 'dv', 'hue': 'total_n', 'iter': 'decoder',
            'fixed': {'propn_cardinal': 0.5, 'fwhm_fraction': 0.5,
                      'cardinal_noise': 1, 'noise_fraction': 1, 'spont': '[0.03, 0.03]',
                      'crf_c50': 24, 'crf_exponent': 2, 'rmax_fraction': 1, 'oblique_rmax': 50},
            'hue_order': None, 'legend_title': 'N', 'title': '',
            'plots': ['OEI', 'thresholds'],
            'filter': {'oblique_fwhm': [50],
                       },
        },
        'NEXT_N_noisyOblique': {
            'folder': 'NEXT_N', 'x': 'contrast', 'y': 'dv', 'hue': 'total_n', 'iter': 'decoder',
            'fixed': {'propn_cardinal': 0.5, 'fwhm_fraction': 0.5,
                      'cardinal_noise': 1, 'noise_fraction': 2, 'spont': '[0.03, 0.03]',
                      'crf_c50': 24, 'crf_exponent': 2, 'rmax_fraction': 1, 'oblique_rmax': 50},
            'hue_order': None, 'legend_title': 'N', 'title': '',
            'plots': ['OEI', 'thresholds'],
            'filter': {'oblique_fwhm': [50],
                       },
        },
    }
    for _f in info:
        info[_f]['decoder_list'] = info[_f].get('decoder_list', ['WTA', 'PV', 'ML'])
    for _f in info:
        if not info[_f].get('x_label'):
            info[_f]['x_label'] = xlabel_con
        if info[_f].get('legend_title') is None:
            info[_f]['legend_title'] = info[_f]['hue']
        if not info[_f].get('legend'):
            info[_f]['legend'] = kwargs.get('legend', False)
        info[_f]['plot_label'] = f"{_f}"
    return info


def dvi_info(**kwargs):
    misc_sym = misc_symbols()
    m_sym = model_symbols()
    upper_greek = uppercase_greek_alphabet()
    lower_greek = lowercase_greek_alphabet()
    dvi_legends = True
    info = {
        '_bwC_scaledNoise': {
            'folder': 'XbwC_HnzC', 'x': 'tmp_fwhm_fraction', 'y': 'dv', 'hue': 'tmp_cardinal_noise', 'iter': 'decoder',
            'fixed': {  # FIXED #
                'task': 'ori_id', 'tmp_total_n': 180, 'tmp_oblique_fwhm': 50, 'tmp_noise_fraction': 2
            },
            'hue_order': None, 'x_label': m_sym['fwhm_fraction'], 'legend_title': m_sym['c_noise'],
            'plots': ['OEI', 'thresholds'],
            'filter': {'contrast': [40],
                       },
            'baseline': 'tmp_baseline', 'comparison_task': 'ori_dis',
        },
        '_bwC_scaledNoise_n20': {
            'folder': 'XbwC_HnzC', 'x': 'tmp_fwhm_fraction', 'y': 'dv', 'hue': 'tmp_cardinal_noise', 'iter': 'decoder',
            'fixed': {  # FIXED #
                'task': 'ori_id', 'tmp_total_n': 20, 'tmp_oblique_fwhm': 50, 'tmp_noise_fraction': 2
            },
            'hue_order': None, 'x_label': m_sym['fwhm_fraction'], 'legend_title': m_sym['c_noise'],
            'plots': ['OEI', 'thresholds'],
            'filter': {'contrast': [40],
                       },
            'baseline': 'tmp_baseline', 'comparison_task': 'ori_dis',
        },
        '_modelFWHM': {
            'folder': 'WTA_ML', 'x': 'oblique_fwhm', 'y': 'dv', 'hue': 'tmp_total_n', 'iter': 'decoder',
            'fixed': {  # FIXED #
                'task': 'ori_id', 'tmp_fwhm_fraction': 0.5, 'tmp_oblique_fwhm': 70, 'tmp_cardinal_noise': 1,
                'tmp_noise_fraction': 2
            },
            'hue_order': None, 'x_label': m_sym['o_fwhm'], 'legend_title': m_sym['N'], 'plots': ['OEI', 'thresholds'],
            'filter': {'contrast': [40],
                       },
            'baseline': 'tmp_baseline', 'comparison_task': 'ori_dis',
        },
        '_modelFWHM_hugeTemplateAnisotropy': {
            'folder': 'WTA_ML', 'x': 'tmp_total_n', 'y': 'dv', 'hue': 'oblique_fwhm', 'iter': 'decoder',
            'fixed': {  # FIXED #
                'task': 'ori_id', 'tmp_fwhm_fraction': 0.1, 'tmp_oblique_fwhm': 50, 'tmp_cardinal_noise': 1,
                'tmp_noise_fraction': 2
            },
            'hue_order': None, 'x_label': m_sym['N'], 'legend_title': m_sym['o_fwhm'], 'plots': ['OEI', 'thresholds'],
            'filter': {'contrast': [40],
                       },
            'baseline': 'tmp_baseline', 'comparison_task': 'ori_dis',
        },
        'unifNoise': {
            'folder': 'WTA_ML', 'x': 'tmp_total_n', 'y': 'dv', 'hue': 'tmp_cardinal_noise', 'iter': 'decoder',
            'fixed': {  # FIXED #
                'task': 'ori_id', 'tmp_fwhm_fraction': 0.5, 'tmp_oblique_fwhm': 50, 'tmp_noise_fraction': 1,
            },
            'hue_order': None, 'x_label': m_sym['N'], 'legend_title': m_sym['c_noise'], 'plots': ['OEI', 'thresholds'],
            'filter': {'contrast': [40],
                       },
            'baseline': 'tmp_baseline', 'comparison_task': 'ori_dis',
        },
        'N_bwC_m2550': {
            'folder': 'XN_HbwC', 'x': 'tmp_total_n', 'y': 'dv', 'hue': 'tmp_fwhm_fraction', 'iter': 'decoder',
            'fixed': {  # FIXED #
                'fwhm_fraction': 0.5,
                'task': 'ori_id', 'tmp_oblique_fwhm': 50, 'tmp_cardinal_noise': 1, 'tmp_noise_fraction': 2
            },
            'hue_order': [1, 0.8, 0.6, 0.4, 0.2], 'x_label': m_sym['N'], 'legend_title': m_sym['fwhm_fraction'], 'plots': ['OEI', 'thresholds'],
            'filter': {'contrast': [40], 'tmp_fwhm_fraction': [1, 0.8, 0.6, 0.4, 0.2],
                       },
            'baseline': 'tmp_baseline', 'comparison_task': 'ori_dis',  # baseline values for comparison
        },
        '_varyN_bwC_m5050': {
            'folder': 'XN_HbwC', 'x': 'tmp_total_n', 'y': 'dv', 'hue': 'tmp_fwhm_fraction', 'iter': 'decoder',
            'fixed': {  # FIXED #
                'fwhm_fraction': 1,
                'task': 'ori_id', 'tmp_oblique_fwhm': 50, 'tmp_cardinal_noise': 1, 'tmp_noise_fraction': 2
            },
            'hue_order': [1, 0.8, 0.6, 0.4, 0.2], 'x_label': m_sym['N'], 'legend_title': m_sym['fwhm_fraction'], 'plots': ['OEI', 'thresholds'],
            'filter': {'contrast': [40], 'tmp_fwhm_fraction': [1, 0.8, 0.6, 0.4, 0.2],
                       },
            'baseline': 'tmp_baseline', 'comparison_task': 'ori_dis',  # baseline values for comparison
        },
        '_varyN_bwC_m1350': {
            'folder': 'XN_HbwC', 'x': 'tmp_total_n', 'y': 'dv', 'hue': 'tmp_fwhm_fraction', 'iter': 'decoder',
            'fixed': {  # FIXED #
                'fwhm_fraction': 0.25,
                'task': 'ori_id', 'tmp_oblique_fwhm': 50, 'tmp_cardinal_noise': 1, 'tmp_noise_fraction': 2
            },
            'hue_order': [1, 0.8, 0.6, 0.4, 0.2], 'x_label': m_sym['N'], 'legend_title': m_sym['fwhm_fraction'], 'plots': ['OEI', 'thresholds'],
            'filter': {'contrast': [40], 'tmp_fwhm_fraction': [1, 0.8, 0.6, 0.4, 0.2],
                       },
            'baseline': 'tmp_baseline', 'comparison_task': 'ori_dis',  # baseline values for comparison
        },
        '_bothFWHM': {
            'folder': 'bothFWHWM', 'x': 'fwhm_fraction', 'y': 'dv', 'hue': 'tmp_fwhm_fraction', 'iter': 'decoder',
            'fixed': {  # FIXED #
                'task': 'ori_id', 'tmp_total_n': 180, 'tmp_oblique_fwhm': 50, 'tmp_noise_fraction': 2,
                'tmp_cardinal_noise': 1
            },
            'hue_order': None, 'x_label': m_sym['fwhm_fraction'], 'legend_title': m_sym['fwhm_fraction'], 'plots': ['OEI', 'thresholds'],
            'filter': {'contrast': [40], 'tmp_fwhm_fraction': [1, 0.8, 0.6, 0.4, 0.2],
                       },
            'baseline': 'tmp_baseline', 'comparison_task': 'ori_dis',
        },
        'EncodingPrecision': {
            'folder': 'EncodingPrecision', 'x': 'oblique_fwhm', 'y': 'dv', 'hue': 'tmp_oblique_fwhm', 'iter': 'decoder',
            'fixed': {'total_n': 180, 'propn_cardinal': 0.5, 'fwhm_fraction': 0.5,
                      'cardinal_noise': 1, 'noise_fraction': 2, 'spont': '[0.03, 0.03]',
                      'crf_c50': 24, 'crf_exponent': 2, 'oblique_rmax': 50, 'rmax_fraction': 1,
                      # Template
                      'tmp_con': 100, 'task': 'ori_id', 'tmp_total_n': 180, 'tmp_propn_cardinal': 0.5,
                      'tmp_fwhm_fraction': 0.5,
                      'tmp_noise_fraction': 2, 'tmp_cardinal_noise': 1},
            'hue_order': None, 'x_label': m_sym['o_fwhm'], 'legend_title': f"tmp {m_sym['o_fwhm']}",
            'plots': ['OEI', 'thresholds'],
            'filter': {'contrast': [40], 'oblique_fwhm': [20, 35, 50, 65, 80], 'tmp_oblique_fwhm': [20, 35, 50, 65, 80],
                       },
            'baseline': 'tmp_baseline', 'comparison_task': 'ori_dis',
        },
        'EncodingPrecision_n20': {
            'folder': 'EncodingPrecision', 'x': 'oblique_fwhm', 'y': 'dv', 'hue': 'tmp_oblique_fwhm', 'iter': 'decoder',
            'fixed': {'total_n': 180, 'propn_cardinal': 0.5, 'fwhm_fraction': 0.5,
                      'cardinal_noise': 1, 'noise_fraction': 2, 'spont': '[0.03, 0.03]',
                      'crf_c50': 24, 'crf_exponent': 2, 'oblique_rmax': 50, 'rmax_fraction': 1,
                      # Template
                      'tmp_con': 100, 'task': 'ori_id', 'tmp_total_n': 20, 'tmp_propn_cardinal': 0.5,
                      'tmp_fwhm_fraction': 0.5,
                      'tmp_noise_fraction': 2, 'tmp_cardinal_noise': 1},
            'hue_order': None, 'x_label': m_sym['o_fwhm'], 'legend_title': f"tmp {m_sym['o_fwhm']}",
            'plots': ['OEI', 'thresholds'],
            'filter': {'contrast': [40], 'oblique_fwhm': [20, 35, 50, 65, 80], 'tmp_oblique_fwhm': [20, 35, 50, 65, 80],
                       },
            'baseline': 'tmp_baseline', 'comparison_task': 'ori_dis',
        },
        'nzScale_varyN_nzF2': {
            'folder': 'nzScale_varyN', 'x': 'tmp_total_n', 'y': 'dv', 'hue': 'tmp_cardinal_noise', 'iter': 'decoder',
            'fixed': {'total_n': 180, 'propn_cardinal': 0.5, 'oblique_fwhm': 50, 'fwhm_fraction': 0.5,
                      'cardinal_noise': 1, 'noise_fraction': 2, 'spont': '[0.03, 0.03]',
                      'crf_c50': 24, 'crf_exponent': 2, 'oblique_rmax': 50, 'rmax_fraction': 1,
                      # Template
                      'tmp_con': 100, 'task': 'ori_id', 'tmp_propn_cardinal': 0.5,
                      'tmp_oblique_fwhm': 50, 'tmp_fwhm_fraction': 0.5,
                      'tmp_noise_fraction': 2},
            'hue_order': None, 'x_label': m_sym['N'], 'legend_title': m_sym['noise_scale'],
            'plots': ['OEI', 'thresholds'],
            'filter': {'contrast': [40], 'tmp_cardinal_noise': [1, 2, 4, 8, 16],
                       'tmp_total_n': [4, 12, 20, 60, 180],
                       },
            'baseline': 'tmp_baseline', 'comparison_task': 'ori_dis',  # baseline values for comparison
        },
        'nzScale_varyN_nzF1': {
            'folder': 'nzScale_varyN', 'x': 'tmp_total_n', 'y': 'dv', 'hue': 'tmp_cardinal_noise', 'iter': 'decoder',
            'fixed': {'total_n': 180, 'propn_cardinal': 0.5, 'oblique_fwhm': 50, 'fwhm_fraction': 0.5,
                      'cardinal_noise': 1, 'noise_fraction': 2, 'spont': '[0.03, 0.03]',
                      'crf_c50': 24, 'crf_exponent': 2, 'oblique_rmax': 50, 'rmax_fraction': 1,
                      # Template
                      'tmp_con': 100, 'task': 'ori_id', 'tmp_propn_cardinal': 0.5,
                      'tmp_oblique_fwhm': 50, 'tmp_fwhm_fraction': 0.5,
                      'tmp_noise_fraction': 1},
            'hue_order': None, 'x_label': m_sym['N'], 'legend_title': m_sym['noise_scale'],
            'plots': ['OEI', 'thresholds'],
            'filter': {'contrast': [40], 'tmp_cardinal_noise': [1, 2, 4, 8, 16],
                       'tmp_total_n': [4, 12, 20, 60, 180],
                       },
            'baseline': 'tmp_baseline', 'comparison_task': 'ori_dis',  # baseline values for comparison
        },
        'nzFrac_2050': {
            'folder': 'nzFrac', 'x': 'tmp_total_n', 'y': 'dv', 'hue': 'tmp_noise_fraction', 'iter': 'decoder',
            'fixed': {'total_n': 180, 'propn_cardinal': 0.5, 'oblique_fwhm': 50, 'fwhm_fraction': 0.5,
                      'cardinal_noise': 1, 'noise_fraction': 2, 'spont': '[0.03, 0.03]',
                      'crf_c50': 24, 'crf_exponent': 2, 'oblique_rmax': 50, 'rmax_fraction': 1,
                      # Template
                      'tmp_con': 100, 'task': 'ori_id', 'tmp_propn_cardinal': 0.5,
                      'tmp_oblique_fwhm': 50, 'tmp_fwhm_fraction': 0.4,
                      'tmp_cardinal_noise': 1},
            'hue_order': None, 'x_label': m_sym['N'], 'legend_title': m_sym['noise_scale'],
            'plots': ['OEI', 'thresholds'],
            'filter': {'contrast': [40], 'tmp_cardinal_noise': [1, 2, 4, 8, 16],
                       'tmp_total_n': [4, 12, 20, 60, 180],
                       },
            'baseline': 'tmp_baseline', 'comparison_task': 'ori_dis',  # baseline values for comparison
        },
        'nzFrac_1050': {
            'folder': 'nzFrac', 'x': 'tmp_total_n', 'y': 'dv', 'hue': 'tmp_noise_fraction', 'iter': 'decoder',
            'fixed': {'total_n': 180, 'propn_cardinal': 0.5, 'oblique_fwhm': 50, 'fwhm_fraction': 0.5,
                      'cardinal_noise': 1, 'noise_fraction': 2, 'spont': '[0.03, 0.03]',
                      'crf_c50': 24, 'crf_exponent': 2, 'oblique_rmax': 50, 'rmax_fraction': 1,
                      # Template
                      'tmp_con': 100, 'task': 'ori_id', 'tmp_propn_cardinal': 0.5,
                      'tmp_oblique_fwhm': 50, 'tmp_fwhm_fraction': 0.2,
                      'tmp_cardinal_noise': 1},
            'hue_order': None, 'x_label': m_sym['N'], 'legend_title': m_sym['noise_scale'],
            'plots': ['OEI', 'thresholds'],
            'filter': {'contrast': [40], 'tmp_cardinal_noise': [1, 2, 4, 8, 16],
                       'tmp_total_n': [4, 12, 20, 60, 180],
                       },
            'baseline': 'tmp_baseline', 'comparison_task': 'ori_dis',  # baseline values for comparison
        },
        '_Noise_N': {
            'folder': 'noise', 'x': 'tmp_total_n', 'y': 'dv', 'hue': 'tmp_noise_fraction', 'iter': 'decoder',
            'fixed': {  # FIXED #
                'task': 'ori_id', 'tmp_oblique_fwhm': 50, 'tmp_fwhm_fraction': 0.4, 'tmp_cardinal_noise': 1
            },
            'hue_order': None, 'x_label': m_sym['N'], 'legend_title': m_sym['noise_fraction'],
            'plots': ['OEI', 'thresholds'],
            'filter': {'contrast': [40], 'tmp_noise_fraction': [1, 2, 4, 8, 16],
                       },
            'baseline': 'tmp_baseline', 'comparison_task': 'ori_dis',
        },
        '_tmp_FWHM': {
            'folder': '_tmp_FWHM', 'x': 'tmp_total_n', 'y': 'dv', 'hue': 'tmp_oblique_fwhm', 'iter': 'decoder',
            'fixed': {
                'task': 'ori_id', 'tmp_fwhm_fraction': 0.4, 'tmp_cardinal_noise': 1, 'tmp_noise_fraction': 2
            },
            'hue_order': None, 'x_label': m_sym['N'], 'legend_title': m_sym['o_fwhm'],
            'plots': ['OEI', 'thresholds'],
            'filter': {'contrast': [40], 'tmp_oblique_fwhm': [30, 40, 50, 60, 70]},
            'baseline': 'tmp_baseline', 'comparison_task': 'ori_dis',
        },
        'noiseScale': {
            'folder': 'N_noiseScale', 'x': 'tmp_total_n', 'y': 'dv', 'hue': 'tmp_cardinal_noise', 'iter': 'decoder',
            'fixed': {  # FIXED #
                'task': 'ori_id', 'tmp_oblique_fwhm': 50, 'tmp_fwhm_fraction': 0.4, 'tmp_noise_fraction': 2
            },
            'hue_order': None, 'x_label': m_sym['N'], 'legend_title': m_sym['c_noise'], 'plots': ['OEI', 'thresholds'],
            'filter': {'contrast': [40], 'tmp_cardinal_noise': [0.1, 0.25, 1, 4, 10],
                       },
            'baseline': 'tmp_baseline', 'comparison_task': 'ori_dis',  # baseline values for comparison
        },
        '_prior_rmax': {
            'folder': 'Prior_rmax', 'x': 'tmp_total_n', 'y': 'dv', 'hue': 'tmp_rmax_fraction', 'iter': 'decoder',
            'fixed': {  # FIXED #
                'task': 'ori_id', 'tmp_fwhm_fraction': 0.4, 'tmp_oblique_fwhm': 50, 'tmp'
                                                                                    '_cardinal_noise': 1,
                'tmp_noise_fraction': 2, 'tmp_oblique_rmax': 50,
            },
            'hue_order': None, 'x_label': m_sym['N'], 'legend_title': m_sym['rmax_fraction'],
            'plots': ['OEI', 'thresholds'],
            'filter': {'contrast': [40], 'tmp_rmax_fraction': [1, 1.2, 1.4, 1.6, 1.8]},
            'baseline': 'tmp_baseline', 'comparison_task': 'ori_dis',
        },
        '_prior_rmax_5050': {
            'folder': 'Prior_rmax', 'x': 'tmp_total_n', 'y': 'dv', 'hue': 'tmp_rmax_fraction', 'iter': 'decoder',
            'fixed': {  # FIXED #
                'task': 'ori_id', 'tmp_fwhm_fraction': 1, 'tmp_oblique_fwhm': 50, 'tmp_cardinal_noise': 1,
                'tmp_noise_fraction': 2, 'tmp_oblique_rmax': 50,
            },
            'hue_order': None, 'x_label': m_sym['N'], 'legend_title': m_sym['rmax_fraction'],
            'plots': ['OEI', 'thresholds'],
            'filter': {'contrast': [40], 'tmp_rmax_fraction': [1, 1.2, 1.4, 1.6, 1.8]},
            'baseline': 'tmp_baseline', 'comparison_task': 'ori_dis',
        },
        '_prior_rmax_1050': {
            'folder': 'Prior_rmax', 'x': 'tmp_total_n', 'y': 'dv', 'hue': 'tmp_cardinal_rmax', 'iter': 'decoder',
            'fixed': {  # FIXED #
                'task': 'ori_id', 'tmp_fwhm_fraction': 0.2, 'tmp_oblique_fwhm': 50, 'tmp_cardinal_noise': 1,
                'tmp_noise_fraction': 2, 'tmp_oblique_rmax': 50,
            },
            'hue_order': None, 'x_label': m_sym['N'], 'legend_title': m_sym['c_rmax'],
            'plots': ['OEI', 'thresholds'],
            'filter': {'contrast': [40], 'tmp_cardinal_rmax': [50, 60, 70, 80, 90]},
            'baseline': 'tmp_baseline', 'comparison_task': 'ori_dis',
        },
        'bwC_narrow': {
            'folder': 'Chapter7', 'x': 'tmp_total_n', 'y': 'dv', 'hue': 'tmp_cardinal_rmax', 'iter': 'decoder',
            'fixed': {  # FIXED #
                'task': 'ori_id', 'tmp_fwhm_fraction': 0.2, 'tmp_oblique_fwhm': 50, 'tmp_cardinal_noise': 1,
                'tmp_noise_fraction': 1, 'tmp_oblique_rmax': 50,
            },
            'hue_order': None, 'x_label': m_sym['N'], 'legend_title': m_sym['c_rmax'],
            'plots': ['OEI', 'thresholds'],
            'filter': {'contrast': [40], 'tmp_cardinal_rmax': [50, 80, 110]},
            'baseline': 'tmp_baseline', 'comparison_task': 'ori_dis',
        },
        'bwC_broad': {
            'folder': 'Chapter7', 'x': 'tmp_total_n', 'y': 'dv', 'hue': 'tmp_cardinal_rmax', 'iter': 'decoder',
            'fixed': {  # FIXED #
                'task': 'ori_id', 'tmp_fwhm_fraction': 1, 'tmp_oblique_fwhm': 50, 'tmp_cardinal_noise': 1,
                'tmp_noise_fraction': 1, 'tmp_oblique_rmax': 50,
            },
            'hue_order': None, 'x_label': m_sym['N'], 'legend_title': m_sym['c_rmax'],
            'plots': ['OEI', 'thresholds'],
            'filter': {'contrast': [40], 'tmp_cardinal_rmax': [50, 80, 110]},
            'baseline': 'tmp_baseline', 'comparison_task': 'ori_dis',
        },
        'rmax': {
            'folder': 'Chapter7', 'x': 'tmp_total_n', 'y': 'dv', 'hue': 'tmp_cardinal_rmax', 'iter': 'decoder',
            'fixed': {  # FIXED #
                'task': 'ori_id', 'tmp_fwhm_fraction': 1, 'tmp_oblique_fwhm': 50, 'tmp_cardinal_noise': 1,
                'tmp_noise_fraction': 1, 'tmp_oblique_rmax': 50,
            },
            'hue_order': None, 'x_label': m_sym['N'], 'legend_title': m_sym['c_rmax'],
            'plots': ['OEI', 'thresholds'],
            'filter': {'contrast': [40]},
            'baseline': 'tmp_baseline', 'comparison_task': 'ori_dis',
        },
        'rmax_lowcon_lowrmax': {
            'folder': 'Chapter7', 'x': 'tmp_total_n', 'y': 'dv', 'hue': 'tmp_cardinal_rmax', 'iter': 'decoder',
            'fixed': {  # FIXED #
                'task': 'ori_id', 'tmp_fwhm_fraction': 0.2, 'tmp_oblique_fwhm': 50, 'tmp_cardinal_noise': 1,
                'tmp_noise_fraction': 1, 'tmp_oblique_rmax': 10, 'tmp_con': 10,
            },
            'hue_order': None, 'x_label': m_sym['N'], 'legend_title': m_sym['c_rmax'],
            'plots': ['OEI', 'thresholds'],
            'filter': {'contrast': [40]},
        },
        'rmax_highcon_lowrmax': {
            'folder': 'Chapter7', 'x': 'tmp_total_n', 'y': 'dv', 'hue': 'tmp_cardinal_rmax', 'iter': 'decoder',
            'fixed': {  # FIXED #
                'task': 'ori_id', 'tmp_fwhm_fraction': 0.2, 'tmp_oblique_fwhm': 50, 'tmp_cardinal_noise': 1,
                'tmp_noise_fraction': 1, 'tmp_oblique_rmax': 10, 'tmp_con': 100,
            },
            'hue_order': None, 'x_label': m_sym['N'], 'legend_title': m_sym['c_rmax'],
            'plots': ['OEI', 'thresholds'],
            'filter': {'contrast': [40]},
        },
        'rmax_vary_n180': {
            'folder': 'Chapter7 - rmax', 'x': 'tmp_oblique_rmax', 'y': 'dv', 'hue': 'tmp_rmax_fraction', 'iter': 'decoder',
            'fixed': {  # FIXED #
                'task': 'ori_id', 'tmp_total_n': 180, 'tmp_fwhm_fraction': 0.2, 'tmp_oblique_fwhm': 50,
                'tmp_noise_fraction': 1, 'tmp_con': 100,
            },
            'hue_order': None, 'x_label': m_sym['o_rmax'], 'legend_title': f"{m_sym['c_rmax']} / {m_sym['o_rmax']}",
            'plots': ['OEI', 'thresholds'],
            'filter': {'contrast': [40], 'tmp_rmax_fraction': [1, 1.5, 2, 3, 4]},
        },
        'rmax_vary_n20': {
            'folder': 'Chapter7 - rmax', 'x': 'tmp_oblique_rmax', 'y': 'dv', 'hue': 'tmp_rmax_fraction', 'iter': 'decoder',
            'fixed': {  # FIXED #
                'task': 'ori_id', 'tmp_total_n': 20, 'tmp_fwhm_fraction': 0.2, 'tmp_oblique_fwhm': 50,
                'tmp_noise_fraction': 1, 'tmp_con': 100,
            },
            'hue_order': None, 'x_label': m_sym['o_rmax'], 'legend_title': f"{m_sym['c_rmax']} / {m_sym['o_rmax']}",
            'plots': ['OEI', 'thresholds'],
            'filter': {'contrast': [40], 'tmp_rmax_fraction': [1, 1.5, 2, 3, 4]},
        },
        'rmax_vary_n4': {
            'folder': 'Chapter7 - rmax', 'x': 'tmp_oblique_rmax', 'y': 'dv', 'hue': 'tmp_rmax_fraction', 'iter': 'decoder',
            'fixed': {  # FIXED #
                'task': 'ori_id', 'tmp_total_n': 4, 'tmp_fwhm_fraction': 0.2, 'tmp_oblique_fwhm': 50,
                'tmp_noise_fraction': 1, 'tmp_con': 100,
            },
            'hue_order': None, 'x_label': m_sym['o_rmax'], 'legend_title': f"{m_sym['c_rmax']} / {m_sym['o_rmax']}",
            'plots': ['OEI', 'thresholds'],
            'filter': {'contrast': [40], 'tmp_rmax_fraction': [1, 1.5, 2, 3, 4]},
        },
        'rmax_unif': {
            'folder': 'Chapter6 - rmax_unif', 'x': 'tmp_total_n', 'y': 'dv', 'hue': 'tmp_oblique_rmax',
            'iter': 'decoder',
            'fixed': {  # FIXED #
                'task': 'ori_id', 'tmp_oblique_fwhm': 50, 'tmp_fwhm_fraction': 1, 'tmp_noise_fraction': 1,
                'tmp_rmax_fraction': 1,
                'tmp_con': 100,
            },
            'hue_order': None, 'x_label': 'N', 'legend_title': f"{m_sym['rmax']}",
            'plots': ['OEI', 'thresholds'],
            'filter': {'contrast': [40]},
        },
        '7.1. FWHM rmaxfrac1': {
            'folder': 'Chapter7 - Plots', 'x': 'tmp_total_n', 'y': 'dv', 'hue': 'tmp_fwhm_fraction',
            'iter': 'decoder',
            'fixed': {
                'task': 'ori_id', 'tmp_oblique_fwhm': 50,
                'tmp_noise_fraction': 1, 'tmp_oblique_rmax': 50, 'tmp_rmax_fraction': 1,
            },
            'hue_order': [1, 0.8, 0.6, 0.4, 0.2], 'x_label': 'N', 'legend_title': f"{m_sym['c_fwhm']} / {m_sym['o_fwhm']}",
            'plots': ['OEI', 'thresholds'],
            'filter': {'contrast': [40], 'tmp_fwhm_fraction': [1, 0.8, 0.6, 0.4, 0.2],
                       'tmp_total_n': [4, 12, 20, 60, 180]},
        },
        '7.1. FWHM rmaxfrac2': {
            'folder': 'Chapter7 - Plots', 'x': 'tmp_total_n', 'y': 'dv', 'hue': 'tmp_fwhm_fraction',
            'iter': 'decoder',
            'fixed': {
                'task': 'ori_id', 'tmp_oblique_fwhm': 50,
                'tmp_noise_fraction': 1, 'tmp_oblique_rmax': 50, 'tmp_rmax_fraction': 2,
            },
            'hue_order': [1, 0.8, 0.6, 0.4, 0.2], 'x_label': 'N', 'legend_title': f"{m_sym['c_fwhm']} / {m_sym['o_fwhm']}",
            'plots': ['OEI', 'thresholds'],
            'filter': {'contrast': [40], 'tmp_fwhm_fraction': [1, 0.8, 0.6, 0.4, 0.2],
                       'tmp_total_n': [4, 12, 20, 60, 180]},
        },
        '7.1. FWHM rmaxfrac4': {
            'folder': 'Chapter7 - Plots', 'x': 'tmp_total_n', 'y': 'dv', 'hue': 'tmp_fwhm_fraction',
            'iter': 'decoder',
            'fixed': {
                'task': 'ori_id', 'tmp_oblique_fwhm': 50,
                'tmp_noise_fraction': 1, 'tmp_oblique_rmax': 50, 'tmp_rmax_fraction': 4,
            },
            'hue_order': [1, 0.8, 0.6, 0.4, 0.2], 'x_label': 'N', 'legend_title': f"{m_sym['c_fwhm']} / {m_sym['o_fwhm']}",
            'plots': ['OEI', 'thresholds'],
            'filter': {'contrast': [40], 'tmp_fwhm_fraction': [1, 0.8, 0.6, 0.4, 0.2],
                       'tmp_total_n': [4, 12, 20, 60, 180]},
        },
        '7.2. RMAXo rmaxfrac1': {
            'folder': 'Chapter7 - Plots', 'x': 'tmp_total_n', 'y': 'dv', 'hue': 'tmp_oblique_rmax', 'iter': 'decoder',
            'fixed': {
                'task': 'ori_id', 'tmp_oblique_fwhm': 50, 'tmp_fwhm_fraction': 0.2,
                'tmp_noise_fraction': 1, 'tmp_rmax_fraction': 1,
            },
            'hue_order': [50, 25, 10, 5, 2.5], 'x_label': 'N', 'legend_title': f"{m_sym['o_rmax']}",
            'plots': ['OEI', 'thresholds'],
            'filter': {'contrast': [40], 'tmp_oblique_rmax': [50, 25, 10, 5, 2.5],
                       'tmp_total_n': [4, 12, 20, 60, 180]},
        },
        '7.2. RMAXo rmaxfrac2': {
            'folder': 'Chapter7 - Plots', 'x': 'tmp_total_n', 'y': 'dv', 'hue': 'tmp_oblique_rmax', 'iter': 'decoder',
            'fixed': {
                'task': 'ori_id', 'tmp_oblique_fwhm': 50, 'tmp_fwhm_fraction': 0.2,
                'tmp_noise_fraction': 1, 'tmp_rmax_fraction': 2,
            },
            'hue_order': [50, 25, 10, 5, 2.5], 'x_label': 'N', 'legend_title': f"{m_sym['o_rmax']}",
            'plots': ['OEI', 'thresholds'],
            'filter': {'contrast': [40], 'tmp_oblique_rmax': [50, 25, 10, 5, 2.5],
                       'tmp_total_n': [4, 12, 20, 60, 180]},
        },
        '7.2. RMAXo rmaxfrac4': {
            'folder': 'Chapter7 - Plots', 'x': 'tmp_total_n', 'y': 'dv', 'hue': 'tmp_oblique_rmax', 'iter': 'decoder',
            'fixed': {
                'task': 'ori_id', 'tmp_oblique_fwhm': 50, 'tmp_fwhm_fraction': 0.2,
                'tmp_noise_fraction': 1, 'tmp_rmax_fraction': 4,
            },
            'hue_order': [50, 25, 10, 5, 2.5], 'x_label': 'N', 'legend_title': f"{m_sym['o_rmax']}",
            'plots': ['OEI', 'thresholds'],
            'filter': {'contrast': [40], 'tmp_oblique_rmax': [50, 25, 10, 5, 2.5],
                       'tmp_total_n': [4, 12, 20, 60, 180]},
        },
        '7.3. EncodingFWHM rmaxfrac1': {
            'folder': 'Chapter7 - Plots', 'x': 'tmp_total_n', 'y': 'dv', 'hue': 'fwhm_fraction', 'iter': 'decoder',
            'fixed': {
                'task': 'ori_id', 'tmp_oblique_fwhm': 50, 'tmp_fwhm_fraction': 0.2,
                'tmp_noise_fraction': 1, 'tmp_rmax_fraction': 1, 'tmp_oblique_rmax': 5,
            },
            'hue_order': [1, 0.8, 0.6, 0.4, 0.2], 'x_label': 'N', 'legend_title': f"{m_sym['fwhm_fraction']}",
            'plots': ['OEI', 'thresholds'],
            'filter': {'contrast': [40], 't'
                                         'mp_oblique_rmax': [50, 25, 10, 5, 2.5],
                       'tmp_total_n': [4, 12, 20, 60, 180], 'fwhm_fraction': [1, 0.8, 0.6, 0.4, 0.2]},        },
        '7.3. EncodingFWHM rmaxfrac2': {
            'folder': 'Chapter7 - Plots', 'x': 'tmp_total_n', 'y': 'dv', 'hue': 'fwhm_fraction', 'iter': 'decoder',
            'fixed': {
                'task': 'ori_id', 'tmp_oblique_fwhm': 50, 'tmp_fwhm_fraction': 0.2,
                'tmp_noise_fraction': 1, 'tmp_rmax_fraction': 2, 'tmp_oblique_rmax': 5,
            },
            'hue_order': [1, 0.8, 0.6, 0.4, 0.2], 'x_label': 'N', 'legend_title': f"{m_sym['fwhm_fraction']}",
            'plots': ['OEI', 'thresholds'],
            'filter': {'contrast': [40], 'tmp_oblique_rmax': [50, 25, 10, 5, 2.5],
                       'tmp_total_n': [4, 12, 20, 60, 180], 'fwhm_fraction': [1, 0.8, 0.6, 0.4, 0.2]},
        },
        '7.3. EncodingFWHM rmaxfrac4': {
            'folder': 'Chapter7 - Plots', 'x': 'tmp_total_n', 'y': 'dv', 'hue': 'fwhm_fraction', 'iter': 'decoder',
            'fixed': {
                'task': 'ori_id', 'tmp_oblique_fwhm': 50, 'tmp_fwhm_fraction': 0.2,
                'tmp_noise_fraction': 1, 'tmp_rmax_fraction': 4, 'tmp_oblique_rmax': 5,
            },
            'hue_order': [1, 0.8, 0.6, 0.4, 0.2],
            'x_label': 'N', 'legend_title': f"{m_sym['fwhm_fraction']}",
            'plots': ['OEI', 'thresholds'],
            'filter': {'contrast': [40], 'tmp_oblique_rmax': [50, 25, 10, 5, 2.5],
                       'tmp_total_n': [4, 12, 20, 60, 180], 'fwhm_fraction': [1, 0.8, 0.6, 0.4, 0.2]},
        },
        'test1': {
            'folder': 'test', 'x': 'tmp_total_n', 'y': 'dv', 'hue': 'tmp_oblique_rmax', 'iter': 'decoder',
            'fixed': {
                'task': 'ori_id', 'tmp_oblique_fwhm': 50, 'tmp_fwhm_fraction': 0.4,
                'tmp_noise_fraction': 1, 'tmp_rmax_fraction': 1,
            },
            'hue_order': [50, 25, 10, 5, 2.5], 'x_label': 'N', 'legend_title': f"{m_sym['o_rmax']}",
            'plots': ['OEI', 'thresholds'],
            'filter': {'contrast': [40], 'tmp_oblique_rmax': [50, 25, 10, 5, 2.5],
                       'tmp_total_n': [4, 12, 20, 60, 180]},
        },
        'test2': {
            'folder': 'test', 'x': 'tmp_total_n', 'y': 'dv', 'hue': 'tmp_oblique_rmax', 'iter': 'decoder',
            'fixed': {
                'task': 'ori_id', 'tmp_oblique_fwhm': 50, 'tmp_fwhm_fraction': 1,
                'tmp_noise_fraction': 1, 'tmp_rmax_fraction': 1,
            },
            'hue_order': [50, 25, 10, 5, 2.5], 'x_label': 'N', 'legend_title': f"{m_sym['o_rmax']}",
            'plots': ['OEI', 'thresholds'],
            'filter': {'contrast': [40], 'tmp_oblique_rmax': [50, 25, 10, 5, 2.5],
                       'tmp_total_n': [4, 12, 20, 60, 180]},
        },
    }
    # Default decoders
    for _f in info:
        info[_f]['decoder_list'] = info[_f].get('decoder_list', ['WTA', 'ML'])
    # Add some default info if missing
    default = {'propn_cardinal': 0.5, 'spont': '[0.03, 0.03]', 'crf_c50': 24, 'crf_exponent': 2, 'oblique_rmax': 50,
               'rmax_fraction': 1, 'cardinal_noise': 1, 'noise_fraction': 2, 'total_n': 180,
               'oblique_fwhm': 50, 'fwhm_fraction': 0.5,
               'tmp_oblique_rmax': 50, 'tmp_con': 100, 'tmp_propn_cardinal': 0.5,
               'tmp_total_n': 180, 'tmp_cardinal_noise': 1, 'tmp_rmax_fraction': 1,
               }
    for _key in default:
        for _f in info:
            if any(info[_f][_param] == _key for _param in ['x', 'hue', 'iter', 'y']):  # check not accounted for
                continue
            info[_f]['fixed'][_key] = info[_f]['fixed'].get(_key, default[_key])  # use default if unaccounted for

    # Replicate each condition for each template decoder
    adjusted_info = {}
    for _f in info:
        for _d in ['ML', 'WTA']:
            # adjusted_info[f"tmp{_d}_{_f}"] = deepcopy(info[_f])
            adjusted_info[f"{_f}___tmp{_d}"] = deepcopy(info[_f])
            adjusted_info[f"{_f}___tmp{_d}"]['fixed']['tmp_decoder'] = _d
    info = adjusted_info.copy()
    # Add plotting info
    for _f in info:
        if not info[_f].get('x_label'):
            info[_f]['x_label'] = info[_f]['x']
        if info[_f].get('legend_title') is None:
            info[_f]['legend_title'] = info[_f]['hue']
        if not info[_f].get('legend'):
            info[_f]['legend'] = dvi_legends
        info[_f]['plot_label'] = f"{_f}"
    return info


def plot_multicond_thresholds(input_df, baseline_params, default_params, conds, decoder_list,
                              **kwargs):
    df = input_df.copy()
    df = assign_conds(df, conds)

    # Get keys specified for custom conditions
    cond_keys = list(set([_i for _keys in conds.values() for _i in _keys]))  # Unique conds
    # Filter oei data for each condition (can be identified by cond_num)
    for i_baseline in baseline_params:  # Separate plots for each baseline condition
        base_keys = list(i_baseline.keys())
        base_vals = list(i_baseline.values())
        base_df = index_dframe(df, base_keys, base_vals)  # Index baseline vals
        # Set default params
        for i_cond in default_params:
            if default_params.get(i_cond) is None:  # Skip if value set to None
                continue
            if any(_i == i_cond for _i in [*cond_keys, *base_keys]):  # Skip if used in conditions
                continue
            else:  # If baseline cond is the same as i_cond from default params
                base_df = index_dframe(base_df, i_cond, default_params[i_cond],
                                       'or')
        plot_df = base_df.copy()
        # Saving info
        base_keys_str = "".join(f"{_i}_" for _i in base_keys)
        base_vals_str = "".join(f"{_i}_" for _i in base_vals)
        to_csv_pkl(plot_df, kwargs['save_dir'], f"{base_keys_str} = {base_vals_str}", _pkl=False)
        # Settings for figure window
        markers = init_markers()
        n_decoders = len(decoder_list)
        ncols = 3
        nrows = int(np.ceil(n_decoders / ncols))
        ax_skip = None
        if (nrows * ncols) % n_decoders != 0:  # Determine which ax idx to skip
            ax_skip = ncols - 1
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
        figwidth = 4.5
        figheight = 4.2
        fig.set_figheight(figheight * nrows)  # scale height for nrows
        fig.set_figwidth(figwidth * ncols)  # assumes 2 cols
        wspace = 0.16

        # Loop through axes of figure and generate plots
        axes = axes.flatten()  # flatten for easier iteration
        for idx, ax in enumerate(axes):
            # Skip ax if necessary
            if ax_skip is None:
                i_decoder = decoder_list[idx]
            else:
                if idx < ax_skip:
                    i_decoder = decoder_list[idx]
                elif idx == ax_skip:
                    ax.remove()
                    continue
                else:
                    i_decoder = decoder_list[idx - 1]
            # Plot data for this decoder
            i_df = index_dframe(plot_df, ['decoder'], [i_decoder])
            hue = 'ori_std'
            palette = kwargs.get('palette', sns.color_palette('colorblind'))
            plot_line(i_df, x='cond', y='dv', hue=hue,
                      palette=palette, x_type='cat', hue_order=None,
                      x_label="Spatial Frequency", y_label=kwargs.get('y_label', ""),
                      title=i_decoder.upper(), title_y=1.05,
                      yscale='linear', y_lim=kwargs.get('y_lim'), legend=False,
                      markers=markers[:len(set(i_df[hue]))], marker_size=10, dodge=0.2, line_width=0.8,
                      tight_layout=True, my_dpi=kwargs.get('dpi', 300),
                      ax=ax, fig=fig)
            xlim_categories = [-0.32, 2.32]

            # Plot comparison / behavioural OEIs
            if kwargs.get('beh_thresh'):
                ax.plot(arr(ax.get_xticks()) - 0.1, kwargs['beh_thresh'][0],
                        linestyle="--", color=palette[0], linewidth=0.01, markersize=13,
                        marker="*", markerfacecolor='none', alpha=0.6)
                ax.plot(arr(ax.get_xticks()) + 0.1, kwargs['beh_thresh'][1],
                        linestyle="--", color=palette[1], linewidth=0.01, markersize=13,
                        marker="*", markerfacecolor='none', alpha=0.6)
                ax.set_xlim(xlim_categories)

            # Remove labels of axes objects not in first row or col
            if not ax.get_subplotspec().is_first_col():
                ax.set_ylabel(' ')
                ax.set_yticklabels([])
            if not ax.get_subplotspec().is_last_row():  # shared xaxis
                ax.set_xlabel(' ')
                ax.set_xticklabels([])
            # Grid
            ax.yaxis.grid(which='both', linestyle='-', alpha=0.4)
        # Legend
        fig.set_clip_on(False)
        [_ax.set_clip_on(False) for _ax in axes]  # Allows legend to exceed boundaries of each plot

        # Single shared axes labels
        for i_ax in fig.axes:
            i_ax.set(xlabel=' ', ylabel=' ')  # Remove axes labels
        fig.axes[0].set_ylabel(kwargs.get('y_label', "Threshold"), labelpad=6,
                               # fontdict={'fontweight': 'bold', 'fontsize': 24},
                               )
        fig.axes[1].set_xlabel('Spatial Frequency', labelpad=13,
                               # fontdict={'fontweight': 'bold', 'fontsize': 24},
                               )

        if ax_skip is None:
            # If legend fits above the figure
            axes[ncols - 1].legend(loc='center', labels=[f"Cardinal", f"Oblique"]).set(
                title="Orientation", bbox_to_anchor=[0.7, 1.6])
        else:
            # If the legend fits at the end of the first row
            axes[ncols - 2].legend(loc='center', labels=[f"Cardinal", f"Oblique"]).set(
                title="Orientation", bbox_to_anchor=[1.35, 0.9])

        # fig.suptitle(f"{m_sym[base_cond]} = {base_val}", fontsize=24, x=0.52, y=.99)

        # Spacing (must be after legend and tight_layout)
        fig.subplots_adjust(wspace=wspace, hspace=0.25,
                            )

        # SAVING #
        if kwargs.get('save_dir'):
            fig.savefig(os.path.join(kwargs['save_dir'], f"{base_keys_str} = {base_vals_str}.png"),
                        dpi=kwargs.get('dpi', 100),
                        bbox_inches="tight"
                        )
            # Log transform y-axis (resets tick labels)
            if kwargs.get('plot_log') is True:
                for idx, ax in enumerate(axes):
                    ax.set(yscale='log', ylim=kwargs.get('y_lim_log'))
                    if idx % ncols != 0:  # shared yaxis
                        if ax.get_figure():  # if ax not removed
                            ax.set_yticklabels([])
                fig.savefig(os.path.join(kwargs['save_dir'], f"{base_keys_str} = {base_vals_str}__log.png"),
                            dpi=kwargs.get('dpi', 100),
                            bbox_inches="tight"
                            )
        plt.close()
    return None


def plot_multicond(input_df, isolated_params, baseline_params, default_params, conds, decoder_list,
                   **kwargs):
    df = input_df.copy()
    df = assign_conds(df, conds)
    # Labels
    m_sym = model_symbols()
    # Get keys specified for custom conditions
    cond_keys = list(set([_i for _keys in conds.values() for _i in _keys]))  # Unique conds
    # Separate plots for each baseline condition
    for i_baseline in baseline_params:
        # Create dataframe containing all relevant conditions with default / variable / baseline params applied
        # Overwrite order: conds -> iso -> baseline -> default
        df_list = []
        for iso_cond in isolated_params:
            for iso_val in isolated_params[iso_cond]:
                # Set variable parameter
                i_df = index_dframe(df, [iso_cond], [iso_val])
                # Set baseline parameter
                for base_param in i_baseline:
                    if base_param != iso_cond:
                        i_df = index_dframe(i_df, [base_param], [i_baseline[base_param]])
                # Set default parameters
                for i_cond in default_params:
                    if default_params.get(i_cond) is None:  # Skip if value set to None
                        continue
                    # Skip if used in custom conditions
                    if i_cond in [*cond_keys, *list(i_baseline), iso_cond]:
                        continue
                    # Set default parameter
                    i_df = index_dframe(i_df, i_cond, default_params[i_cond],
                                        'or')  # Set default param
                # Check all conditions accounted for
                if len(i_df['cond'].unique()) == len(conds):
                    hue_label = f"{m_sym[iso_cond]} = {iso_val}"
                    i_df['hue'] = hue_label
                else:
                    print(f"Indexing did not isolate {len(conds)} conditions - "
                          f"{iso_cond} = {isolated_params[iso_cond]}")
                df_list.append(i_df)

        # Generate dataframe for plot
        plot_df = pd.concat(df_list)
        # Saving info
        base_keys_str = "".join(f"{_i}_" for _i in list(i_baseline))
        base_vals_str = "".join(f"{_i}_" for _i in [i_baseline[_k] for _k in i_baseline])
        to_csv_pkl(plot_df, kwargs['save_dir'], f"{base_keys_str} = {base_vals_str}", _pkl=False)

        # Settings for figure window
        markers = init_markers()
        n_decoders = len(decoder_list)
        ncols = 3
        nrows = int(np.ceil(n_decoders / ncols))
        ax_skip = None
        if (nrows * ncols) % n_decoders != 0:  # Determine which ax idx to skip
            ax_skip = ncols - 1
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
        figwidth = 4.5
        figheight = 4.8
        wspace = 0.16
        fig.set_figheight(figheight * nrows)  # scale height for nrows
        fig.set_figwidth(figwidth * ncols)  # assumes 2 cols

        # Loop through figure axes and plot
        axes = axes.flatten()  # flatten for easier iteration
        for idx, ax in enumerate(axes):
            # Skip ax if necessary
            if ax_skip is None:
                i_decoder = decoder_list[idx]
            else:
                if idx < ax_skip:
                    i_decoder = decoder_list[idx]
                elif idx == ax_skip:
                    ax.remove()
                    continue
                else:
                    i_decoder = decoder_list[idx - 1]
            # Plot data for this decoder
            i_df = index_dframe(plot_df, ['decoder'], [i_decoder])
            hue = 'hue'
            plot_line(i_df, x='cond', y='dv', hue=hue,
                      palette='colorblind', x_type='cat', hue_order=None,
                      x_label="Spatial Frequency", y_label="OEI",
                      title=i_decoder.upper(), title_y=1.05,
                      yscale='linear', y_lim=kwargs.get('y_lim'), legend=False,
                      markers=markers[:len(set(i_df[hue]))], marker_size=13, dodge=0.3, line_width=0.8,
                      tight_layout=True, my_dpi=kwargs.get('dpi', 100),
                      ax=ax, fig=fig)
            # Plot comparison / behavioural OEIs
            if kwargs.get('beh_oei'):
                ax.plot(kwargs['beh_oei'], linestyle="--", color="k", linewidth=0.001, markersize=15,
                        marker="*", markerfacecolor='none', alpha=0.6)

            # Remove labels of axes objects not in first row or col
            if not ax.get_subplotspec().is_first_col():
                ax.set_ylabel(' ')
                ax.set_yticklabels([])
            if not ax.get_subplotspec().is_last_row():  # shared xaxis
                ax.set_xlabel(' ')
                ax.set_xticklabels([])

            # Grid
            ax.yaxis.grid(which='both', linestyle='-', alpha=0.4)
            # Horizontal line at OEI = 1
            ax.axhline(y=1, color='k', linestyle='--', alpha=0.4, linewidth=2)

        # Legend
        fig.set_clip_on(False)
        [_ax.set_clip_on(False) for _ax in axes]  # Allows legend to exceed boundaries of each plot

        # Single shared axes labels
        for i_ax in fig.axes:
            i_ax.set(xlabel=' ', ylabel=' ')  # Remove axes labels
        fig.axes[0].set_ylabel('OEI', labelpad=8,
                               # fontdict={'fontweight': 'bold', 'fontsize': 26}
                               )
        fig.axes[1].set_xlabel('Spatial Frequency', labelpad=13,
                               # fontdict={'fontweight': 'bold', 'fontsize': 24}
                               )

        if ax_skip is None:
            # If legend fits above the figure
            axes[ncols - 1].legend(loc='center').set(title="Parameters", bbox_to_anchor=[0.65, 1.6])
        else:
            # If the legend fits at the end of the first row
            axes[ncols - 2].legend(loc='center').set(title="Parameters", bbox_to_anchor=[1.35, 0.9])

        # fig.suptitle(f"{m_sym[base_cond]} = {base_val}", fontsize=24, x=0.52, y=.99)

        # Spacing (after legend and tight_layout)
        fig.subplots_adjust(wspace=wspace, hspace=figheight * 0.25)

        # Add figure label (or add left margin by setting color="white" and exploit bbox_inches="tight" when saving)
        if i_baseline == {'propn_cardinal': 0.5}:
            fig_label = "A."
        else:
            fig_label = "B."
        fig.suptitle(fig_label, fontsize=50, fontweight="bold", x=-0.02, y=0.88, color="black",
                     fontname="Cambria",
                     )

        # SAVING #
        if kwargs.get('save_dir'):
            fig.savefig(os.path.join(kwargs['save_dir'], f"{base_keys_str} = {base_vals_str}.png"),
                        dpi=kwargs.get('dpi', 100),
                        bbox_inches="tight"
                        )
            # Log transform y-axis (resets tick labels)
            if kwargs.get('plot_log') is True:
                for idx, ax in enumerate(axes):
                    ax.set(yscale='log', ylim=kwargs.get('y_lim_log'))
                    if idx % ncols != 0:  # shared yaxis
                        if ax.get_figure():  # if ax not removed
                            ax.set_yticklabels([])
                fig.savefig(os.path.join(kwargs['save_dir'], f"{base_keys_str} = {base_vals_str}__log.png"),
                            dpi=kwargs.get('dpi', 100),
                            bbox_inches="tight"
                            )
        plt.close()

        # Return
        if kwargs.get('return_fig') is True:
            return fig
