import numpy as np
from mytools.dframe_tools import index_dframe


def get_ori_ratio(dframe, ratio, dv='threshold', ori_label='orientation'):
    """calculate ratio across ? for each condition"""
    # if len(dframe) != 6:  # [H, V, -45, 45, cardinal, oblique]
    #     return None
    oris = None
    if ratio.upper() == 'OEI':  # oblique effect index
        cardinal_df = index_dframe(dframe, [ori_label], ['horizontal', 'vertical'], 'or')
        oblique_df = index_dframe(dframe, [ori_label], ['minus45', 'plus45'], 'or')
        ratio_val = oblique_df[dv].sum() / cardinal_df[dv].sum()
        return ratio_val
    elif ratio.upper() == 'HVI':  # horizontal-vertical index
        oris = ['vertical', 'horizontal']
    elif ratio.upper() == 'IOR':  # inter-oblique index
        oris = ['plus45', 'minus45']
    x_df = dframe[dframe[ori_label] == oris[0]]
    x = x_df[dv].to_list()[0]
    y_df = dframe[dframe[ori_label] == oris[1]]
    y = y_df[dv].to_list()[0]
    ratio_val = x / y
    return ratio_val


def geomean(values):
    # put ratios on log-scale to allow us to take the mean then convert back
    return np.exp(np.mean(np.log(values)))
