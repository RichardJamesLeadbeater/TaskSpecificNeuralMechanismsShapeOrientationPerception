import itertools as itls
import numpy as np
import pandas as pd
import scipy.stats as st
from mytools.dframe_tools import init_cols, index_dframe, get_cond_combos
from mytools.dict_tools import append_dicty
from mytools.misc import add_cardinal_and_oblique


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


def get_each_ratio(dframe, conds, ratio_labels=['OEI', 'HVI', 'IOR'], ori_label='ori'):
    if not isinstance(conds, list):  # conds must be input as a list
        return None
    cols = ['ratio score', 'value']
    ratio_data = init_cols([c for c in conds] + cols)
    cond_combos = get_cond_combos(dframe, conds)  # all unique cond combinations
    for i_combo in cond_combos:
        i_dframe = index_dframe(dframe, conds, i_combo)
        if i_dframe.empty:
            continue
        for i_ratio_label in ratio_labels:
            ratio_val = get_ori_ratio(i_dframe, i_ratio_label, ori_label)
            append_dicty(ratio_data, list(i_combo) + [i_ratio_label, ratio_val]
                         )
    ratio_data = pd.DataFrame(ratio_data)
    return ratio_data


def mean_across_cond(dframe, condition=None, ori_label='ori', dv_label='threshold'):
    output = init_cols([ori_label, condition, dv_label, 'stdev', 'sem'])
    for i in dframe[ori_label].unique():
        i_ori = i
        if not condition:  # used when taking mean across all conds for each observer
            i_data = dframe[(dframe[ori_label] == i_ori)]
            append_dicty(output, [i_ori, i_data[dv_label].mean(), i_data[dv_label].std(),
                                  st.sem(i_data[dv_label])])
        else:
            for cond in dframe[condition].unique():  # used when taking mean across all observers for each cond
                i_data = dframe[(dframe[ori_label] == i_ori) & (dframe[condition] == cond)]
                append_dicty(output, [i_ori, cond, i_data[dv_label].mean(),
                                      i_data[dv_label].std(), st.sem(i_data[dv_label])])
    return pd.DataFrame.from_dict(output)


def get_overall_mean(ratio_data, dv_label='threshold'):
    all_conds_all_observers = {}
    for col in ['OEI']:
        all_conds_all_observers[col] = ratio_data[col].mean()
    return pd.DataFrame(all_conds_all_observers, index=[dv_label])


def get_ratio_of_means(dframe, condition, ratio_labels=None, ori_label='ori'):
    if ratio_labels is None:
        ratio_labels = ['OEI', 'HVI', 'IOR']
    # output = init_cols([condition, 'ratio score', 'mean_ratio'])
    output_table = init_cols([condition] + ratio_labels)
    for cond in dframe[condition].unique():  # keep separate to keep col lengths equal
        output_table[condition].append(cond)
    # calculate each oriratio for each condition
    for ratio in ratio_labels:
        for cond in dframe[condition].unique():
            i_data = dframe[(dframe[condition] == cond)]
            output_table[ratio].append(get_ori_ratio(i_data, ratio, ori_label=ori_label))
    output_table = pd.DataFrame(output_table)
    return output_table


def get_mean_of_ratios(dframe, condition, use_geomean=False):
    mean_ratio_data = init_cols([condition, 'ratio_score', 'value'])
    for i_cond in dframe[condition].unique():
        for i_ratio in dframe['ratio score'].unique():
            i_dframe = index_dframe(dframe, [condition, 'ratio score'], [i_cond, i_ratio])
            if use_geomean:
                mean_val = geomean(i_dframe['value'].to_numpy())
            else:
                mean_val = np.mean(i_dframe['value'].to_numpy())
            append_dicty(mean_ratio_data, [i_cond, i_ratio, mean_val])
    mean_ratio_data = pd.DataFrame(mean_ratio_data)
    return mean_ratio_data


def cols2rows(dframe, change_cols, new_col):
    # useful to convert summary_data data tables for plotting
    keep_cols = [i for i in dframe.columns if all(j != i for j in change_cols)]
    output = init_cols(keep_cols + [new_col] + ['values'])
    for i_change in change_cols:
        output[new_col] += [i_change] * len(dframe)
        output['values'] += dframe[i_change].to_list()
        for i_keep in keep_cols:
            output[i_keep] += dframe[i_keep].to_list()

    return pd.DataFrame(output)


def get_ylim_logscale(dframe, dframe_dv='mean'):
    if dframe[dframe_dv].min() > 0.1:
        minval = (dframe[dframe_dv].min() / 1.1)
        return minval, None

    # auto log scale for facetgrid can cut off smallest values, this puts ylim to suitable range
    logscale_minval = dframe[dframe_dv].min() * (10 ** dframe[dframe_dv].min())
    # leave default to auto_val (None)
    if 1 > logscale_minval >= 0.1:
        log_ymin = 0.1  # if in range of 0.1 to 1 then set min to 0.1
    elif 0.1 > logscale_minval >= 0.01:
        log_ymin = 0.01
    elif 0.01 > logscale_minval >= 0.001:
        log_ymin = 0.001
    else:
        log_ymin = None
    return log_ymin, None


def plot_each_ratio(summarytable, file_name, file_path):
    # PLOTS EACH OBSERVERS RATIO ACROSS CONDITIONS
    try:
        line_plot = sns.catplot(x=iv, y='values',
                                hue='participant', hue_order=summarytable['participant'].unique().tolist(),
                                kind='point', legend=False,
                                capsize=.04, sharey=True, palette='colorblind',
                                data=summarytable, n_boot=None, col='ratio score')
    except:
        z = 0
    all_ymin = 1000
    all_ymax = -1
    for ax in line_plot.axes:
        for subax in ax:
            i_ylim = subax.get_ylim()
            if i_ylim[0] < all_ymin:
                all_ymin = i_ylim[0]
            if i_ylim[1] > all_ymax:
                all_ymax = i_ylim[1]
            for i_collection in subax.collections:
                i_collection.set_sizes(i_collection.get_sizes() / 1.8)
    line_plot.set(ylim=(0.0, np.ceil(all_ymax)))
    # i_bar.fig.suptitle(f"Orientation ratios", size=18, y=.98)
    line_plot.set_titles(col_template="{col_name}", size=title_size, y=1, weight='bold')
    line_plot.set_xlabels(iv, size=label_size)
    line_plot.set_ylabels('ratio score', size=label_size)
    for j in line_plot.axes.flatten():
        j.tick_params(labelleft=True, labelbottom=True)
        j.set_yticklabels(j.get_yticklabels(), size=tick_size)
        j.set_xticklabels(j.get_xticklabels(), size=tick_size)
    line_plot.fig.subplots_adjust(wspace=width_space, hspace=height_space)
    line_plot.tight_layout()
    line_plot.fig.set(dpi=100)
    line_plot.savefig(os.path.join(file_path, f"{file_name}_each_ratio_log.png"))
    # i_bar.set(yscale='log', ylim=(all_ymin, np.ceil(all_ymax)))
    line_plot.set(yscale='log', ylim=ylimit_log)

    for j in line_plot.axes.flatten():  # removes y-axis minor ticks
        j.tick_params(labelleft=True, labelbottom=True, which='both')
        j.set_yticklabels(j.get_yticklabels(), size=tick_size, minor=True)
        j.set_xticklabels(j.get_xticklabels(), size=tick_size)
    line_plot.savefig(os.path.join(file_path, f"{file_name}_each_ratio_log.png"))


def plot_mean_ratios(dframe, x, y, hue, hue_order, palette, y_label='', x_label='', file_name='', file_path=''):
    bar_plot = sns.catplot(data=dframe, x=x, y=y, hue=hue, hue_order=hue_order, palette=palette,
                           kind='point', linestyles=[(0, (8, 0.75))] * 3, legend=False, n_boot=None)
    # change size of markers
    [i.set_sizes(i.get_sizes() / 1.8) for i in bar_plot.ax.collections]
    y_lim = bar_plot.ax.get_ylim()
    bar_plot.ax.set_ylim(0, np.ceil(y_lim[1]))
    # set labels
    bar_plot.set_xlabels(x_label, size=label_size)
    bar_plot.set_ylabels(y_label, size=label_size)

    for j in bar_plot.axes.flatten():
        j.tick_params(which='both', labelleft=True, labelbottom=True)
        j.set_yticklabels(j.get_yticklabels(), size=tick_size)
        j.set_xticklabels(j.get_xticklabels(), size=tick_size)

    bar_plot.add_legend(fontsize=legend_fontsize)
    bar_plot.tight_layout()
    bar_plot.fig.set(dpi=my_dpi)
    bar_plot.savefig(os.path.join(file_path, f"{file_name}.png"))

    bar_plot.set(yscale='log', ylim=ylimit_log)
    for j in bar_plot.axes.flatten():  # removes y-axis minor ticks
        j.tick_params(labelleft=True, labelbottom=True, which='both')
        j.set_yticklabels(j.get_yticklabels(), size=tick_size, minor=True)
        j.set_xticklabels(j.get_xticklabels(), size=tick_size)
    bar_plot.savefig(os.path.join(file_path, f"{file_path}{file_name}.png"))
