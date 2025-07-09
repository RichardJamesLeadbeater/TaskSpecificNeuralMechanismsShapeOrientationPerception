import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
import os
import pandas as pd
import seaborn as sns


def init_markers():
    markers = ['o', 's', 'D', '^', 'v', '>', '<', 'p', '*', '+', 'x', 'h', 'H', '.', ',', '1', '2', '3', '4']
    return markers


def barplot_err(data, x, y, xerr=None, yerr=None, **kwargs):
    # barplot with custom SE (produced by +- the t_sem from single value)
    _data = []
    for _i in data.index:
        _data_i = pd.concat([data.loc[_i:_i]] * 3, ignore_index=True, sort=False)
        _row = data.loc[_i]
        if xerr is not None:
            _data_i[x] = [_row[x] - _row[xerr], _row[x], _row[x] + _row[xerr]]
        if yerr is not None:
            _data_i[y] = [_row[y] - _row[yerr], _row[y], _row[y] + _row[yerr]]
        _data.append(_data_i)

    _data = pd.concat(_data, ignore_index=True, sort=False)

    # _ax = sns.catplot(x=x, y=y, data=_data, ci='sd', kind='bar', **kwargs)
    _ax = sns.barplot(x=x, y=y, data=_data, ci='sd', **kwargs)
    return _ax


def load_seaborn_prefs(style="ticks", context="talk", custom_rc=None):
    if custom_rc is None:
        custom_rc = {}
    axes_color = [0.2, 0.2, 0.2, 0.95]
    rc_default = {
        'axes.edgecolor': axes_color, 'xtick.color': axes_color, 'ytick.color': axes_color,
        'axes.linewidth': 1, 'patch.linewidth': 1.2,
        'xtick.major.width': 1, 'xtick.minor.width': 1, 'ytick.major.width': 1, 'ytick.minor.width': 1,
        'errorbar.capsize': 0.04, 'hatch.color': 'k', 'lines.linewidth': 2, 'lines.markersize': 10,
        'axes.titlesize': 25, 'xtick.labelsize': 15, 'ytick.labelsize': 17,
        'axes.labelsize': 20, 'font.family': 'sans-serif',
        'legend.title_fontsize': 13, 'legend.fontsize': 13, 'legend.handletextpad': 0.25,
    }
    # update dict from kwargs entries
    for k in custom_rc:
        # Notify user if default value not provided for this parameter
        if not rc_default.__contains__(k):
            print(f"KeyError: {k} is not a key in rc_default\n"
                  f"...now setting in mpl.rcParams")
        # Set custom parameter
        rc_default[k] = custom_rc[k]
    # set rc_default params through seaborn
    sns.set_theme(style=style, context=context, rc=rc_default)
    # matplotlib params set separately from seaborn
    defaults = {'figure.figsize': (6, 4), 'figure.dpi': 400, 'hatch.linewidth': 3.4,
                }
    for i_param in ['figure.figsize', 'hatch.linewidth']:
        if custom_rc.__contains__(i_param):
            mpl.rcParams[i_param] = custom_rc[i_param]
        else:
            mpl.rcParams[i_param] = defaults[i_param]


def customise_barplot(barplot, title=None, ylabel=None, xlabel=None, y_major_ticks=None, y_minor_ticks=None, ylim=None,
                      hatch=True, len_x=1, legend=False, despine=True, tight=True, dpi=None):
    if title is not None:
        barplot.set_title(title)
    if ylabel is not None:
        barplot.set_ylabel(ylabel, fontweight='bold')
    if xlabel is not None:
        barplot.set_xlabel(xlabel, fontweight='bold')
    if y_major_ticks is not None:
        barplot.set_yticks(y_major_ticks)
        barplot.set_yticklabels([f"{int(i)}" for i in y_major_ticks])
    if y_minor_ticks is not None:
        barplot.set_yticks(y_minor_ticks, minor=True)
    if ylim is not None:
        barplot.set_ylim(ylim)
    if hatch is True:
        hatches = [""] * len_x + [""] * len_x + ["\\"] * len_x + ["/"] * len_x
        for i, i_bar in enumerate(barplot.patches):
            i_bar.set_hatch(hatches[i])
            i_bar.set_edgecolor('w')
    if legend is False:
        barplot.get_legend().remove()
    if despine is True:
        sns.despine(top=True, right=True)
    if dpi is not None:
        barplot.figure.set_dpi(dpi)
    if tight is True:
        barplot.figure.tight_layout()


def plot_bar(dframe, x, y, hue, hue_order, palette, title='', y_label='', x_label='', legend=True, legend_title=None,
             y_lim=None, y_major_ticks=None, y_minor_ticks=None, ci=68, yscale='linear',
             custom_err=False, custom_err_label='t_sem', save_name='', save_dir=None, fig_dims=(7, 4), dpi=100,
             close_plot=False, ax=None, fig=None,
             **kwargs):
    if ax is None:
        _fig, _ax = plt.subplots(figsize=fig_dims)
    elif fig is None:
        _fig, _ax = ax.figure()
    else:
        _fig, _ax = fig, ax
    sns.despine(_fig, _ax, top=True, right=True, offset=0)  # can affect text formatting so put here
    if custom_err is False:
        bar_plot = sns.barplot(data=dframe, x=x, y=y, hue=hue, hue_order=hue_order, palette=palette,
                               capsize=0.05,
                               err_kws={'color': [0.1, 0.1, 0.1, 0.8],
                                        'linewidth': 2},
                               errorbar=('ci', 68), n_boot=2000, ax=_ax)
    else:
        bar_plot = barplot_err(data=dframe, x=x, y=y, yerr=custom_err_label, hue=hue, hue_order=hue_order,
                               palette=palette,
                               capsize=.05, err_kws={'color': [0.1, 0.1, 0.1, 0.8],
                                                     'linewidth': 2},
                               ax=_ax)
    # PLOT LABELS #
    # bar_plot.set_title(title, y=1.13, fontweight='bold')
    # bar_plot.set_ylabel(y_label, size=20, fontweight='bold')
    # bar_plot.set_xlabel(x_label, size=24, labelpad=9)
    if title is not None:
        bar_plot.set_title(title.upper(), y=kwargs.get('title_y'), fontweight='bold')
    bar_plot.set_yscale(yscale)
    # HATCHES #
    if kwargs.get('hatches') is True:
        n_clusters = len(dframe[x].unique())  # n bar clusters
        hatches = ['', '', '\\', '/']
        plot_hatches = [i_hatch for i_hatch in hatches for _ in range(n_clusters)]  # repeat hatches in each bar cluster
        plot_hatches = plot_hatches + hatches  # legend hatches?
        for hatch_idx, i_hatch in enumerate(plot_hatches):
            _bar = bar_plot.patches[hatch_idx]
            _bar.set_hatch(i_hatch)
            _bar.set_edgecolor('w')
    if kwargs.get('hatch_cvso') is True:
        n_clusters = len(dframe[x].unique())  # n bar clusters
        hatches = ['++', 'xx']
        plot_hatches = [i_hatch for i_hatch in hatches for _ in range(n_clusters)]  # repeat hatches in each bar cluster
        for idx, _bar in enumerate(bar_plot.patches):
            _bar.set_hatch(plot_hatches[idx])
            _bar.set_edgecolor('w')
    # LEGEND #
    _ax.legend().set(bbox_to_anchor=[1.02, 0.75], title=legend_title)
    if legend is False:
        _ax.get_legend().remove()
    _ax.set(ylim=y_lim, yscale=yscale)  # reset after legend
    # PLOT LABELS #
    bar_plot.set_ylabel(y_label, labelpad=kwargs.get('ylabel_pad', 4.5), fontweight='bold')
    bar_plot.set_xlabel(x_label, labelpad=kwargs.get('xlabel_pad', 10), fontweight='bold')
    # TICK LABELS #
    # must go after legend
    bar_plot.set_ylim(y_lim)
    if y_major_ticks is not None:
        bar_plot.set_yticks(y_major_ticks)
        if isinstance(y_major_ticks[0], int):
            y_major_ticks_labels = [f"{int(i)}" for i in y_major_ticks]
        else:
            y_major_ticks_labels = [f"{float(i)}" for i in y_major_ticks]
        bar_plot.set_yticklabels(y_major_ticks_labels)
    if y_minor_ticks is not None:
        bar_plot.set_yticks(y_minor_ticks, minor=True)
    # FIGURE LAYOUT #
    _ax.yaxis.grid(which='both', linestyle='-', alpha=0.4)
    _fig.set_dpi(dpi)
    _fig.tight_layout()
    # SAVE FIGURE #
    if all(i for i in [save_dir, save_name]):
        bar_plot.figure.savefig(os.path.join(os.path.join(save_dir, save_name)), dpi=dpi)
    # close plot before continuing
    if close_plot is True:
        plt.close(_fig)
    return _fig, _ax


def plot_line(dframe, x, y, hue, palette='colorblind', hue_order=None, title='', y_label='', x_label='', legend=True,
              x_lim=None, y_lim=None, y_major_ticks=None, y_minor_ticks=None, ci=None, yscale='linear',
              ax=None, legend_title=None, x_major_ticks=None, x_minor_ticks=None,
              save_name='', save_dir=None, fig_dims=(6, 4), dpi=400, marker=None, x_type='cat', scatter=False,
              close_plot=False, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_dims)
    else:
        fig = ax.figure
    sns.despine(fig, ax, top=True, right=True, offset=0)  # can affect text formatting so put here
    # Set errorbar parameter
    if ci is not None:
        errorbar = ('ci', ci)
    else:
        errorbar = None
    if scatter is True:
        line_plot = sns.scatterplot(data=dframe, x=x, y=y, hue=hue, hue_order=hue_order, errorbar=('ci', ci),
                                    marker=marker,
                                    style=hue, alpha=kwargs.get('alpha'), palette=palette,
                                    err_kws={'linewidth': kwargs.get('errwidth')}, capsize=kwargs.get('capsize'),
                                    ax=ax)
    else:
        if x_type == 'num':
            line_plot = sns.lineplot(data=dframe, x=x, y=y, hue=hue, hue_order=hue_order, errorbar=('ci', ci),
                                     alpha=kwargs.get('alpha'), palette=palette, markers=kwargs.get('markers'),
                                     err_kws={'linewidth': kwargs.get('errwidth')}, capsize=kwargs.get('capsize', 0),
                                     ax=ax)
            ax.set(xlim=(0, None))
        elif x_type == 'cat':
            if kwargs.get('markers') is None:
                kwargs['markers'] = "o"
            line_plot = sns.pointplot(data=dframe, x=x, y=y, hue=hue, hue_order=hue_order, errorbar=errorbar,
                                      palette=palette, markers=kwargs.get('markers'), dodge=kwargs.get('dodge'),
                                      err_kws={'linewidth': kwargs.get('errwidth')}, capsize=kwargs.get('capsize', 0),
                                      ax=ax)
    ax.legend().set(bbox_to_anchor=kwargs.get('bbox', [1.05, 0.75]), title=legend_title)
    if legend is False:
        ax.get_legend().remove()
    ax.set(ylim=y_lim, yscale=yscale)  # y-axis
    ax.set(xlim=x_lim)  # y-axis
    # CHANGE MARKER SIZES AND LINEWIDTHS #
    for i_line in ax.lines:
        i_line.set_alpha(kwargs.get('line_alpha'))
        # if kwargs.get('line_width'):
        #     i_line.set_linewidth(kwargs['line_width'])
        i_line.set_markeredgecolor([1, 1, 1])
        i_line.set_markeredgewidth(0.5)
        if any(_i == x_type for _i in ['num']):
            if kwargs.get('marker_size') is not None:
                i_line.set_markersize(kwargs['marker_size'])
    if not any(_xt == x_type for _xt in ['num']):
        for i_marker in ax.collections:
            if kwargs.get('marker_size') or kwargs.get('marker_boundary'):
                i_marker.set_sizes(np.atleast_1d(kwargs.get('marker_size', 175)))
                i_marker.set_edgecolor([.9, .9, .9])
                # i_marker.set_linewidths(1)
            if kwargs.get('marker_alpha'):
                i_marker.set_alpha(kwargs.get('marker_alpha'))
    # PLOT LABELS #
    if title is not None:
        line_plot.set_title(title.upper(), y=kwargs.get('title_y'), fontweight='bold')
    line_plot.set_ylabel(y_label, labelpad=kwargs.get('ylabel_pad', 4.5), fontweight='bold')
    line_plot.set_xlabel(x_label, labelpad=kwargs.get('xlabel_pad', 10), fontweight='bold')
    # TICK LABELS #
    if y_major_ticks is not None:
        line_plot.set_yticks(y_major_ticks)
        line_plot.set_yticklabels([f"{int(i)}" for i in y_major_ticks])
    if y_minor_ticks is not None:
        line_plot.set_yticks(y_minor_ticks, minor=True)
    # X TICK LABELS #
    if x_major_ticks is not None:
        line_plot.set_xticks(x_major_ticks)
        line_plot.set_xticklabels([f"{int(i)}" for i in x_major_ticks])
    if x_minor_ticks is not None:
        line_plot.set_xticks(x_minor_ticks, minor=True)
    # FIGURE LAYOUT #
    ax.yaxis.grid(which='both', linestyle='-', alpha=0.4)
    fig.tight_layout()

    # SAVE FIGURE #
    if all(i for i in [save_dir, save_name]):
        filepath = os.path.join(os.path.join(save_dir, save_name))
        fig.savefig(filepath, dpi=dpi)
    # close plot before continuing
    if close_plot is True:
        plt.close(fig)
    if kwargs.get('show') is True:
        plt.show()
    return fig, ax


def plot_iqr(dframe, x, y, hue, hue_order=None, palette='colorblind', title='', x_label='', y_label='', legend=True,
             y_lim=None, y_major_ticks=None, y_minor_ticks=None,
             save_name='', save_dir=None, fig_dims=(7, 4), close_plot=True, dpi=100,
             include_strip=False, **kwargs):
    # interquartile range plots
    _fig, _ax = plt.subplots(figsize=fig_dims)
    sns.despine(_fig, _ax, top=True, right=True, offset=0)  # can affect text formatting so put here
    # plot strip plot then overlay boxplot
    if include_strip is True:
        strip_plot = sns.stripplot(data=dframe, x=x, y=y, hue=hue, hue_order=hue_order,
                                   palette=np.asarray(palette) / 2, linewidth=1.6,
                                   dodge=True, edgecolor=[1] * 3, alpha=0.5, size=6, jitter=.12,
                                   ax=_ax)  # alpha>0 to show
    # plot box plot with custom styles (median, boxprops)
    median_props = dict(linestyle='--', linewidth=2.2, color='k', alpha=1)
    mean_props = {"marker": "s", "markerfacecolor": "white", "markeredgecolor": "black",
                  "markersize": 6}
    box_props = dict(linestyle='-', linewidth=1.4, edgecolor=[0.95] * 3, alpha=1)
    box_plot = sns.boxplot(data=dframe, x=x, y=y, hue=hue, hue_order=hue_order, palette=palette,
                           linewidth=0.8, showfliers=False, showmeans=True, meanprops=mean_props,
                           medianprops=median_props, boxprops=box_props, ax=_ax)
    if legend is False:
        _ax.get_legend().remove()  # remove legend (for reports screenshot)
    # PLOT LABELS #
    if title is not None:
        box_plot.set_title(title.upper(), y=kwargs.get('title_y'), fontweight='bold')
    box_plot.set_ylabel(y_label, labelpad=3, fontweight='bold')
    box_plot.set_xlabel(x_label, labelpad=8, fontweight='bold')
    # TICK LABELS #
    _ax.set_ylim(y_lim)
    if y_major_ticks is not None:
        _ax.set_yticks(y_major_ticks)
        _ax.set_yticklabels([f"{int(i)}" for i in y_major_ticks])
    if y_minor_ticks is not None:
        _ax.set_yticks(y_minor_ticks, minor=True)
    # HATCHES #
    n_bars = len(dframe[x].unique())  # n bar clusters
    hatches = ['', '', '\\', '/'] * n_bars
    for hatch, patch in zip(hatches, _ax.artists):
        patch.set_hatch(hatch)
    # FIGURE LAYOUT #
    # _ax.set_xticklabels(_ax.get_xmajorticklabels(), fontweight='bold')  # sets xtick labels to bold
    _ax.yaxis.grid(which='both', linestyle='-', alpha=0.4)
    _fig.set_dpi(dpi)
    _fig.tight_layout()
    # SAVE FIGURE #
    if all(i for i in [save_dir, save_name]):
        _fig.savefig(os.path.join(os.path.join(save_dir, save_name)))
    # close plot before continuing
    if close_plot is True:
        plt.close(_fig)
    return _fig, _ax


def plot_bland_altman(dframe, x, y, hue=None, hue_order=None, mean_diff=None, limits=None, equality=None, legend=False,
                      title=None, figsize=None, xlims=None, ylims=None, xlabel=None, ylabel=None, yscale='linear',
                      palette=None, markersize=None, alpha=0.8, ratio_col=None, ratio_ylabel='', ax=None):
    if ax is None:
        if figsize is None:
            figsize = (5, 5)
        fig, ax = plt.subplots(figsize=figsize)

    scatplot = sns.scatterplot(data=dframe, x=x, y=y, hue=hue, hue_order=hue_order, palette=palette, alpha=alpha,
                               legend=legend, style=hue, ax=ax)
    if legend is True:
        ax.legend().set(bbox_to_anchor=[1.7, 0.9])
        ratio_ylabel = ''
    # scatplot.get_legend().remove()  # necessary step so 'full' will work
    if xlims is None:
        xlims = scatplot.get_xlim()
    if ylims is None:
        ylims = scatplot.get_ylim()
    if mean_diff is not None:
        scatplot.axes.plot(xlims, [mean_diff, mean_diff], c='r', ls='--', linewidth=1.5)
    if limits is not None:
        scatplot.axes.plot(xlims, [limits[0], limits[0]], c='b', ls=':', linewidth=2)
        scatplot.axes.plot(xlims, [limits[1], limits[1]], c='b', ls=':', linewidth=2)
    if equality is not None:
        scatplot.axes.plot(xlims, [equality, equality], c=[0, 0, 0, 0.2], ls='-', linewidth=2)

    if title is not None:
        scatplot.set_title(title, y=1.03)
    if xlims is not None:
        scatplot.set_xlim(xlims)
    if ylims is not None:
        scatplot.set_ylim(ylims)
    if xlabel is None:
        xlabel = scatplot.get_xlabel()
    if ylabel is None:
        ylabel = scatplot.get_ylabel()
    scatplot.set_yscale(yscale)
    scatplot.set_xlabel(xlabel)
    scatplot.set_ylabel(ylabel)
    scatplot.figure.tight_layout()

    # Add twin y-axis with nested function
    if ratio_col is not None:
        ax2 = ax.twinx()
        plot_bland_altman(dframe, x=x, y=ratio_col, hue=hue, xlims=xlims, ylims=np.exp(ylims), hue_order=hue_order,
                          ylabel=ratio_ylabel, palette=palette, alpha=alpha, yscale='log', ratio_col=None, ax=ax2)
    z = 0
    return ax


def multifig_formatter(fig, **kwargs):
    # Define axes labels
    axes = fig.axes
    x_label = kwargs.get('x_label', axes[0].get_xlabel())
    y_label = kwargs.get('y_label', axes[0].get_ylabel())

    # Edit current axes labels
    for i_ax in axes:
        # Remove all axes labels
        i_ax.set_xlabel(' ')
        i_ax.set_ylabel(' ')
        # Remove tick labels on all but the first col and last row
        if i_ax.get_subplotspec().is_first_col() is False:
            i_ax.set_yticklabels([])
        if i_ax.get_subplotspec().is_last_row() is False:
            i_ax.set_xticklabels([])
        # Set shared labels
        ax_loc = [i_ax.get_subplotspec().rowspan.start,
                  i_ax.get_subplotspec().colspan.start
                  ]
        if ax_loc[0] == 1 and ax_loc[1] == 0:
            i_ax.set_ylabel(y_label, labelpad=12,
                            fontdict={'fontsize': kwargs.get('fontsize'),
                                      'fontweight': 'bold'}
                            )
        if ax_loc[0] == 2 and ax_loc[1] == 1:
            i_ax.set_xlabel(x_label, labelpad=10,
                            fontdict={'fontsize': kwargs.get('fontsize'),
                                      'fontweight': 'bold'}
                            )
    fig.tight_layout()

    return fig


def coords_ax_to_fig(fig, ax, ax_coords):
    # Axes coords -> display pixels -> figure fraction coords
    pixel_coords = ax.transAxes.transform(ax_coords)
    fig_coords = fig.transFigure.inverted().transform(pixel_coords)

    return fig_coords


def coords_fig_to_ax(fig, ax, fig_coords):
    # Figure fraction coords -> display pixels -> axes coords
    pixel_coords = fig.transFigure.transform(fig_coords)
    ax_coords = ax.transAxes.inverted().transform(pixel_coords)

    return ax_coords
