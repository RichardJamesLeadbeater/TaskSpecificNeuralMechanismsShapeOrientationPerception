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
    return fig, ax


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


def _setup_figure(ax, fig, fig_dims):
    """Helper to setup figure and axes consistently."""
    if ax is None:
        _fig, _ax = plt.subplots(figsize=fig_dims)
    elif fig is None:
        _ax = ax
        _fig = _ax.figure
    else:
        _fig, _ax = fig, ax
    return _fig, _ax


def _apply_labels(ax, title, x_label, y_label, title_y=1.13):
    """Helper to apply consistent labels."""
    if title is not None:
        ax.set_title(title, y=title_y, fontweight='bold')
    ax.set_ylabel(y_label, labelpad=3, fontweight='bold')
    ax.set_xlabel(x_label, labelpad=9, fontweight='bold')


def _apply_ticks(ax, y_major_ticks, y_minor_ticks, x_major_ticks, x_minor_ticks):
    """Helper to apply consistent tick formatting."""
    if y_major_ticks is not None:
        ax.set_yticks(y_major_ticks)
        ax.set_yticklabels([f"{int(i)}" for i in y_major_ticks])
    if y_minor_ticks is not None:
        ax.set_yticks(y_minor_ticks, minor=True)
    if x_major_ticks is not None:
        ax.set_xticks(x_major_ticks)
        ax.set_xticklabels([f"{int(i)}" for i in x_major_ticks])
    if x_minor_ticks is not None:
        ax.set_xticks(x_minor_ticks, minor=True)


def _apply_legend(ax, legend, legend_title=''):
    """Helper to apply consistent legend formatting."""
    if legend is not False:
        ax.legend(
            title=legend_title,
            loc='center left',
            bbox_to_anchor=(1.02, 0.85),
            borderaxespad=0,
        )
    else:
        if ax.get_legend():
            ax.get_legend().remove()


def _save_figure(fig, save_dir, save_name, my_dpi):
    """Helper to save figure consistently."""
    if all(i for i in [save_dir, save_name]):
        filepath = os.path.join(save_dir, save_name)
        fig.savefig(filepath, dpi=my_dpi)


def plot_bar(dframe, x, y, hue, hue_order=None, palette='colorblind', title='', y_label='', x_label='',
             y_lim=None, y_major_ticks=None, y_minor_ticks=None, _log=False, log_y_lim=None,
             custom_err=False, custom_err_label='t_sem', save_name='', save_dir=None,
             fig_dims=(7, 4), my_dpi=100, close_plot=False, ax=None, fig=None, **kwargs):
    """Create bar plot with error bars."""
    _fig, _ax = _setup_figure(ax, fig, fig_dims)

    if kwargs.get('show_errorBars', True) is True:
        sns.barplot(data=dframe, x=x, y=y, hue=hue, hue_order=hue_order, palette=palette,
                    capsize=0.05, err_kws={'linewidth': 2, 'color': [0.1, 0.1, 0.1, 0.8]},
                    errorbar=('ci', 68), n_boot=2000, ax=_ax)
    else:
        sns.barplot(data=dframe, x=x, y=y, hue=hue, hue_order=hue_order, palette=palette,
                    capsize=0, err_kws={'linewidth': 0, 'color': [1, 1, 1]},
                    errorbar=None, ax=_ax)

    _ax.yaxis.grid(which='both', linestyle='-', alpha=0.4)
    _ax.set(ylim=y_lim)

    if _log:
        _ax.set_yscale('log')
        _ax.set_ylim(log_y_lim)

    # Apply hatches if requested
    if kwargs.get('hatches'):
        n_bars = len(dframe[x].unique())
        hatches = [""] * n_bars + ["/"] * n_bars
        for idx, _bar in enumerate(_ax.patches):
            _bar.set_hatch(hatches[idx])
            _bar.set_edgecolor('w')

    _apply_labels(_ax, title, x_label, y_label, title_y=kwargs.get('title_y', 1.1))
    _apply_ticks(_ax, y_major_ticks, y_minor_ticks, None, None)
    _apply_legend(_ax, kwargs.get('legend', True), kwargs.get('legend_title', ''))

    sns.despine(fig=_fig, ax=_ax, top=True, right=True)
    _fig.set_dpi(my_dpi)
    _fig.tight_layout()

    _save_figure(_fig, save_dir, save_name, my_dpi)

    if close_plot:
        plt.close(_fig)

    return _fig, _ax


def plot_distribution(
    dframe, x, y, hue, hue_order, palette,
    title='', y_label='', x_label='',
    y_lim=None, y_major_ticks=None, y_minor_ticks=None,
    _log=False, log_y_lim=None,
    save_name='', save_dir=None,
    fig_dims=(7, 4), my_dpi=100, width=0.8,
    close_plot=False, ax=None, fig=None,
    show_points=True, **kwargs
):
    """Create distribution plot (boxplot with optional individual data points)."""

    _fig, _ax = _setup_figure(ax, fig, fig_dims)

    # Boxplot (summary of distribution)
    sns.boxplot(
        data=dframe,
        x=x,
        y=y,
        hue=hue,
        hue_order=hue_order,
        palette=palette,
        width=width,
        showcaps=True,
        boxprops={'alpha': .9},
        fliersize=0,
        whis=kwargs.get('whis', 1.5),
        ax=_ax
    )

    # Optional individual data points
    if show_points:
        if kwargs.get('strip_color'):
            palette = [kwargs['strip_color']] * len(hue_order)
        sns.stripplot(
            data=dframe.drop_duplicates(subset=[x, y, hue]),
            x=x,
            y=y,
            hue=hue,
            hue_order=hue_order,
            palette=palette,
            dodge=True,
            alpha=kwargs.get('strip_alpha', 0.3),
            size=5,
            legend=False,
            ax=_ax
        )

        # # Remove duplicate legend from stripplot
        # _ax.legend_.remove()

    _ax.yaxis.grid(which='both', linestyle='-', alpha=0.4)
    _ax.set(ylim=y_lim)

    if _log:
        _ax.set_yscale('log')
        _ax.set_ylim(log_y_lim)

    _apply_labels(_ax, title, x_label, y_label, title_y=1.13)
    _apply_ticks(_ax, y_major_ticks, y_minor_ticks, None, None)
    _apply_legend(_ax, kwargs.get('legend', True), kwargs.get('legend_title', ''))

    sns.despine(fig=_fig, ax=_ax, top=True, right=True)
    _fig.set_dpi(my_dpi)
    _fig.tight_layout()

    _save_figure(_fig, save_dir, save_name, my_dpi)

    if close_plot:
        plt.close(_fig)

    return _fig, _ax
