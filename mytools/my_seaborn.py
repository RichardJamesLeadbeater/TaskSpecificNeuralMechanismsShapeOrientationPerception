import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns


def init_markers():
    """Return list of matplotlib marker styles."""
    return ['o', 's', 'D', '^', 'v', '>', '<', 'p', '*', '+', 'x', 'h', 'H', '.', ',', '1', '2', '3', '4']


def remove_legend_title(ax):
    """Remove legend title while keeping legend items."""
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels)


def barplot_err(data, x, y, xerr=None, yerr=None, **kwargs):
    """Create barplot with custom standard errors."""
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
    _ax = sns.barplot(x=x, y=y, data=_data, ci='sd', **kwargs)
    return _ax


def load_seaborn_prefs(custom_rc=None, style="ticks", context="talk", palette='colorblind'):
    """Load seaborn preferences with custom matplotlib rcParams."""
    sns.set_palette(palette)
    sns.set_theme(style=style, context=context)

    if custom_rc is None:
        custom_rc = {}

    axes_color = [0.2, 0.2, 0.2, 0.95]
    mpl_defaults = {
        # Axes
        'axes.edgecolor': axes_color,
        'axes.linewidth': 1,
        'axes.titlesize': 22,
        'axes.titleweight': 'normal',
        'axes.labelsize': 18,
        'axes.labelweight': 'bold',

        # Ticks
        'xtick.color': axes_color,
        'ytick.color': axes_color,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'xtick.major.width': 1,
        'xtick.minor.width': 1,
        'ytick.major.width': 1,
        'ytick.minor.width': 1,

        # Lines & markers
        'lines.linewidth': 2,
        'lines.markersize': 7,

        # Patches & hatching
        'patch.linewidth': 1.2,
        'hatch.color': 'k',
        'hatch.linewidth': 3.4,

        # Error bars
        'errorbar.capsize': 0.04,

        # Legend
        'legend.fontsize': 13,
        'legend.title_fontsize': 20,

        # Figure
        'figure.figsize': (6, 4),

        # Fonts
        'font.family': 'sans-serif',
    }

    for k, v in mpl_defaults.items():
        if k not in custom_rc:
            custom_rc[k] = v

    try:
        mpl.rcParams.update(custom_rc)
    except KeyError as e:
        print(f"Warning: {e.args[0]} is not a valid matplotlib rcParam")


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


def _apply_labels(ax, title, x_label, y_label, title_y=1.13, title_fontweight='bold', **kwargs):
    """Helper to apply consistent labels."""
    if title is not None:
        ax.set_title(title, y=title_y, fontweight=title_fontweight)
    if y_label is not None:
        ax.set_ylabel(y_label, labelpad=3, fontweight='bold')
    if x_label is not None:
        ax.set_xlabel(x_label, labelpad=9, fontweight='bold')


def _apply_ticks(ax, y_major_ticks, y_minor_ticks, x_major_ticks, x_minor_ticks, **kwargs):
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


def _apply_legend(ax, legend, legend_title='', **kwargs):
    """Helper to apply consistent legend formatting."""
    if legend is not False:
        ax.legend(
            title=legend_title,
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
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

    if custom_err is False:
        if kwargs.get('show_errorBars', True) is True:
            sns.barplot(data=dframe, x=x, y=y, hue=hue, hue_order=hue_order, palette=palette,
                        capsize=kwargs.get('capsize', 0.05), err_kws={'linewidth': 2, 'color': [0.1, 0.1, 0.1, 0.8]},
                        errorbar=('ci', 68), n_boot=2000, ax=_ax)
        else:
            sns.barplot(data=dframe, x=x, y=y, hue=hue, hue_order=hue_order, palette=palette,
                        capsize=0, err_kws={'linewidth': 0, 'color': [1, 1, 1]},
                        errorbar=None, ax=_ax)
    else:
        barplot_err(data=dframe, x=x, y=y, yerr=custom_err_label, hue=hue, hue_order=hue_order,
                    palette=palette, errwidth=2, capsize=.05, errcolor=[.1, .1, .1, 0.8], ax=_ax)

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

    _apply_labels(_ax, title, x_label, y_label, **kwargs)
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


def plot_parameter_solutions(
    dframe, parameters=None, solution_col='de_seed', error_col='total_error',
    best=None, deduplicate=True, cmap='viridis', norm=None, markers=None,
    title='', y_label='Parameter value', x_label='', y_lim=None, palette='colorblind', colour_by='error',
    y_major_ticks=None, y_minor_ticks=None, save_name='', save_dir=None,
    fig_dims=(7, 5), my_dpi=100, line_alpha=0.7, best_line_alpha=1, line_width=0.5,
    best_line_width=3, point_size=35, best_point_size=35,
    close_plot=False, ax=None, fig=None, legend=False, legend_loc='center right', show_colorbar=False, **kwargs
):
    """Plot parameter solutions coloured by error using an optional shared colour scale."""

    _fig, _ax = _setup_figure(ax, fig, fig_dims)

    # Infer parameter names from freeParam_* columns when not explicitly supplied
    if parameters is None:
        free_param_cols = [col for col in dframe.columns if col.startswith('freeParam_')]
        if not free_param_cols:
            raise ValueError('No parameters supplied and no freeParam_* columns found.')
        parameters = dframe[free_param_cols].iloc[0].tolist()

    if not parameters:
        raise ValueError('parameters must contain at least one parameter.')

    missing = [param for param in parameters if param not in dframe.columns]
    if missing:
        raise ValueError(f'Parameter columns not found: {missing}')

    if error_col not in dframe.columns:
        raise ValueError(f'Error column not found: {error_col}')

    if colour_by not in ('error', 'category', 'bestfit'):
        raise ValueError("colour_by must be either 'error' or 'category' or 'bestfit'.")

    # Keep one row per solution when the input contains repeated observations
    plot_df = dframe.copy()
    if deduplicate:
        plot_df = plot_df.drop_duplicates(subset=[solution_col])

    if colour_by == 'error':
        errors = plot_df[error_col].astype(float)

        # Create a colour normalisation if one was not supplied
        if norm is None:
            norm = plt.Normalize(vmin=errors.min(), vmax=errors.max())

        cmap_obj = plt.get_cmap(cmap)
        colours = [cmap_obj(norm(error)) for error in errors]
    else:
        colours = sns.color_palette(palette, n_colors=len(plot_df))

    # Identify the best solution
    if best == 'min':
        best_idx = plot_df[error_col].idxmin()
    elif best == 'max':
        best_idx = plot_df[error_col].idxmax()
    else:
        best_idx = None

    best_solution = plot_df.loc[best_idx, solution_col] if best else None

    # Plot each solution with colour representing its error
    markers = markers if markers and len(markers) >= len(plot_df) else ['o'] * len(plot_df)
    for idx, (_, row) in enumerate(plot_df.iterrows()):
        if best and row[solution_col] == best_solution:
            continue
        _ax.plot(
            range(len(parameters)), [row[param] for param in parameters], color=colours[idx],
            alpha=line_alpha,  linewidth=line_width, label=str(row[solution_col]), marker=markers[idx],
            markersize=point_size ** 0.5, markeredgecolor='none', markeredgewidth=0, zorder=2,
        )

    # Plot best solution
    for idx, (_, row) in enumerate(plot_df.iterrows()):
        if best and row[solution_col] == best_solution:

            if colour_by == 'bestfit':
                colours[idx] = 'black'

            _ax.plot(
                range(len(parameters)), [row[param] for param in parameters], color=colours[idx],
                alpha=best_line_alpha,  linewidth=best_line_width, label=str(row[solution_col]), marker=markers[idx],
                markersize=best_point_size ** 0.5, markeredgecolor='black', markeredgewidth=2.0, zorder=3
            )

    # Optionally add a colourbar for standalone use
    if show_colorbar and colour_by == 'error':
        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array(errors)
        cbar = _fig.colorbar(sm, ax=_ax, pad=0.02)
        cbar.set_label(error_col)

    # Configure axes
    _ax.set_xticks(range(len(parameters)))
    _ax.set_xticklabels(parameters)
    _ax.set(ylim=y_lim)
    _ax.yaxis.grid(which='both', linestyle='-', alpha=0.4)

    _apply_labels(_ax, title, x_label, y_label, **kwargs)
    _apply_ticks(_ax, y_major_ticks, y_minor_ticks, None, None)

    sns.despine(fig=_fig, ax=_ax, top=True, right=True)

    _fig.set_dpi(my_dpi)

    if close_plot:
        plt.close(_fig)

    return _fig, _ax


def plot_parameter_solutions_by_decoder(
    dframe, decoder_col='decoder', decoders=None, parameters=None, show_title=True,
    solution_col='de_seed', error_col='total_error', best='min', colour_by='bestfit',
    cmap='viridis', y_label='Parameter value', y_lim=None, panel_dims=(6, 4), palette='colorblind',
    my_dpi=200, sharey=True, axes=None, colorbar_label=None, cbar_lim=None, show_colorbar=True, **kwargs
):
    """Plot parameter solutions for multiple decoders with a shared error colour scale."""

    # Determine decoder order
    if decoders is None:
        decoders = dframe[decoder_col].drop_duplicates().tolist()

    if colour_by == 'error':
        # Create one shared colour scale across all decoders
        errors = dframe[error_col].astype(float)
        if cbar_lim is None:
            cbar_lim = (errors.min(), errors.max())
        norm = plt.Normalize(vmin=cbar_lim[0], vmax=cbar_lim[1])
        plot_cmap = plt.get_cmap(cmap)

    elif colour_by == 'category':
        norm = None
        plot_cmap = sns.color_palette(palette, n_colors=len(parameters))

    elif colour_by == 'bestfit':
        norm = None
        palette = [(0.5, 0.5, 0.5)] * len(parameters)
        plot_cmap = sns.color_palette(palette, n_colors=len(parameters))

    else:
        raise ValueError("colour_by must be 'error' or 'category' or 'bestfit'")

    # Create the multi-panel figure
    if axes is None:
        w, h = panel_dims
        n_cols = dframe[decoder_col].nunique()
        fig, axes = plt.subplots(1, len(decoders), figsize=(w * n_cols, h), sharey=sharey)
    else:
        fig = np.atleast_1d(axes)[0].figure

    axes = np.atleast_1d(axes)

    # Plot each decoder using the same colour normalisation
    for i, decoder in enumerate(decoders):
        decoder_df = dframe[dframe[decoder_col] == decoder].copy()

        plot_parameter_solutions(
            decoder_df, parameters=parameters, solution_col=solution_col, palette=palette,
            error_col=error_col, best=best, cmap=plot_cmap, norm=norm, colour_by=colour_by,
            title=str(decoder) if show_title else '', y_label=y_label if i == 0 else '',
            y_lim=y_lim, ax=axes[i], fig=fig, show_colorbar=False, **kwargs
        )

    if colour_by == 'error':
        # Add one colourbar shared by all panels, positioned to the far right
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap), norm=norm)
        sm.set_array(errors)

        # These are coordinates relative to axes[-1] [x0, y0, width, height]
        cax = axes[-1].inset_axes(
            [1.05, 0.0, 0.05, 1.0],
            transform=axes[-1].transAxes,
        )

        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label(colorbar_label if colorbar_label is not None else error_col,
                       labelpad=kwargs.get('cbar_labelpad'))

    fig.subplots_adjust(wspace=kwargs.get('wspace', 0.22))

    return fig, axes


def plot_iqr(dframe, x, y, hue, hue_order=None, palette='colorblind', title='', x_label='', y_label='',
             y_lim=None, y_major_ticks=None, y_minor_ticks=None, save_name='', save_dir=None,
             fig_dims=(7, 4), my_dpi=100, close_plot=True, include_strip=False, ax=None, fig=None, **kwargs):
    """Create interquartile range (box) plot."""
    _fig, _ax = _setup_figure(ax, fig, fig_dims)

    if include_strip:
        sns.stripplot(data=dframe, x=x, y=y, hue=hue, hue_order=hue_order,
                      palette=np.asarray(palette) / 2, linewidth=1.6,
                      dodge=True, edgecolor=[1] * 3, alpha=0.5, size=6, jitter=.12, ax=_ax)

    median_props = dict(linestyle='--', linewidth=2.2, color='k', alpha=1)
    mean_props = {"marker": "s", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 6}
    box_props = dict(linestyle='-', linewidth=1.4, edgecolor=[0.95] * 3, alpha=1)

    sns.boxplot(data=dframe, x=x, y=y, hue=hue, hue_order=hue_order, palette=palette,
                linewidth=0.8, showfliers=False, showmeans=True, meanprops=mean_props,
                medianprops=median_props, boxprops=box_props, ax=_ax)

    _ax.set(ylim=y_lim)

    # Apply hatches
    n_bars = len(dframe[x].unique())
    hatches = [""] * n_bars + [""] * n_bars + ["\\"] * n_bars + ["/"] * n_bars
    for hatch, patch in zip(hatches, _ax.artists):
        patch.set_hatch(hatch)

    _apply_labels(_ax, title, x_label, y_label, title_y=1.13)
    _apply_ticks(_ax, y_major_ticks, y_minor_ticks, None, None)
    _apply_legend(_ax, kwargs.get('legend', True), kwargs.get('legend_title', ''))

    sns.despine(fig=_fig, ax=_ax, top=True, right=True, offset=5)
    _fig.set_dpi(my_dpi)
    _fig.tight_layout()

    _save_figure(_fig, save_dir, save_name, my_dpi)

    if close_plot:
        plt.close(_fig)

    return _fig, _ax


def plot_line(dframe, x, y, hue, palette='colorblind', hue_order=None, title='', y_label='', x_label='',
              y_lim=None, y_major_ticks=None, y_minor_ticks=None, yscale='linear',
              x_major_ticks=None, x_minor_ticks=None, save_name='', save_dir=None,
              fig_dims=(6, 4), my_dpi=300, marker="o", x_type='cat',
              ax=None, fig=None, close_plot=False, **kwargs):
    """Create line/point plot with flexible configuration."""
    _fig, _ax = _setup_figure(ax, fig, fig_dims)
    sns.despine(fig=_fig, ax=_ax, top=True, right=True, offset=0)

    # Add horizontal reference lines
    if kwargs.get('hlines') is not None:
        for h in kwargs['hlines']:
            _ax.axhline(h, xmin=0.0, xmax=1, color=(0.4, 0.4, 0.4), alpha=0.25,
                        linestyle='--', linewidth=1.7)

    # Create appropriate plot type
    if x_type == 'scatter':
        sns.scatterplot(data=dframe, x=x, y=y, hue=hue, hue_order=hue_order,
                        marker=marker, style=hue, alpha=kwargs.get('alpha'), ax=_ax)
    elif x_type == 'num':
        sns.lineplot(data=dframe, x=x, y=y, hue=hue, hue_order=hue_order, marker=marker,
                     alpha=kwargs.get('alpha'), palette=palette, ax=_ax)
        _ax.set(xlim=(0, None))
    elif x_type == 'cat':
        if len(dframe[hue].unique()) == 1:
            kwargs['dodge'] = False
        sns.pointplot(data=dframe, x=x, y=y, hue=hue, hue_order=hue_order, palette=palette,
                      markers=kwargs.get('markers', "o"), dodge=kwargs.get('dodge'),
                      linewidth=kwargs.get('linewidth', 1.5),
                      errorbar=kwargs.get('errorbar'),
                      err_kws={'linewidth': kwargs.get('errwidth')}, ax=_ax)

    if kwargs.get('grid') is not True:
        _ax.yaxis.grid(which='both', linestyle='-', alpha=0.4)

    _ax.set(ylim=y_lim, yscale=yscale)

    # Style lines and markers
    for i_line in _ax.lines:
        i_line.set_linewidth(kwargs.get('linewidth', 2))
        if x_type in ['num', 'cat']:
            i_line.set_markersize(kwargs.get('marker_size', 15))
            i_line.set_markeredgecolor([1, 1, 1])
            i_line.set_markeredgewidth(0.7)
            i_line.set_alpha(kwargs.get('line_alpha', 0.9))
            i_line.set_linestyle(kwargs.get('linestyles', '-'))

    if x_type not in ['num', 'cat']:
        for i_marker in _ax.collections:
            if kwargs.get('marker_size'):
                i_marker.set_sizes(kwargs['marker_size'])
            if kwargs.get('marker_alpha'):
                i_marker.set_alpha(kwargs.get('marker_alpha'))

    _apply_labels(_ax, title, x_label, y_label, **kwargs)
    _apply_ticks(_ax, y_major_ticks, y_minor_ticks, x_major_ticks, x_minor_ticks)
    _apply_legend(_ax, kwargs.get('legend', True), kwargs.get('legend_title', ''))

    if kwargs.get('tight_layout') is not False:
        try:
            _fig.tight_layout(rect=(0, 0, 0.85, 1))
        except:
            pass
        _fig.tight_layout()

    _save_figure(_fig, save_dir, save_name, my_dpi)

    if close_plot:
        plt.close(_fig)

    return _fig, _ax


def plot_scatter(dframe, x, y, hue, palette='colorblind', hue_order=None, title='', y_label='', x_label='',
                 y_lim=None, y_major_ticks=None, y_minor_ticks=None, yscale='linear',
                 x_major_ticks=None, x_minor_ticks=None, save_name='', save_dir=None,
                 fig_dims=(6, 4), my_dpi=100, marker=None, x_type='cat',
                 ax=None, fig=None, close_plot=False, **kwargs):
    """Create scatter plot (wrapper around plot_line for backward compatibility)."""
    return plot_line(dframe=dframe, x=x, y=y, hue=hue, palette=palette, hue_order=hue_order,
                     title=title, y_label=y_label, x_label=x_label, y_lim=y_lim,
                     y_major_ticks=y_major_ticks, y_minor_ticks=y_minor_ticks, yscale=yscale,
                     x_major_ticks=x_major_ticks, x_minor_ticks=x_minor_ticks,
                     save_name=save_name, save_dir=save_dir, fig_dims=fig_dims, my_dpi=my_dpi,
                     marker=marker, x_type=x_type, ax=ax, fig=fig, close_plot=close_plot, **kwargs)



