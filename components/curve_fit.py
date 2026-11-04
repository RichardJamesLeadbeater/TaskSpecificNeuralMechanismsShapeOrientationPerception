import numpy as np
import os
from scipy.optimize import curve_fit
import pylab
from mytools.my_seaborn import load_seaborn_prefs


def logistic_func(x, a, b, intercept=0.5):
    # x=xval    a=midpoint    b=slope
    maxval = 1
    y = intercept + (maxval - intercept) / (1 + np.exp(-(x - a) / b))

    return y


def fit_logistic(x, y, maxfev=1000, p0=None, bounds=(-np.inf, np.inf)):
    # set default
    if p0 is None:
        p0 = [np.mean(x), 0.5]  # starting parameters for mod_orispace fit

    if bounds == (-np.inf, np.inf):
        bounds = ([x.min(), 0.2], [x.max(), 10])  # bounds for threshold and slope (higher vals = more gradual slope)

    x = np.asarray(x)
    y = np.asarray(y)

    # determine params of logistic mod_orispace using least-squares method
    popt, pcov = curve_fit(logistic_func, x, y, method='trf', maxfev=maxfev, p0=p0, bounds=bounds)
    # popt, pcov = curve_fit(logistic_function, x_vals, y_vals, method='lm', maxfev=4000)

    thresh_opt = popt[0]
    thresh_std = np.sqrt(np.diag(pcov))[0]
    slope_opt = popt[1]
    slope_std = np.sqrt(np.diag(pcov))[1]

    # output of logistic function given x-vals and optimised params
    x = np.linspace(x.min(), x.max(), len(x) * 1000)
    y = logistic_func(x, *popt)

    # namedtuple does not work with multiprocessing
    return {'x': x, 'y': y, 'threshold': thresh_opt, 'slope': slope_opt,
            't_std': thresh_std, 'slope_std': slope_std}


def plot_psychometric(x_axis, y_axis, x_fit_axis, y_fit_axis, threshold_value=None, title='', close_all=True,
                      _save=False, save_dir=None, save_name='psychometric_function', _percent=False, legend=True,
                      dpi=450, x_label='Orientation Difference ($^\circ$)', y_label='Proportion Correct',
                      y_lim=(0.45, 1.02), x_lim=None, x_scale='linear'):
    x_axis = np.asarray(x_axis)
    y_axis = np.asarray(y_axis)
    # load_seaborn_prefs()
    if close_all:
        pylab.close('all')
    else:
        pass
    if _percent is True:  # convert to percentage units
        y_axis *= 100
        y_fit_axis *= 100
        y_label = 'Percent Correct (%)'
        y_lim = (45, 105)
        yticks = [50, 75, 100]
    else:
        yticks = [0.5, 0.75, 1.0]
    load_seaborn_prefs(custom_rc={'lines.markersize': 7, 'axes.labelsize': 16})
    fig = pylab.figure(figsize=(6, 4))
    pylab.plot(x_axis, y_axis, 'o', label='data')
    pylab.plot(x_fit_axis, y_fit_axis, label='fit')
    # pylab.ylim(0 - .01, 1.01)
    # pylab.xlim(x_axis.min() - 0.01, x_axis.max() + 0.01)
    if threshold_value is not None:
        threshold_idx = np.argmin(abs(x_fit_axis - threshold_value))
        pylab.plot([threshold_value, threshold_value], [0, y_fit_axis[threshold_idx]], 'k--', alpha=0.5)
        pylab.plot([0, threshold_value], [y_fit_axis[threshold_idx], y_fit_axis[threshold_idx]], 'k--', alpha=0.5)
    # adjust plot
    ax = fig.axes[0]
    if x_lim is None:
        x_lim = [x_axis.min(), x_axis.max()]
    x_lim[1] = x_lim[1] + 0.02
    ax.set(ylim=y_lim, xlim=x_lim)
    ax.set_xlabel(x_label, fontweight='bold', fontfamily='sans-serif')  # fontsize=16,
    ax.set_ylabel(y_label, fontweight='bold', fontfamily='sans-serif')
    ax.set_xscale(x_scale)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{_y}" for _y in yticks])
    # ax.set_xticklabels(labels=ax.get_xticklabels(), fontsize=10)
    # ax.set_yticklabels(labels=ax.get_yticklabels(), fontsize=10)
    pylab.title(title, pad=20, fontsize=18)
    if legend is True:
        pylab.legend(loc='best')
    fig.tight_layout()
    if _save is True or save_dir is not None:
        pylab.savefig(os.path.join(save_dir, f"{save_name}.png"), dpi=dpi)

