import numpy as np
from scipy.optimize import curve_fit, minimize
import os
import pylab
import pandas as pd
from mytools.bootstrap import resample, bootstrap_logistic_fit
from mytools.dict_tools import init_cols, append_dicty
from mytools.dframe_tools import index_dframe, get_cond_combos


class LevelHandler:
    """Contains info about the current val and holds data for correct answers"""
    def __init__(self, std_ori_, val, max_count=10):
        self.std_ori = std_ori_
        self.val = val
        self.max_count = max_count  # n_reps of levels
        self.current_count = 0
        self.n_correct = 0
        self.proportion_correct = None
        self.trial_rotation = None  # CW or CCW

    def calc_proportion_correct(self):
        try:
            self.proportion_correct = (1 / self.current_count) * self.n_correct
        except ZeroDivisionError:
            self.proportion_correct = 0
        return self.proportion_correct


class MocsHandler:
    def __init__(self, std_ori, levels, n_reps, label=''):
        self.std_ori = std_ori
        self.levels = levels  # all possible ori_offset levels
        self.n_reps = n_reps  # n_reps of each val
        self.all_levels = [LevelHandler(std_ori, i_, n_reps) for i_ in levels]  # each val as class
        self.label = label
        self.current_level = None
        self.continue_trials = True
        self.results = None
        self.extraInfo = None

    def set_trial_level(self):
        # randomly choose val from list in self.all_levels
        possible_level_idx = []  # possible levels for present trial
        for idx, i_level in enumerate(self.all_levels):
            if i_level.current_count < self.n_reps:
                possible_level_idx.append(idx)
        chosen_level_idx = np.random.choice(possible_level_idx)
        self.current_level = self.all_levels[chosen_level_idx]
        return self.current_level

    def get_trial_level(self, update=False):
        if update:  # new val
            trial_level = self.set_trial_level()  # randomise val for this trial
            return trial_level
        else:
            self.current_level = self.all_levels[-1]  # if not update give last val
            return self.current_level

    def update_correct(self, is_correct, mocs_level=None):
        if mocs_level is not None:
            self.current_level = mocs_level
        else:
            mocs_level = self.current_level

        if all(i != is_correct for i in [0, 1]):
            raise TypeError('0 for incorrect\n1 for correct')
        else:
            mocs_level.current_count += 1  # update the current count of this and original
            mocs_level.n_correct += is_correct
        self.check_if_continue()
        # end experiment when all levels shown 10 times

    def check_if_continue(self):
        if all(i_lvl.current_count >= self.n_reps for i_lvl in self.all_levels):
            self.continue_trials = False
            self.get_results()
        else:
            self.continue_trials = True
        return self.continue_trials

    def get_results(self):
        mocs_results = init_cols(['val', 'proportion_correct'])
        # calc proportion correct for each val
        for i_lvl in self.all_levels:
            mocs_results['val'].append(np.round(i_lvl.val, 2))
            mocs_results['proportion_correct'].append(np.round(i_lvl.calc_proportion_correct(), 3))
        self.results = pd.DataFrame(mocs_results)
        self.results['std_ori'] = self.std_ori
        self.results['label'] = self.label
        return self.results


def logistic_func(x, a, b, maxval=1, minval=0.5):
    # x=xval    a=midpoint    b=slope
    # c + (L - c) / (1 + exp(-k * (x - x0)))
    # y = minval + ((maxval - minval) / (1 + np.exp((x - a) / b)))  # curve shifted
    y = minval + (1 - minval) / (1 + np.exp(-b * (x - a)))  # curve shifted
    return y


def fit_logistic(x, y, maxfev=1000, p0=None, bounds=(-np.inf, np.inf), method='trf',
                 _bootstrapping=False):
    if any(_m == method for _m in ['trf']):
        # set default
        if p0 is None:
            p0 = [np.mean(x), 3]  # starting parameters for curve fit
        if bounds is None:
            bounds = ([0, 0], [x.max(), 10])  # bounds for threshold and slope (higher vals = more gradual slope)
    if _bootstrapping is True:
        maxfev = 500  # speed up fit if bootstrapping
    x = np.asarray(x)
    y = np.asarray(y)
    # determine params of logistic curve using least-squares method
    try:
        popt, pcov = curve_fit(logistic_func, x, y,
                               method=method,  # lm, trf (bounds), dogbox
                               maxfev=maxfev, p0=p0, bounds=bounds,
                               )
    except RuntimeError:
        print(f"Fit failed: y = {y}\nRetrying with maxfev=5000")
        try:
            popt, pcov = curve_fit(logistic_func, x, y,
                                   method=method,  # lm, trf (bounds), dogbox
                                   maxfev=5000, p0=p0, bounds=bounds,
                                   )
        except RuntimeError:
            print(f"Fit failed on retry")
            return None

    thresh_opt = popt[0]
    thresh_std = np.sqrt(np.diag(pcov))[0]
    if _bootstrapping is True:  # just return required value if bootstrapping
        return thresh_opt
    slope_opt = popt[1]
    slope_std = np.sqrt(np.diag(pcov))[1]
    # output of logistic function given x-vals and optimised params
    x_fit = np.linspace(np.min(x), np.max(x), len(x) * 1000)
    y_fit = logistic_func(x_fit, *popt)
    # namedtuple does not work with multiprocessing
    return {'x': x_fit, 'y': y_fit, 'threshold': thresh_opt, 'slope': slope_opt,
            'threshold_std': thresh_std, 'slope_std': slope_std}


def fit_threshold(dframe, cond_list, level='offset', dv='proportion_correct', min_n=-np.inf, count_label=None,
                  threshold_propn=None, bootstrap_sem=False, bs_n_resamples=1000, _use_mp=True):
    cond_combos = get_cond_combos(dframe, cond_list)
    out_dframe = init_cols([*cond_list, 'n', 'threshold', 't_std', 't_sem', 'slope', 's_sem'])
    for i_condcombo in cond_combos:
        i_dframe = index_dframe(dframe, columns=cond_list, values=i_condcombo)
        if i_dframe.empty:
            continue
        x_data = sorted(i_dframe[level].unique().astype('float'))
        y_data = []
        if bootstrap_sem is True:
            bs_y = []
        for i_lvl in x_data:
            i_data_lvl = i_dframe[(i_dframe[level] == i_lvl)]
            if len(i_data_lvl) < min_n:  # ensure minimum number of runs is completed
                continue
            if count_label is None:
                n_runs = len(i_data_lvl)
            else:
                n_runs = len(i_dframe[count_label].unique())
            if i_data_lvl.empty is True:
                continue
            pcorr = i_data_lvl[dv].mean()  # mean proportion correct
            y_data.append(pcorr)
            if bootstrap_sem is True:
                resampled_y_data = resample(i_data_lvl[dv], n_samples=bs_n_resamples)
                bs_y.append(np.mean(resampled_y_data, 1))  # mean value for each resampling
        if len(y_data) != len(x_data):  # ensure minimum number of runs is completed
            continue
        # Set 0.5 to chance val to aid curve fitting
        x_data = np.asarray([0] + list(x_data))
        y_data = np.asarray([0.5] + list(y_data))
        dfit = fit_logistic(x_data, y_data, maxfev=1000, p0=(np.mean(x_data), 3), bounds=([0, 0], [x_data.max(), 10]),
                            method='trf')
        if dfit is None:  # if fit failed
            continue
        if bootstrap_sem is False:
            # If using metamers
            if threshold_propn is not None:  # metamers
                threshold_idx = np.argmin(abs(dfit['y'] - 0.57))
                dfit['threshold'] = dfit['x'][threshold_idx]
            # APPEND DATA
            append_dicty(out_dframe, [*i_condcombo, n_runs, dfit['threshold'], dfit['threshold_std'],
                                      dfit['threshold_std'] / np.sqrt(n_runs), dfit['slope'],
                                      dfit['slope_std'] / np.sqrt(n_runs),])
        else:
            # BOOTSTRAP STANDARD ERROR
            bs_y = ([np.asarray([0.5] * bs_n_resamples)] + bs_y)  # add in 0.5 perf for fake cond (ori_diff=0)
            bs_y = list(np.asarray(bs_y).transpose())  # transpose: each array corresponds to a single run
            bs_data = bootstrap_logistic_fit(fit_logistic, x_data, bs_y, maxfev=500,
                                             p0=(np.mean(x_data), 3), bounds=([0, 0], [x_data.max(), 10]),
                                             method='trf', use_mp=_use_mp)
            append_dicty(out_dframe, [*i_condcombo, n_runs, dfit['threshold'], bs_data.sem * np.sqrt(n_runs),
                                      bs_data.sem, None, None])
    out_dframe = pd.DataFrame(out_dframe).sort_values(by=cond_list)
    return out_dframe


def plot_psychometric(x_axis, y_axis, x_fit_axis, y_fit_axis, threshold_value=None, title='', close_all=True,
                      _save=False, save_dir=None, save_name='psychometric_function', _percent=False, legend=True,
                      dpi=300, x_label='Orientation Difference ($^\circ$)', y_label='Proportion Correct',
                      y_lim=(0.5, 1.02), x_lim=None, **kwargs):
    if close_all:
        pylab.close('all')
    else:
        pass
    if _percent is True:  # convert to percentage units
        y_axis *= 100
        y_fit_axis *= 100
        y_label = 'Percent Correct (%)'
        y_lim = (0, 102)
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
    # ax.set_xticklabels(labels=ax.get_xticklabels(), fontsize=10)
    # ax.set_yticklabels(labels=ax.get_yticklabels(), fontsize=10)
    pylab.title(title, pad=20, fontsize=18)
    if legend is True:
        pylab.legend(loc='best')
    fig.set_dpi(dpi)
    fig.tight_layout()
    if _save is True or save_dir is not None:
        pylab.savefig(os.path.join(save_dir, f"{save_name}.png"))

