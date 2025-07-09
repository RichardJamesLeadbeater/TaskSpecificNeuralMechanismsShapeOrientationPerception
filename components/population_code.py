import numpy as np
from numpy import asarray as arr, newaxis as nax
from scipy.linalg import hankel
from scipy.optimize import minimize
import pickle
import random
from .general_functions import get_attrib, save_attrib, find_nearest, generate_gaussian, generate_cosine, generate_vonmises,\
    sort_by_standard, normalise, poissonlike_noise, get_boundaries
from components.run import set_random_seed
from . import debugger
import matplotlib.pyplot as plt
import os
from mytools.dframe_tools import init_cols


def prefs_sampling_error(prefs_fun, scale, desired_n, **kwargs):
    # Calculate error between the desired and actual n_cardinal
    scale = scale - kwargs.get('bias', 0)  # Remove bias if given
    # Get n_cardinal
    output = prefs_fun(scale=scale, size=kwargs.get('size'))
    n = output['n_cardinal']
    error = abs(n - desired_n)
    if kwargs.get('_print') is True:
        print(f"\nscale = {scale}"
              f"\nerror = {error}")
    return error


def gen_population_code(cardinal_params, oblique_params, orispace, pref_sampling, **kwargs):
    # Set parameters for population code and init
    if pref_sampling == 'indiv':
        # Set parameters for each individual ori_population
        [_params.set_n_prefs() for _params in [cardinal_params, oblique_params]]
        ori_pops = {}
        for label, val in zip(['H', '45', 'V', '135'], [0, 45, 90, 135]):
            if any(i == label for i in ['H', 'V']):
                i_params = cardinal_params.__dict__.copy()  # copy required to remove dependency from class
            else:
                i_params = oblique_params.__dict__.copy()
            # custom vals for each population
            i_params['name'] = label
            i_params['std_ori'] = val
            i_params['boundaries'] = get_boundaries(val, i_params['pref_range'], orispace.max_tile)
            ori_pops[label] = NeuralPopulation(params=i_params,
                                               baseline_params=oblique_params.__dict__.copy(),  # for distribs
                                               orispace=orispace, name=label, _rate_or_n=kwargs.get('rate_or_n', 'n'),
                                               debug=kwargs.get('debug', False))
        population_code = PopulationCode(orispace=orispace,
                                         neural_populations=[ori_pops[key] for key in ori_pops],
                                         pooling='summation', debug=kwargs.get('debug', False),
                                         pref_sampling='indiv')
        # adjust boundaries based off average sampling rate of each neighbouring population
        population_code.adjust_boundaries()
        # generate tunings & prefs for each pop
        population_code.gen_tunings(stack=True, window=True)  # stack & calc rollingwindow of prefs
        # collate params from each NeuralPop
        population_code.get_params()
    elif pref_sampling == 'whole':
        # Params defined for the whole populations (default)
        global_params = {'n_prefs': (cardinal_params.n_prefs + oblique_params.n_prefs) * 2,
                         'propn_cardinal': cardinal_params.n_prefs / (cardinal_params.n_prefs + oblique_params.n_prefs),
                         'noise_type': 'poisson',
                         'pref_distrib': 'cos',  # 'flat' or 'cos' (flat works better for low n and high propn_c)
                         'param_distrib': 'cos',
                         'tuning_dist': 'von_mises',
                         'pooling_q': cardinal_params.pooling_q,
                         }
        if kwargs.get('g_params') is not None:  # update default global_params
            for _key in kwargs['g_params']:
                global_params[_key] = kwargs['g_params'][_key]

        anisotropy_info = {'param_labels': ['n_prefs', 'fwhm', 'pref_distrib', 'param_distrib', 'rmax', 'spont',
                                            'noise_scale', 'crf_exponent', 'crf_c50'],
                           'cardinal_params': cardinal_params.__dict__, 'oblique_params': oblique_params.__dict__}
        population_code = PopulationCode(orispace=orispace, anisotropy=anisotropy_info, global_params=global_params,
                                         debug=kwargs.get('debug', False), pref_sampling='whole')
        population_code.init_population_code()
    else:
        population_code = None
    return population_code


def test_gen_population_code(cardinal_params, oblique_params, orispace, _rate_or_n='n', _debug=False):
    # set parameters for each ori_population
    ori_pops = {}
    for ori_label, ori_val in zip(['H', '45', 'V', '135'], [0, 45, 90, 135]):
        # Parameters of each orientation category
        if any(i == ori_label for i in ['H', 'V']):
            i_params = cardinal_params.__dict__.copy()  # copy required to remove dependency from class
        else:
            i_params = oblique_params.__dict__.copy()
        i_params['name'] = ori_label
        i_params['std_ori'] = ori_val
        i_params['boundaries'] = get_boundaries(ori_val, i_params['pref_range'], orispace.max_tile)
        # Init neural population for this orientation category
        ori_pops[ori_label] = NeuralPopulation(params=i_params, baseline_params=oblique_params.__dict__.copy(),
                                               orispace=orispace, name=ori_label, _rate_or_n=_rate_or_n)
    # Init population code containing the neural population for each orientation category
    population_code = PopulationCode(orispace=orispace, neural_populations=[ori_pops[key] for key in ori_pops],
                                     debug=_debug, pref_sampling='indiv')
    # adjust boundaries based off average sampling rate of each neighbouring population
    population_code.adjust_boundaries()
    # generate tunings & prefs for each pop
    population_code.gen_tunings(stack=True, window=True)  # stack & calc rollingwindow of prefs
    # collate params from each NeuralPop
    population_code.get_params()
    return population_code


# TODO: implement in model
def correlated_noise():
    # auto correlation
    # Generate temporally correlated Gaussian noise
    n_samples = 180
    rho = 0.5  # correlation strength
    sigma = 1  # stdev of noise
    # Generate uncorrelated Gaussian white noise
    white_noise = np.random.normal(0, sigma, n_samples)

    # Create correlated noise by filtering white noise through an AR process
    correlated_noise = np.zeros(n_samples)
    for t in range(1, n_samples):
        # Each point in space depends on the previous value
        correlated_noise[t] = rho * correlated_noise[t - 1] + (1 - rho) * white_noise[t]
    # Optionally, normalize the correlated noise to have the desired standard deviation
    correlated_noise *= sigma / np.std(correlated_noise)


def autocorrelation_spatial():
    tiling = np.linspace(-180, 180, 36000 + 1)
    correlation_length = 1  # Adjust this value to control the correlation length (rate of decay in filter)

    # Create a spatial filter to introduce spatial correlation
    def spatial_filter(x, correlation_length, filter='exp'):
        if filter == 'exp':
            fun = np.exp(-x / correlation_length)
        elif filter == 'gauss':
            fun = np.exp(-0.5 * (x / correlation_length) ** 2)
        return fun

    # Generate Gaussian noise along the spatial dimension
    noise = np.random.normal(0, 1, len(tiling))
    # Apply the spatial filter to the white noise
    filter_type = 'gauss'
    filter_mode = 'full'  # mode or full
    correlated_noise = np.convolve(noise, spatial_filter(x, correlation_length, filter_type), mode=filter_mode)
    # Optionally, normalize the correlated noise to have a desired standard deviation
    desired_sigma = 1.0  # Adjust this value as needed
    correlated_noise *= desired_sigma / np.std(correlated_noise)
    # Plot the spatially correlated noise
    plt.figure(figsize=(8, 6))
    if filter_mode == 'mode':
        plt.plot(x, correlated_noise)
    elif filter_mode == 'full':
        plt.plot(tiling, correlated_noise[:len(tiling)])
    plt.title("1D Spatially Correlated Noise")
    plt.xlabel("Spatial Dimension (e.g., Orientation)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()
    z = 0


class Params:
    def __init__(self, dicty=None, fwhm=None, sigma=None, kappa=None, deg_per_pref=None, n_prefs=None,
                 pref_distrib='flat',
                 param_distrib='flat', pref_range=None, rmax=None, spont=None, crf_exponent=None, crf_c50=None,
                 pooling=None, noise_scale=1, noise_type='poisson', correlated_noise=None,
                 baseline_params=None, pooling_exponent=None, pooling_extent=None, run_n_prefs=None, run_density=None,
                 run_sampling_rate=None, boundaries=None, std_ori=None, pooling_q=None,
                 template_popcode=None):
        # use dict to set attributes
        if isinstance(dicty, dict):
            for i_key in Params().__dict__:
                if dicty.get(i_key) is not None:
                    setattr(self, i_key, dicty[i_key])  # if exists then set value
                else:
                    setattr(self, i_key, Params().__dict__[i_key])  # default value if not in dicty
        else:
            # set model parameters to be varied
            self.deg_per_pref = deg_per_pref  # neurons per degree
            self.n_prefs = n_prefs  # todo sampling alternative
            self.pref_distrib = pref_distrib
            self.param_distrib = param_distrib
            self.pref_range = pref_range
            self.fwhm = fwhm  # FWHM = s * sqrt(2ln2) = s * 2.3548
            self.sigma = sigma  # gaussian tuning: s = fwhm / sqrt(2ln2)
            self.kappa = kappa  # von mises tuning: k = ln0.5 / (cos(2h) - 1)  , h=hwhm
            # set constant model parameters`
            self.boundaries = boundaries
            self.rmax = rmax
            self.spont = spont
            self.pooling = pooling
            self.crf_exponent = crf_exponent
            self.crf_c50 = crf_c50  # semisaturation constant
            self.noise_scale = noise_scale
            self.noise_type = noise_type
            self.correlated_noise = correlated_noise
            self.baseline_params = baseline_params
            self.pooling_q = pooling_q
            self.pooling_exponent = pooling_exponent  # for magnitude-weighted pooling
            self.pooling_extent = pooling_extent  # for distance-weighted pooling
            self.std_ori = std_ori
            # update when altered during model initialising
            self.run_n_prefs = run_n_prefs
            self.run_density = run_density  # nrn per deg
            self.run_sampling_rate = run_sampling_rate  # deg per nr
            # Template PopCod
            self.template_popcode = template_popcode
        # Calculate sigma from FWHM
        if self.fwhm is not None:
            if self.sigma is None:
                self.sigma = self.fwhm / (2 * np.sqrt(2 * np.log(2)))  # s = fwhm / 2.3548
            if self.kappa is None:
                h = self.fwhm / 2
                self.kappa = np.log(0.5) / (np.cos(2 * h) - 1)  # Swindale (1998)

    def set_fwhm(self, fwhm):
        self.sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        self.fwhm = fwhm

    def set_n_prefs(self, n_prefs=None):
        if n_prefs is None:
            n_prefs = self.n_prefs
        if n_prefs != 0:
            self.deg_per_pref = 45 / n_prefs
        self.n_prefs = n_prefs

    def set_noise_scale(self, noise_scale):
        self.noise_scale = noise_scale

    def set_rmax(self, rmax=None):
        self.rmax = rmax


class Tiling:
    """tiling of feature space"""
    def __init__(self, min_tile, max_tile, step_size, name=None, _save_external=False):
        self.min_tile = min_tile
        self.max_tile = max_tile
        self.range = abs(min_tile - max_tile)
        self.step_size = step_size
        self.tiling = np.round(np.arange(min_tile, max_tile + step_size, step_size), 4)
        self.name = name
        if name is None:
            self.name = 'tiling'
        # if _save_external is True:
        #     self.save_attrib('tiling', _replace=True)  # saves externally and replaces w/ None

    def save_attrib(self, attrib_label, path=None, _replace=True):
        # save out attrib and replace og with None
        save_attrib(self.__dict__[attrib_label], f"{self.name}_{attrib_label}", path)
        if _replace is True:
            self.__dict__[attrib_label] = None  # .get_tiling() needed

    def get_attrib(self, attrib_label, path=None, _save_internal=False):
        attrib = get_attrib(f"{self.name}_{attrib_label}", path)  # returns tiling array
        if _save_internal is True:
            self.__dict__[attrib_label] = attrib
        return attrib


class Stimuli:
    """values of MyStimuli across feature space"""

    def __init__(self, ori=None, con=None, tiling=None):
        self.tiling = tiling
        self.ori = ori
        self.con = con
        _, self.ori_idx = find_nearest(tiling, ori)

    def get_stim_ori_idx(self, stim_ori=None):
        if stim_ori is None:
            stim_ori = self.ori
        self.ori, self.ori_idx = find_nearest(self.tiling, stim_ori)
        return self.ori_idx


class NeuralPopulation:
    """params and sampling for neural population with independent properties"""
    def __init__(self, params, orispace, baseline_params=None, name=None, debug=False, summation='summation',
                 _rate_or_n='n'):
        self.params = Params(params)  # params must be input as dicty
        self.orispace = orispace
        if baseline_params is not None:
            self.baseline_params = Params(baseline_params)  # must be input as dicty
        # initialise
        self.param_distribs = None
        self.prefs = None
        self.prefs_idx = None
        self.n_prefs = None  # set once list of prefs generated
        self.density = None  # neurons per degree, set once prefs are generated (assumes 45 deg range)
        self.tunings = None
        self.population_response = None
        self.population_noise = None
        self.neurons = None
        self.name = name
        if name is None:
            self.name = self.__name__
        self.summation = summation
        self._rate_or_n = _rate_or_n
        # debug
        self.debug = debug

    def save_attrib(self, attrib_label, path=None, _replace=True):
        # save out attrib and replace og with None
        save_attrib(self.__dict__[attrib_label], f"{self.name}_{attrib_label}", path)
        if _replace is True:
            self.__dict__[attrib_label] = None  # .get_tiling() needed

    def get_attrib(self, attrib_label, path=None, _save_internal=False):
        attrib = get_attrib(f"{self.name}_{attrib_label}", path)  # returns tiling array
        if _save_internal is True:
            self.__dict__[attrib_label] = attrib
        return attrib

    def gen_param_distrib(self, tiling, attrib):
        # distrib of param vals across orispace scaled from baseline to mod_val
        mod = self.params.__dict__[attrib]  # modified val
        bl = self.baseline_params.__dict__[attrib]  # baseline val
        curve = None
        if len(tiling) == 1:  # 1 neuron per population
            z = 0
            return arr([mod])
            # return my_asarray([mod])
        if self.params.param_distrib == 'flat':
            curve = np.ones(len(tiling))
            return curve * mod
            # return param_distrib
        elif self.params.param_distrib == 'cos':
            curve = generate_cosine(tiling)
        elif self.params.param_distrib.split('_')[0] == 'gauss':
            sigma = float(self.params.param_distrib.split('_')[1])
            curve = generate_gaussian(tiling, np.mean(tiling), sigma)
        # uses ratio between y1 and y2 to scale from min=scalar
        param_distrib = normalise(curve, val_start=bl, val_end=mod)  # relative to baseline val
        return param_distrib

    def generate_params(self):
        #  in: self.params.boundaries, self.params.deg_per_pref, min_tile, max_tile, tiling
        # fun: adjust sampling rate to nearest possible
        #      generate prefs and round to nearest value in tiling
        #      get idx of each pref in tiling
        # out: prefs, prefs_idx, deg_per_pref for each neuron from bounds and sampling freq
        # if 1D array then put boundaries into list
        if self.params.n_prefs == 0:
            self.prefs, self.prefs_idx = [], []
            return None
        tiling = self.orispace.tiling
        self.params.boundaries = np.atleast_2d(self.params.boundaries)
        prefs = []
        # only include param distribs between numbers
        p_dicty = self.params.__dict__
        param_distribs = init_cols([p for p in p_dicty if any(isinstance(p_dicty[p], typ) for typ in [float, int])])
        # prefs and param distribs within each boundary
        for idx, i_bounds in enumerate(self.params.boundaries):
            step_size = self.orispace.step_size
            # gen i_tiles rather than index, to deal with bounds beyond tiling limits
            if np.all(i_bounds == i_bounds[0]):  # if just one preference
                i_tiles = arr([i_bounds[0]])
                # i_tiles = my_asarray([i_bounds[0]])
            else:
                i_tiles = np.arange(i_bounds[0], i_bounds[1] + step_size, step_size)
            # PREFS #
            i_prefs = self.generate_prefs(i_bounds)
            # get prefs idcs w/in this segment of orispace and use to index params
            (i_prefs, i_prefs_idx) = find_nearest(i_tiles, i_prefs)
            i_prefs = np.atleast_1d(i_prefs)  # convert to array (enables scalars)
            i_prefs_idx = np.atleast_1d(i_prefs_idx)
            # PARAMS #
            # create distributions of params between boundaries in orispace
            for p in param_distribs:
                p_dist = self.gen_param_distrib(i_tiles, attrib=p)  # create distribution
                p_dist = p_dist[i_prefs_idx]  # use prefs_index to select suitable params
                param_distribs[p].append(p_dist)
            # Restrict distribs to limits of orispace +- 180 (deals with bounds of distribs beyond tiling limits)
            idcs = np.where((i_prefs >= np.min(tiling)) & (i_prefs <= np.max(tiling)))
            i_prefs = i_prefs[idcs]
            prefs.append(i_prefs)
            for p in param_distribs:
                if not param_distribs[p]:
                    continue
                param_distribs[p][-1] = param_distribs[p][-1][idcs]
        # altered prefs and sampling / density info
        n_prefs = np.max([len(i) for i in prefs])
        self.params.run_n_prefs = n_prefs
        self.params.run_density = n_prefs / 45  # assumes max len will cover 45 deg, min will cover 22.5
        self.params.run_sampling_rate = 45 / n_prefs
        # prefs into 1d array
        prefs = np.hstack([i for i in prefs])
        prefs, prefs_idx = find_nearest(tiling, prefs)  # prefs to nearest vals in whole orispace and get indices
        self.prefs = prefs
        self.prefs_idx = prefs_idx
        # params into 1d arrays
        for p in param_distribs:
            param_distribs[p] = np.hstack([i for i in param_distribs[p]])
        self.param_distribs = param_distribs
        # self.save_attrib('param_distribs', _replace=True)  # save externally to save space
        return self.prefs, self.prefs_idx, self.params.deg_per_pref

    def generate_prefs(self, bounds):
        # orispace  bounds
        ori_range = abs(bounds[0] - bounds[1])
        # new space to avoid bounds.min() < tiling.min()
        std_space = np.arange(bounds[0], bounds[1] + self.orispace.step_size, 0.001)
        curve = None
        if self.params.pref_distrib == 'flat':
            if self.params.n_prefs == 1:
                arr([np.mean(bounds)])
                # prefs = my_asarray([np.mean(bounds)])
            else:
                prefs = np.linspace(bounds[0], bounds[1], self.params.n_prefs)
            return prefs
        elif self.params.pref_distrib == 'cos':
            # curve = generate_cosine(np.linspace(0, 360, 100000))
            curve = generate_cosine(np.linspace(0, 1, 100000))
            density_ratio = 3
            curve = normalise(curve, 1, density_ratio)  # add baseline sampling rate and scale of density change
            probabilities = curve / np.sum(curve)
            b = np.random.choice(np.linspace(bounds[0], bounds[1], 100000), size=100000, p=probabilities)
            b = arr(sorted(b))  # Put in value order
            # b = my_asarray(sorted(b))
            idcs = np.round(np.linspace(0, len(b) - 1, self.params.n_prefs)).astype('int')  # uniform across indices
            prefs = b[idcs]
            # plt.plot(prefs)

            ######## TEST - 31aug23
            def rokem(size, beta, delta):
                theta = np.linspace(0, np.pi * 2, size)
                p = (beta * (1 - np.cos(4 * theta)) + delta) / 360
                return p
            a = rokem(size=100000, beta=2, delta=10)
            b = np.random.choice(np.linspace(0, 180, 100000), size=100000, p=a / np.sum(a))
            b = arr(sorted(b))
            # b = my_asarray(sorted(b))  # put in value order
            n_prefs = 180
            idcs = np.round(np.linspace(0, len(b) - 1, n_prefs)).astype('int')  # uniform across indices
            prefs = b[idcs]
            # plt.plot(prefs)
            ########

        elif self.params.pref_distrib.split('_')[0] == 'gauss':
            sigma = float(self.params.pref_distrib.split('_')[1])
            curve = generate_gaussian(std_space, np.mean(std_space), sigma)
            curve = curve[abs(curve - np.mean(curve)) < sigma * 3]  # remove all values > 3SDs from mean
        # NOTE: you can only change rate if you change n_neurons, and you can only change n_neurons if you change rate
        deg_per_nrn = normalise(1 / curve, val_start=self.baseline_params.deg_per_pref,
                                val_end=self.params.deg_per_pref, _invert=True)
        # todo 1. alternate sampling
        if self._rate_or_n == 'rate':  # can fix min and max sampling rate but alters n_neurons
            # note: np.trapz(deg_per_nrn * (range / n_deg), std_space) == ori_range
            n_degrees = np.trapz(y=deg_per_nrn, x=std_space)
            n_prefs = np.round(np.trapz(1 / deg_per_nrn, std_space)).astype('int')  # estimate n_prefs required
        else:
            n_prefs = self.params.n_prefs  # can fix n_prefs but alters sampling rate
        # Loop through step sizes (deg_per_nrn) and set initial pref values

        # TEST #
        m = self.baseline_params.deg_per_pref / self.params.deg_per_pref
        deg_per_nrn *= m
        # / #

        idcs = np.round(np.linspace(0, len(deg_per_nrn) - 1, n_prefs - 1)).astype('int')
        prefs = [0]  # init with fixed val
        for i_step in deg_per_nrn[idcs]:  # loop through each step size and set pref
            prefs = np.hstack([prefs, prefs[-1] + i_step])
        prefs = normalise(prefs, val_start=bounds[0], val_end=bounds[1])  # transform prefs to range between bounds
        if self.debug is True:
            plt.hist(prefs, bins=5)
            debugger.pref_spacing(prefs, self.baseline_params.deg_per_pref, self.params.deg_per_pref)  # how close to ideal
        return prefs

    def generate_tunings(self):
        """Input prefs and generate tunings"""
        if self.params.n_prefs == 0:
            return None
        param_distribs = self.param_distribs
        sigma = param_distribs['sigma']
        sigma = arr([sigma]).T
        # sigma = my_asarray([sigma]).transpose()

        # tiling = self.orispace.get_attrib('tiling')
        tiling = self.orispace.tiling
        tunings = generate_gaussian(tiling, self.prefs, sigma)  # gaussian tuning for each pref
        # normalise so each tuning curve has max=1
        cols_idcs = np.argmax(tunings, axis=1)
        rows_idcs = np.arange(0, len(cols_idcs), 1)
        max_vals = tunings[rows_idcs, cols_idcs]

        # tunings = tunings / my_asarray([max_vals]).transpose()  # normalise to 1
        tunings = tunings / arr([max_vals]).T
        tunings = (tunings.T * param_distribs['rmax']).T  # normalise to rmax
        self.tunings = tunings
        # self.save_attrib('tunings', _replace=True)
        if self.debug is True:
            plt.plot(tiling, tunings.T)
            plt.close('all')
            debugger.measure_fwhm(tunings[int(len(tunings) / 2), :], tiling)
        return tunings

    def contrast_response(self, pop_resp=None, stim_contrast=100):
        # scales population response according to contrast response function (CRF)
        # Albrecht & Hamilton (1992):
        #   n=3.4 (crf_exponent) c50=24 (semi-saturation constant: stim_contrast for half-max response)
        if pop_resp is None:
            if self.population_response is None:
                # pop_resp = self.get_attrib('tunings')
                pop_resp = self.tunings
            else:
                pop_resp = self.population_response
        n = self.params.crf_exponent
        c50 = self.params.crf_c50
        # 'hyperbolic ratio' (Albrecht & Hamilton, 1982; Contreras & Palmer, 2003)
        pop_resp = pop_resp * (stim_contrast ** n / (stim_contrast ** n + c50 ** n))
        self.population_response = pop_resp
        return self.population_response

    def spontaneous_firing(self, pop_resp=None):
        if pop_resp is None:
            # pop_resp = self.get_attrib('tunings')
            pop_resp = self.tunings
        # param_distribs = self.get_attrib('param_distribs')
        param_distribs = self.param_distribs
        # pop_resp = (pop_resp.T + (param_distribs['rmax'] * param_distribs['spont'])).T
        pop_resp = (pop_resp.T + param_distribs['spont']).T  # spont independent from rmax
        self.population_response = pop_resp
        return self.population_response

    def noisy_response(self, pop_resp=None):
        # noise added onto population response
        # param_distribs = self.get_attrib('param_distribs')
        param_distribs = self.param_distribs
        noise = None
        if pop_resp is None:
            # pop_resp = self.get_attrib('tunings')
            pop_resp = self.tunings
        if self.params.noise_type == 'poisson':
            # noise = poissonlike_noise(pop_resp, self.params.noise_scale)
            noise = poissonlike_noise(pop_resp, param_distribs['noise_scale'])
        if self.params.correlated_noise is not None:
            noise = self.gen_correlated_noise(noise)
        pop_resp += noise
        pop_resp[pop_resp <= 0] = 0  # remove negative firing rates
        # limit firing rates to rmax
        rmax_distrib = (np.ones(pop_resp.shape).T * param_distribs['rmax']).T  # match shape to allow indexing
        pop_resp[pop_resp > rmax_distrib] = rmax_distrib[rmax_distrib < pop_resp]  # set max to each nrns rmax
        # test pop_resp += 1000 then pop_resp should == rmax_distrib
        # update attribs
        self.population_response = pop_resp
        self.population_noise = noise
        return pop_resp, noise

    def gen_correlated_noise(self, noise):
        og_noise = noise.copy()
        if self.params.correlated_noise == 'common_source':
            # each neuron with shared_noise value that diminishes across orientation-space away from .pref
            test = []
            for nrn_idx, nrn in enumerate(self.prefs):
                # todo common source should be dependent on distance in ori-space, not just n_units from pref
                common_source_noise = generate_gaussian(self.prefs, nrn, sigma=2)
                test.append(common_source_noise)
                common_source_noise = (common_source_noise * np.ones([len(noise.T), 1])).T  # nrns x nreps
                extra_noise = noise[nrn_idx, :] * common_source_noise
                # extra_noise = (noise.transpose() * common_source_noise).transpose
                noise += extra_noise
            # for i in test:
            #     plt.plot(i)
            #     # todo apply correlated noise at a global level
            #     #       - neurons close to the boundary will not receive correlated noise from the other population
        if self.debug is True:
            debugger.noise_correlation_between_neurons(noise, og_noise)
        return noise

    # def get_neurons(self):
    #     self.neurons = [Neuron(pref, self.prefs_idx[idx], population=self) for idx, pref in enumerate(self.prefs)]


class PopulationCode:
    def __init__(self, orispace, anisotropy=None, neural_populations=None, name=None, debug=True, **kwargs):
        self.pref_sampling = kwargs.get('pref_sampling')  # whether old version used (individual neural populations)
        self.trial_resp_tuned = None
        self.trial_resp = None
        self.trial_param_distribs = None
        self.params_co = None
        self.param_distribs = None
        self.params = None  # callable with .get_params()
        self.debug = debug  # if True then error checks will occur
        self.orispace = orispace
        self.global_params = kwargs.get('global_params')
        if kwargs.get('global_params') is not None:
            self.global_params['n_samples'] = int(self.global_params['n_prefs'] * (self.orispace.range / 180))  # whole range (not per 180)
        # Anisotropic parameters
        self.anisotropy = anisotropy
        if neural_populations is None:
            neural_populations = []
        self.neural_populations = neural_populations
        # the following will be appended in methods of class
        self.all_tunings = []
        self.all_prefs = []
        self.all_prefs_idx = []
        self.all_neurons = None
        # the following will be calculated in methods of class
        self.all_prefs_windows = None
        self.all_prefs_windows_idx = None
        self.trial_stim_ori = None
        self.trial_stim_ori_idx = None
        self.trial_stim_con = None
        self.trial_prefs = None
        self.trial_prefs_idx = None
        self.trial_tunings = None
        self.trial_tunings_idx = None
        self.resp_tuned = None
        self.resp_contrast = None
        self.all_noise = None
        self.resp_noisy = None
        self.noise_type = 'poisson'
        self.decoded = None
        if name is None:
            name = 'populationcode'
        self.name = name
        # Template popcode
        self.template_popcode = None
        self.template_decoder = None
        self.saved_response = {}  # saves decoded vals in dict key=stim_ori
        self.decoded_template = None

    def save_attrib(self, attrib_label, path=None, _replace=True):
        # save out attrib and replace og with None
        save_attrib(self.__dict__[attrib_label], f"{self.name}_{attrib_label}", path)
        if _replace is True:
            self.__dict__[attrib_label] = None  # .get_tiling() needed

    def get_attrib(self, attrib_label, path=None, _save_internal=False):
        attrib = get_attrib(f"{self.name}_{attrib_label}", path)  # returns tiling array
        if _save_internal is True:
            self.__dict__[attrib_label] = attrib
        return attrib

    def get_bounds_adjusters(self, value_list=None, operation='mean'):
        # used to get average sampling frequency to allow boundary adjustment
        """for each value, get average with each neighbouring value"""
        if value_list is None:
            value_list = []
            for i_pop in self.neural_populations:
                if i_pop.params.param_distrib == 'flat':
                    value_list.append(i_pop.params.deg_per_pref)  # if jarring change from each pops pref_rate
                else:
                    value_list.append(i_pop.baseline_params.deg_per_pref)  # if smooth distribution from bl to pref_rate
        avgs = [[]] * len(value_list)
        for idx, val in enumerate(value_list):
            if idx == 0:  # if first val then circle to end as cannot index 1-1
                x = len(value_list) - 1
                y = idx + 1
            elif idx == len(value_list) - 1:  # if last val then circle to start as cannot index end+1
                x = idx - 1
                y = 0
            else:  # middle indices pull vals to left and right of val
                x = idx - 1
                y = idx + 1
            if operation == 'mean':
                avgs[idx] = [np.mean([value_list[idx], value_list[x]]), np.mean([value_list[idx], value_list[y]])]
        return avgs

    def adjust_boundaries(self):
        # Adjust boundaries to smoothly switch between different sampling rates
        # boundaries = [my_asarray(i_pop.params.boundaries) for i_pop in self.neural_populations]
        boundaries = [arr(i_pop.params.boundaries) for i_pop in self.neural_populations]
        # If only 1 pref in a neural population, skip adjustment
        if any(i_pop.params.n_prefs <= 1 for i_pop in self.neural_populations):
            return boundaries
        # get average sampling frequency of adjacent orientations
        # adjuster_values = my_asarray(self.get_bounds_adjusters())
        adjuster_values = arr(self.get_bounds_adjusters())
        for idx, i_bounds in enumerate(boundaries):  # loop through each ori bounds
            i_adjusted_bounds = i_bounds + ((adjuster_values[idx] * [1, -1]) / 2)
            self.neural_populations[idx].params.boundaries = i_adjusted_bounds

    def stack_populations(self, sort_by_prefs=True):
        all_prefs = []
        all_prefs_idx = []
        all_tunings = []
        # all_params = init_cols(self.neural_populations[0].param_distribs.keys())  # for old model version
        for idx, i_pop in enumerate(self.neural_populations):
            all_prefs = np.hstack([all_prefs, i_pop.prefs])
            all_prefs_idx = np.hstack([all_prefs_idx, i_pop.prefs_idx])
            # for i_param in i_pop.param_distribs:
            #     all_params[i_param] = np.hstack([all_params[i_param], i_pop.param_distribs[i_param]])
            all_tunings.append(i_pop.tunings)
        # all_tunings = np.vstack([i for i in all_tunings])  # concat all appended tunings
        ### edits
        all_prefs = [_i for _i in all_prefs if _i is not None]
        all_prefs_idx = [_i for _i in all_prefs_idx if _i is not None]
        all_tunings = np.vstack([_i for _i in all_tunings if _i is not None])  # concat all appended tunings### edit
        ###
        # sort prefs_idx and tunings by indices of prefs in ascending order
        (sorted_all_prefs, sorted_all_prefs_idx, sorted_all_tunings) = sort_by_standard(all_prefs, all_prefs_idx, all_tunings)
        # for i_param in all_params:
        #     (_, all_params[i_param]) = sort_by_standard(all_prefs, all_params[i_param])
        # assign
        self.all_prefs = sorted_all_prefs
        self.all_prefs_idx = sorted_all_prefs_idx
        self.all_tunings = sorted_all_tunings
        # # PLOT tunings #
        # fig, ax = plt.subplots(figsize=(4, 2), dpi=300)
        # for idx, i_pop in enumerate(self.neural_populations):
        #     if any(n == i_pop.name for n in ['45', '135']):
        #         c = [0, .05, .3]  # obliques
        #         noize = 0.4
        #     else:
        #         c = [.8, .2, 0]  # cardinals
        #         noize = 0.1
        #     noise_for_tunings = poissonlike_noise(i_pop.tunings, noize)
        #     noisy_tuns = i_pop.tunings + noise_for_tunings
        #     noisy_tuns[noisy_tuns <= 0] = 0  # remove negative firing rates
        #     # limit firing rates to rmax
        #     rmax_distrib = (np.ones(noisy_tuns.shape).T * 60).T  # match shape to allow indexing
        #     noisy_tuns[noisy_tuns > rmax_distrib] = rmax_distrib[rmax_distrib < noisy_tuns]  # set max to each nrns rmax
        #     # plt.plot(self.orispace.tiling, noisy_tuns.T, c=c)  # plots noisy tuning curves
        #     plt.plot(self.orispace.tiling, i_pop.tunings.T, c=c)  # plots tuned response
        # ax.set_xlabel('Orientation', fontweight='bold', fontsize=12)
        # ax.set_ylabel('Firing Rate\n(spikes / sec)', fontweight='bold', fontsize=12)
        # ax.set_xlim(-90, 90)
        # ax.set_xticks([-90, -45, 0, 45, 90])
        # ax.set_xticklabels([f"{t}$^\circ$" for t in ax.get_xticks()])
        # # ax.get_yaxis().set_visible(False)  # remove y-axis labels
        # fig.tight_layout()
        # plt.close()

    def rolling_window(self, vector=None, window=None):
        # create rolling prefs window (where each col = pref) across any range of space
        # chosen prefs window centres on stim_ori to ensure circular tuning
        if vector is None:
            vector = self.all_prefs
        if window is None:
            # >= 0 to < 180 consists of all possible orientations in standard size feature space
            # Using 180 causes ML and PV estimates (esp low contrast) to be biased toward the centre
            # window = np.where(self.all_prefs < self.all_prefs[0] + 180)[0]  # len of window all that matters
            window = np.arange(0, 360)
        m = len(vector) - len(window)
        all_prefs_windows_idx = hankel(np.arange(0, m + 1), np.arange(m, len(vector)))
        all_prefs_windows = vector[all_prefs_windows_idx]
        self.all_prefs_windows_idx = all_prefs_windows_idx
        self.all_prefs_windows = all_prefs_windows  # cols = prefs
        return self.all_prefs_windows, self.all_prefs_windows_idx

    def gen_anisotropic_distribution(self, o_lim=None, c_lim=None):
        # NEW MODEL - all_prefs from a single distribution #
        g_params = self.global_params
        tiling = self.orispace.tiling
        n_cycles = 4  # cycles per 360 deg, therefore 90 deg period required 4 cycles
        # peaks=cardinal  troughs=obliques
        # Set distribution for params and prefs
        for i_dist in ['pref_distrib', 'param_distrib']:
            distrib = self.global_params[i_dist]
            if any(_i == distrib for _i in ['cos', 'vonmises']):
                if g_params[i_dist] == 'cos':
                    curve = generate_cosine(tiling, n_cycles=n_cycles)
                elif g_params == 'vonmises':
                    curve = generate_vonmises(tiling, pref=0, sigma=7.5, n_cycles=n_cycles, input_format='deg')
                    curve = normalise(curve, 0, 1)
                o_lim = 0  # proportion of cycle at which oblique cuts off (e.g., 0.5 = half-wave rectified)
                c_lim = 1
                curve[curve < o_lim] = o_lim  # obliques
                curve[curve > c_lim] = c_lim  # cardinals
            elif g_params[i_dist] == 'flat':
                curve = generate_cosine(tiling, n_cycles=n_cycles)
                curve[curve <= 0.5] = 0  # oblique
                curve[curve > 0.5] = 1  # cardinal
            self.anisotropy[i_dist] = curve
        z = 0

    def init_population_code(self):
        # NEW MODEL - all_prefs from a single distribution #
        # Define distribution curve for cardinal and oblique parameters
        self.gen_anisotropic_distribution()
        self.test_generate_prefs()  # sampled orientation preferences
        # Check sampling rates - difference between adjacent prefs
        # prefs = self.all_prefs
        # diffs = [abs(_i - prefs[_idx + 1]) for _idx, _i in enumerate(prefs) if _idx < len(prefs) - 1]
        # fig = plt.plot(prefs[:-1], diffs)
        # fig[0].axes.set(xlabel="Orientation Preference", xticks=np.linspace(-180, 180, 9),
        #                 ylabel="Spacing from adjacent pref [i+1]")
        # plt.close('all')
        #
        self.test_generate_params()  # parameters for each neuron sampled according to orientation preference
        self.test_generate_tunings()  # tuning curve for each pref
        if self.global_params['tuning_dist'] != 'von_mises':
            self.rolling_window()  # all possible windows of ori_prefs, to centre on stimulus ori

    def gen_tunings(self, stack=True, window=True):
        for idx, i_pop in enumerate(self.neural_populations):
            # PARAMS #
            self.neural_populations[idx].generate_params()
            # TUNINGS #
            self.neural_populations[idx].generate_tunings()
        if stack:
            self.stack_populations()  # Combine tunings and param_distribs from all populations into one
        if window:
            self.rolling_window()  # Rolling window of preferences to simulate wrapped tuning

    def gen_prefs_distrib(self, scale, size=100000):
        # NEW MODEL - all_prefs from a single distribution #
        # Anisotropic probability distribution
        p_dist = self.anisotropy['pref_distrib']
        p_dist = normalise(p_dist, 1, scale)  # Scale cosine curve
        # Sample orientation space according to cosine curve (probability distribution)
        dist = np.random.choice(self.orispace.tiling, size=size, p=p_dist / np.sum(p_dist))
        # dist = my_asarray(sorted(dist))
        dist = arr(sorted(dist))
        # Sample preferences from sampling distribution
        idcs = np.round(np.linspace(0, len(dist) - 1, self.global_params['n_samples'] + 1)).astype('int')
        prefs = dist[idcs]
        prefs = prefs[:-1]  # Remove last pref as repeated in orispace
        n_cardinal = np.sum((prefs % 90) < 22.5) + np.sum((prefs % 90) > 67.5)
        output = {'prefs': prefs, 'n_cardinal': n_cardinal,
                  'propn_cardinal': np.round(n_cardinal / (self.global_params['n_samples']), 3)}
        return output

    def test_generate_prefs(self):
        # NEW MODEL - all_prefs from a single distribution #
        # ORIENTATION PREFERENCE DISTRIBUTIONS #
        g_params = self.global_params
        n_cardinal = int(np.round(g_params['n_samples'] * g_params['propn_cardinal']))
        tiling = self.orispace.tiling
        _gen_prefs = True  # can be set to False if loading prefs (useful during optimisation)

        # Load prefs
        if g_params.get('save_prefs') is True:
            prefs_path = os.path.join(os.getcwd(), 'dump')
            if not os.path.exists(prefs_path):
                os.makedirs(prefs_path)
            prefs_filename = f"{g_params['pref_distrib']}_N{g_params['n_samples'] / 2}_pC{g_params['propn_cardinal']}"
            prefs_filepath = os.path.join(prefs_path, f"{prefs_filename}.pkl")
            if os.path.exists(prefs_filepath):  # if file has been made previously
                with open(prefs_filepath, 'rb') as file:
                    prefs_info = pickle.load(file)
                _gen_prefs = False

        if _gen_prefs is True:
            # Generate preferences using defined distribution
            if g_params['propn_cardinal'] == 0.5:
                chosen_scale = 1  # no need for minimisation routine as flat distribution
            else:
                # MINIMISE - scale curve to best reproduce the desired propn_cardinal
                if g_params['n_samples'] < 90:
                    print(f"WARNING\n"
                          f"--- Prefs generated from scaled distribution with low N"
                          f"--- May cause differences between the desired and actual propn_cardinal")
                bias = 50  # adding bias increases step sizes of minimisation routine (good for integer vals)
                x0 = arr([1 + bias])
                # x0 = my_asarray([1 + bias])
                # bounds = my_asarray([[_i + bias for _i in [0.05, 100]]])
                bounds = arr([[_i + bias for _i in [0.05, 100]]])
                min_fun = lambda scale: prefs_sampling_error(prefs_fun=self.gen_prefs_distrib, scale=scale, size=100000,
                                                             desired_n=n_cardinal, bias=bias, _print=False)
                result = minimize(min_fun, x0=x0, bounds=bounds, method='Nelder-Mead', options={"adaptive": True})
                chosen_scale = result.x[0] - bias

            # GENERATE PREFS #
            # Using chosen scale for cosine curve
            prefs_info = self.gen_prefs_distrib(scale=chosen_scale, size=2000000)
            # If actual propn_cardinal substantially deviates from desired, then print
            if np.round(prefs_info['propn_cardinal'], 1) != np.round(self.global_params['propn_cardinal'], 1):
                print(f"Desired propn_cardinal = {g_params['propn_cardinal']}"
                      f"\nActual propn_cardinal = {prefs_info['propn_cardinal']}")
            else:
                if g_params.get('save_prefs') is True:
                    with open(prefs_filename, 'wb') as file:
                        pickle.dump(prefs_info, file)

        # Assign to model
        (prefs, prefs_idx) = find_nearest(tiling, prefs_info['prefs'])
        self.all_prefs = prefs
        self.all_prefs_idx = prefs_idx
        return prefs

    def test_generate_params(self):
        # NEW MODEL - all_prefs from a single distribution #
        # PARAMETER DISTRIBUTIONS #
        # Generate parameters that change as a function of orientation, based on cardinal and oblique properties
        anisotropy = self.anisotropy
        param_distribs = {}
        for i_param in anisotropy['param_labels']:
            if i_param == 'n_prefs':
                continue  # self.gen_prefs()
            if isinstance(self.anisotropy['cardinal_params'][i_param], str):
                continue
            cardinal_val = anisotropy['cardinal_params'][i_param]
            oblique_val = anisotropy['oblique_params'][i_param]
            i_param_distrib = normalise(anisotropy['param_distrib'],
                                        val_start=oblique_val, val_end=cardinal_val  # from oblique to cardinal
                                        )
            i_param_distrib = i_param_distrib[self.all_prefs_idx]  # Index for each ori_pref
            param_distribs[i_param] = i_param_distrib
        self.param_distribs = param_distribs
        return param_distribs

    def test_generate_tunings(self):
        # NEW MODEL - all_prefs from a single distribution #
        g_params = self.global_params
        param_distribs = self.param_distribs
        tiling = self.orispace.tiling
        fwhm = param_distribs['fwhm']
        # Gaussian tuning
        if g_params['tuning_dist'] == 'gaussian':
            sigma = fwhm / 2.3548
            sigma = arr([sigma]).transpose()  # 1 val per ori_pref
            tunings = generate_gaussian(tiling, self.all_prefs, sigma)  # gaussian tuning for each pref
        # Von Mises tuning
        elif g_params['tuning_dist'] == 'von_mises':
            rad_fwhm = np.radians(fwhm)  # must be in radians
            kappa = np.log(0.5) / (np.cos(rad_fwhm) - 1)  # Swindale (1998)
            kappa = kappa[:, nax]  # 1 val per ori_pref
            tunings = generate_vonmises(tiling, self.all_prefs, kappa, input_format='deg')
        else:
            return None
        # Normalise so each tuning curve (row) has min=0 and max=1
        cols_idcs = np.argmax(tunings, axis=1)
        rows_idcs = np.arange(0, len(cols_idcs), 1)
        max_vals = tunings[rows_idcs, cols_idcs]
        tunings = tunings / arr([max_vals]).T
        # tunings = tunings / my_asarray([max_vals]).transpose()  # normalise to 1
        # Set maximum firing rate to rmax
        tunings = (tunings.T * param_distribs['rmax']).T  # normalise to rmax
        self.all_tunings = tunings
        if self.debug is True:
            plt.plot(tiling, tunings[::5].T)
            plt.yticks(np.linspace(tiling.min(), tiling.max(), 16))
            plt.grid()
        return tunings


    def get_params(self):
        joined_params = init_cols([p for p in Params().__dict__])
        for i_pop in self.neural_populations:
            i_params = i_pop.params.__dict__
            for p in i_params:
                if isinstance(i_params[p], float):
                    np.round(i_params[p], 5)
                joined_params[p].append(i_params[p])
        self.params = Params(joined_params)
        # if all(isinstance(val, float) for val in run_info[p]):
        #     run_info[p] = np.round(run_info[p], 3)
        return self.params

    def contrast_response(self, pop_resp, stim_contrast=100):
        # Scales population response according to contrast response function (CRF) - Naka-Rushton
        n = self.trial_param_distribs['crf_exponent']
        c50 = self.trial_param_distribs['crf_c50']
        crf_resp = pop_resp.T * (stim_contrast ** n / (stim_contrast ** n + c50 ** n))
        # Spontaneous firing proportional to rmax (independent from contrast scaling)
        # r0 = self.trial_param_distribs['spont'] * self.trial_param_distribs['rmax']
        r0 = self.trial_param_distribs['spont']  # set spont firing rate independently from rmax
        crf_resp = crf_resp + r0
        crf_resp = np.transpose(crf_resp)
        return crf_resp

    def noisy_response(self, pop_resp):
        # Noise added onto tuned response
        if self.global_params['noise_type'] == 'poisson':
            noise = poissonlike_noise(pop_resp, self.trial_param_distribs['noise_scale'])
        else:
            noise = np.random.normal(size=len(pop_resp))
        noisy_resp = pop_resp + noise
        noisy_resp[noisy_resp <= 0] = 0  # remove negative firing rates
        return noisy_resp

    def gen_response(self, stim_ori_idx, stim_contrast, n_reps=1):
        # NEW MODEL - all_prefs from a single distribution #
        # window_tunings = self.all_tunings[self.trial_tunings_idx, :]  # Tunings centred on a given stim_ori
        resp_tuned = self.trial_tunings[:, stim_ori_idx]  # Tuned response to stim_ori
        resp_tuned = np.transpose(resp_tuned * np.ones([n_reps, 1]))  # Tuned response across trials / repeats
        resp_tuned = self.contrast_response(resp_tuned, stim_contrast)
        resp_noisy = self.noisy_response(resp_tuned)
        self.trial_resp_tuned = resp_tuned
        self.trial_resp = resp_noisy
        self.resp_noisy = resp_noisy
        return resp_noisy

    def gen_population_response(self, stim_ori_idx, stim_contrast, trial_tunings_idx=None, n_reps=1):
        # modify tunings according to contrast and spont then gen noisy resp
        if trial_tunings_idx is None:
            trial_tunings_idx = self.trial_tunings_idx
        all_resp_tuned = []
        all_resp_noisy = []
        all_noise = []
        all_pop_prefs = []
        for i_pop in self.neural_populations:
            all_pop_prefs = np.hstack([all_pop_prefs, i_pop.prefs])  # append all_prefs
            # resp_tuned = i_pop.get_attrib('tunings')  # tuned_resp of one population
            resp_tuned = i_pop.tunings
            if i_pop.tunings is None:  ### edits
                continue
            resp_tuned = resp_tuned[:, stim_ori_idx]  # stim response of this population
            resp_tuned = np.transpose(resp_tuned * np.ones([n_reps, 1]))  # repeat for n_level_reps (noise added after)
            # scale for contrast before noise to prevent interaction (stim contrast independent of internal noise)
            resp_noisy = i_pop.contrast_response(resp_tuned, stim_contrast)
            # add spont firing before poisson noise to enable fluctuations in internal noise
            resp_noisy = i_pop.spontaneous_firing(resp_noisy)  # independent of contrast scaling and other noise
            resp_noisy, noise = i_pop.noisy_response(resp_noisy)  # noisy_response across n_level_reps
            all_noise.append(noise)
            all_resp_tuned.append(resp_tuned)
            all_resp_noisy.append(resp_noisy)
        all_noise = np.vstack([i for i in all_noise])  # get array of noise (for entire PopCode)
        all_resp_tuned = np.vstack([i for i in all_resp_tuned])  # stack resp for each neural pop
        all_resp_noisy = np.vstack([i for i in all_resp_noisy])  # stack resp for each neural pop
        (_, all_resp_tuned) = sort_by_standard(all_pop_prefs, all_resp_tuned)  # both in asc. prefs order
        (_, all_resp_noisy) = sort_by_standard(all_pop_prefs, all_resp_noisy)  # both in asc. prefs order
        self.all_noise = all_noise  # prefs window has not been applied
        # all_tunings = self.get_attrib('all_tunings')
        all_tunings = self.all_tunings
        self.trial_tunings = all_tunings[trial_tunings_idx, :]
        # # # # # PLOT 180 deg section of tunings #
        # plt.figure(figsize=(4, 2), dpi=300)
        # plt.plot(self.orispace.tiling, self.trial_tunings.T)
        self.resp_tuned = all_resp_tuned[trial_tunings_idx, :]  # apply prefs window
        self.resp_noisy = all_resp_noisy[trial_tunings_idx, :]
        # # # # # PLOT 180 deg section of responses #
        # fig, ax = plt.subplots(figsize=(4, 2), dpi=300)
        # for idx, i_pop in enumerate(self.neural_populations):
        #     if any(n == i_pop.name for n in ['45', '135']):
        #         c = [0, .05, .3]  # obliques
        #     else:
        #         c = [.8, .2, 0]  # cardinals
        #     plt.plot(self.orispace.tiling, i_pop.population_response.T, c=c)
        # ax.set_xlabel('Orientation', fontweight='bold', fontsize=12)
        # ax.set_ylabel('Firing Rate\n(spikes / sec)', fontweight='bold', fontsize=12)
        # ax.set_xlim(-90, 90)
        # ax.set_xticks([-90, -45, 0, 45, 90])
        # ax.set_xticklabels([f"{t}$^\circ$" for t in ax.get_xticks()])
        # # ax.get_yaxis().set_visible(False)  # remove y-axis labels
        # fig.tight_layout()
        # plt.close()
        # plt.figure(figsize=(4, 2), dpi=300)
        # plt.plot(self.orispace.tiling, self.resp_noisy)
        # #
        return self.resp_noisy, self.resp_tuned, self.all_noise

    def shift_prefs(self, stim_ori):
        # Only used with wrapped Gaussians but may be invalid anyway - consider deleting
        all_windows = self.all_prefs_windows
        all_prefs = arr(self.all_prefs)
        # all_prefs = my_asarray(self.all_prefs)
        # get central value of every possible prefs window
        centre_vals = np.mean([all_windows.min(1), all_windows.max(1)], 0)
        # get and use row_idx (window) where centre is closest to stim_ori
        window_idx = abs(centre_vals - stim_ori).argmin()
        window_vals = all_windows[window_idx, :]
        # get range of idxs so we can index tunings array
        min_idx = abs(all_prefs - window_vals.min()).argmin()  # start of window
        max_idx = abs(all_prefs - window_vals.max()).argmin()  # end of window
        window_idxs = np.arange(min_idx, max_idx + 1)  # all idxs within window
        if len(window_vals) != len(window_idxs):
            print('unequal len of trial_prefs and trial_prefs_idx, consider increasing resolution of space')
        # Assign trial model parameters (within window)
        self.trial_tunings_idx = window_idxs  # Tunings idxs for stim_ori
        self.trial_tunings = self.all_tunings[window_idxs, :]  # Tuning curves within window
        if self.pref_sampling == 'whole':
            trial_param_distribs = {}  # Parameter distributions for this trial
            for i_param in self.param_distribs:
                trial_param_distribs[i_param] = self.param_distribs[i_param][window_idxs]  # Params within window
            self.trial_param_distribs = trial_param_distribs
        # Assign preferences and idcs
        self.trial_prefs, self.trial_prefs_idx = find_nearest(self.orispace.tiling, window_vals)
        return self.trial_tunings_idx

    def gen_stim_response(self, stim_ori, stim_con=100, trial_decoder=None, n_reps=1):
        # # # Generate a population response for a given orientation value # # #
        g_params = self.global_params
        tiling = self.orispace.tiling
        _, stim_ori_idx = find_nearest(tiling, stim_ori)  # idx of stim_ori in orientation space
        # update attributes
        self.trial_stim_ori = stim_ori
        self.trial_stim_ori_idx = stim_ori_idx
        self.trial_stim_con = stim_con
        # Save used window of prefs and tunings for later decoding
        if g_params['tuning_dist'] != 'von_mises':  # No need for moving prefs window
            self.shift_prefs(stim_ori)  # Moving prefs window
        else:
            self.trial_tunings = self.all_tunings
            self.trial_param_distribs = self.param_distribs
            self.trial_prefs = self.all_prefs
        # resp dependent on contrast, cut within prefs window, for n repetitions
        if self.pref_sampling == 'indiv':
            resp_noisy, resp_tuned, noise = self.gen_population_response(stim_ori_idx, stim_con, n_reps=n_reps)
        elif self.pref_sampling == 'whole':  # response for whole population at once
            resp_noisy = self.gen_response(stim_ori_idx, stim_con, n_reps=n_reps)
        if self.debug is True:
            # Checks prefs window has max tuned_resp in its centre
            centre = debugger.max_resp_in_centre_of_window(self, _print=self.debug)
            if centre is False:
                z = 0
            self.decode_response(trial_decoder, stim_ori, n_reps=n_reps, gen_resp=False)
        return resp_noisy

    def pooled_population_response(self, stim_ori, stim_con, n_reps):
        # fun: return summed population activity for each repetition
        resp_noisy = self.gen_stim_response(stim_ori=stim_ori, stim_con=stim_con, trial_decoder=None, n_reps=n_reps)
        pooled_resp = None
        if self.pooling['method'] == 'summation':  # magnitude = summed activity
            pooled_resp = np.sum(resp_noisy, 0)
        elif self.pooling['method'] == 'vector_magnitude':  # weighted magnitude of response
            pool_exponent = self.pooling['exponent']
            weighted_resp = []
            for i_nrn in range(len(resp_noisy)):
                # loop through each nrn_resp and ** crf_exponent
                weighted_resp.append(resp_noisy[i_nrn, :] ** pool_exponent)
            weighted_resp = np.vstack([i_nrn_resp for i_nrn_resp in weighted_resp])
            pooled_resp = np.sum(weighted_resp, 0) ** (1 / pool_exponent)  # sum and return to og units
            if self.debug:
                test_summation = np.sum(resp_noisy, 0)
                debugger.weighted_resp_plots(resp_noisy, weighted_resp)
        elif self.pooling['method'] == 'distance':
            # scale response pooling based on distance from centre of prefs distribution
            orispace = self.orispace.tiling
            min_tile = np.min(self.trial_prefs)
            max_tile = np.max(self.trial_prefs)
            tiling_sample = orispace[(orispace >= min_tile) & (orispace <= max_tile)]
            # weights = generate_cosine(x=tiling_sample)
            pool_sigma = self.get_pooling_params(ori_std=stim_ori, label='extent')
            weights = generate_gaussian(x=tiling_sample, mu=np.mean([min_tile, max_tile]), sigma=pool_sigma)  # s=10 is ~45
            # get idxs of prefs within tiling_sample
            _, sample_prefs_idx = find_nearest(tiling_sample, self.trial_prefs)
            # use idxs of prefs within tiling_sample to pull appropriate weights
            weights = weights[sample_prefs_idx] / np.max(weights[sample_prefs_idx])
            if self.debug is True:
                plt.subplots()
                plt.plot(self.trial_prefs, weights)
                plt.close()
            # apply weights to population rdistance_weighted response
            # pooled_resp = my_asarray([weights]).T * resp_noisy
            pooled_resp = arr([weights]).T * resp_noisy
            if self.debug is True:
                plt.subplots()
                plt.plot(self.trial_prefs[:], pooled_resp[:, ::100])
                plt.title(f"{stim_con} % contrast")
                plt.close()
            pooled_resp = np.sum(pooled_resp, 0)
        return pooled_resp

    def detection(self, response_decoder='summation', stim_ori=None, stim_con=None, n_reps=None):
        # return pooled population response for detection task #
        if all(i or i == 0 for i in [stim_ori, stim_con, n_reps]):
            self.gen_stim_response(stim_ori, stim_con, None, n_reps)
        if response_decoder == 'peak':
            decoded_val = np.max(self.resp_noisy, 0)
        elif response_decoder == 'sum':
            decoded_val = np.sum(self.resp_noisy, 0)  # sum across neurons for each trial
        elif response_decoder == 'mag':  # weighted by response magnitude
            decoded_val = self.vector_mag()
        elif response_decoder == 'distance':  # weighted by distance from stim
            decoded_val = self.distance_sum()
        elif response_decoder == 'distance_v_mag':
            decoded_val = self.distance_vectormag()
        elif response_decoder == 'PV':
            decoded_val = self.popvector(_detection=True)  # estimates actual contrast [?]
        else:
            print('No detection decoder chosen')
            decoded_val = None
        return decoded_val

    def vector_mag(self, pop_resp=None, q=None):
        # weighted sum of population response (Quick, 1974; also see Goris 2013)
        if pop_resp is None:
            pop_resp = self.resp_noisy
        if q is None:
            # q = self.get_pooling_params(ori_std=self.trial_stim_ori, label='exponent')
            # q = 6  # q was 6 in thesis which is fine (see self.debug below)
            q = self.global_params['pooling_q']  # 1st lot of paper simulations (should I add this to conditions... yes...)
        vector_mag_val = np.sum(pop_resp ** q, 0) ** (1 / q)  # sum and return to og units
        if self.debug:
            if self.trial_stim_con > 10:  # test when contrast is higher
                q1 = 1  # no weighting
                q2 = 6
                # This shows how the weighting occurs across the population (averaged across trials)
                fig, ax = plt.subplots()
                a = np.sum(pop_resp ** q1, 1)
                b = np.sum(pop_resp ** q2, 1)
                ax.plot(normalise(a, 0, 1), c='k')
                ax.plot(normalise(b, 0, 1), linestyle=":")
                plt.show()
                plt.close()
        return vector_mag_val

    def distance_sum(self, pop_resp=None, pool_sigma=None):
        # scale response pooling based on distance from stim_ori
        if pop_resp is None:
            pop_resp = self.resp_noisy
        if pool_sigma is None:
            # pool_sigma = self.get_pooling_params(ori_std=self.trial_stim_ori, label='exponent')
            pool_sigma = 10
        weights = generate_gaussian(x=self.trial_prefs, mu=self.trial_stim_ori, sigma=pool_sigma)
        weights = weights / np.max(weights)  # 0 to 1
        # pooled_val = my_asarray([weights]).T * pop_resp  # apply weights to population distance_weighted response
        pooled_val = arr([weights]).T * pop_resp  # apply weights to population distance_weighted response
        distance_val = np.sum(pooled_val, 0)
        return distance_val

    def distance_vectormag(self):
        # scale response pooling based on distance from stim_ori
        pop_resp = self.resp_noisy
        pool_sigma = self.get_pooling_params(ori_std=self.trial_stim_ori, label='exponent')
        weights = generate_gaussian(x=self.trial_prefs, mu=self.trial_stim_ori, sigma=pool_sigma)
        weights = weights / np.max(weights)  # 0 to 1
        # w_pop_resp = my_asarray([weights]).T * pop_resp  # apply weights to population distance_weighted response
        w_pop_resp = arr([weights]).T * pop_resp  # apply weights to population distance_weighted response
        # VECTOR MAG POOLING
        q = self.get_pooling_params(ori_std=self.trial_stim_ori, label='exponent')
        distance_vmag = np.sum(w_pop_resp ** q, 0) ** (1 / q)  # sum and return to og units
        return distance_vmag

    def decode_response(self, response_decoder='WTA', stim_ori=None, stim_con=None, n_reps=None, gen_resp=True,
                        save_decoded=False, load_decoded=False):
        # If using stored decoded estimates
        if load_decoded is True and self.saved_response.get(stim_ori) is not None:
            decoded_val = self.saved_response[stim_ori]
            decoded_val = random.sample(list(decoded_val), n_reps)
            return decoded_val
        else:
            if all(i or i == 0 for i in [stim_ori, stim_con, n_reps]):
                if gen_resp is True:
                    self.gen_stim_response(stim_ori, stim_con, response_decoder, n_reps)
            decoded_val = None
            if response_decoder == 'WTA':
                decoded_val = self.wta()
            elif response_decoder == 'PV':
                decoded_val = self.popvector()
            elif response_decoder == 'ML':
                decoded_val = self.maxlikelihood()
            # plt.hist(decoded_val, alpha=0.8)
            # Add tiny amount of noise to separate shared estimates
            decoded_val = decoded_val + np.random.uniform(-0.0001, 0.0001, size=len(decoded_val))
            self.decoded = decoded_val
        # If storing decoded estimates for later use
        if save_decoded is True:
            self.saved_response[stim_ori] = self.decoded
        return decoded_val

    def cramer_rao(self, stim_ori, stim_con=100, n_reps=1):
        self.gen_stim_response(stim_ori, stim_con, trial_decoder=None, n_reps=n_reps)
        resp_noisy = self.resp_noisy  # each nrn noisy_resp across trials
        resp_tuned = self.trial_tunings  # each nrn tuned_resp across orispace
        # Define the parameter and likelihood function
        resp_tuned = resp_tuned + (1 ** -10)  # removes any vals=zero
        log_likelihood = resp_noisy.T @ np.log(resp_tuned)  # outputs: [trials x orispace]
        # # # ... in this matrix multiplication, summed signal (across neurons) each point of orispace
        # Calculate Fisher Information Matrix
        score_function = np.gradient(log_likelihood, axis=1)  # use axis 1 as only interested in each trial
        fisher_info = -np.gradient(score_function, axis=1)
        # Calculate Cramer-Rao bound (CRB) for given stim_ori by inverting Fisher Information Matrix
        crb = fisher_info ** -1
        crb_idx = crb[:, self.trial_stim_ori_idx]
        crb_ave = np.mean(crb_idx)
        return crb_ave

    def wta(self, pop_resp=None, trial_prefs=None):
        # Pref of the neurons which produced the strongest response each trial
        # pop_resp: 2D neuron(rows) x trial(cols) for resp(vals)
        # trial_prefs: 1D prefs(vals)
        if pop_resp is None:
            pop_resp = self.resp_noisy
        if trial_prefs is None:
            trial_prefs = self.trial_prefs
        trial_max = pop_resp.max(0)  # max response for each trial
        ismax = (trial_max == pop_resp).astype('int')  # bool 01 matrix of max for each trial
        ismax = ismax + np.random.uniform(-0.1, 0.1, size=ismax.shape)  # add random noise to distinguish shared vals
        trial_pref_idx = ismax.argmax(0)
        wta_est = trial_prefs[trial_pref_idx]  # get idx of pref with max response
        return wta_est

    def popvector(self, weights=None, pop_prefs=None, _detection=False):
        # pop_resp: 2D neuron(rows) x trial(cols) for resp(vals)
        # trial_prefs: 1D prefs(vals)
        if weights is None:
            weights = self.resp_noisy
            # weights could also be set as self.resp_tuned
        if pop_prefs is None:
            pop_prefs = self.trial_prefs
        pop_prefs = np.radians(pop_prefs)
        # Account for 180 deg of circular space
        if self.orispace.range == 180:
            pop_prefs = pop_prefs * 2  # convert to circular space (i.e., 360 deg)

        # Circular population vector
        # 1. Response of each neurons represented in the complex plane
        response_vectors = weights * np.exp(1j * pop_prefs[:, nax])
        # 2. Sum all vectors
        resultant_vector = np.sum(response_vectors, 0)
        # 3. Estimate angle of average vector in complex plane
        pv_est = np.angle(resultant_vector, deg=True)

        if self.orispace.range == 90:  # Valid to do as long as orispace is +- 0
            pv_est = pv_est / 2

        return pv_est


    def maxlikelihood(self, pop_resp=None, pop_tunings=None, tiling=None):
        if pop_resp is None:
            pop_resp = self.resp_noisy
        if pop_tunings is None:
            pop_tunings = self.trial_tunings
        if tiling is None:
            tiling = self.orispace.tiling
            # tiling = self.orispace.get_attrib('tiling')

        pop_tunings = np.clip(pop_tunings, 1e-10, None)  # Removes any zero vals (for log)
        log_likelihood = pop_resp.T @ np.log(pop_tunings)  # matrix multiplication; outputs: size=[trials, tiling]
        # Add random noise to distinguish shared vals
        log_likelihood = log_likelihood + np.random.uniform(-0.0001, 0.0001, size=log_likelihood.shape)
        max_likelihood_idx = log_likelihood.argmax(1)  # gives idx in feature space of ML est  *** .argmax()
        maxlikelihood_est = tiling[max_likelihood_idx]

        return maxlikelihood_est

    def gen_template_ori(self, target_ori, n_reps):
        # Generate the decoded template / internal representation of the target orientation
        template = self.decoded_template[target_ori]  # template vals for this orientation
        set_random_seed()
        np.random.shuffle(template)
        template_vals = random.sample(list(template), n_reps)
        return template_vals

    def get_template_noise(self, ori_std, nearest=True):
        #  return internal_std info from nearest pop to input ori
        if nearest is True:
            # search to find nearest match
            ori_std = np.degrees(np.radians(ori_std) % np.pi)  # converts to standard orientation space
            popidx = np.argmin([abs(ori_std - ipop.params.std_ori) for ipop in self.neural_populations])  # get popn idx
            template_noise = self.neural_populations[popidx].params.template_noise
        else:
            # search to find exact match
            try:
                popidx = [idx_ for idx_, pop in enumerate(self.neural_populations) if pop.params.std_ori == ori_std][0]
            except:
                template_noise = None
            else:
                template_noise = self.neural_populations[popidx].params.template_noise
        return template_noise

    def get_pooling_params(self, ori_std, label=None):
        # find nearest base ori to stim_ori and pull relevant params
        ori_std = ori_std % 180  # accounts for any extreme vals
        popidx = np.argmin([abs(ori_std - ipop.params.std_ori) for ipop in self.neural_populations])  # get popn idx
        pooling_params = {'exponent': self.neural_populations[popidx].params.pooling_exponent,
                          'extent': self.neural_populations[popidx].params.pooling_extent
        }
        if label is None:
            return pooling_params
        else:
            return pooling_params[label]
