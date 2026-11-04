import numpy as np
from numpy import asarray as arr, newaxis as nax
from scipy.linalg import hankel
from scipy.optimize import minimize
import random

from scipy.special import logsumexp

from .general_functions import find_nearest, generate_gaussian, generate_cosine, \
    generate_vonmises, convert_to_circular_space, \
    normalise, poissonlike_noise, generate_wrapped_gaussian
from components.run import set_random_seed
import matplotlib.pyplot as plt


def prefs_sampling_error(prefs_fun, scale, desired_n, **kwargs):
    """
    For a given scale, calculate error between the desired and actual n_cardinal
    """
    scale = np.exp(scale)  # Works with negative values
    output = prefs_fun(scale=scale)
    n = output['n_cardinal']
    error = abs(n - desired_n)
    if kwargs.get('_print') is True:
        print(f"\nscale = {scale}"
              f"\nerror = {error}")
    return error


def gen_population_code(cardinal_params, oblique_params, ori_space, pref_sampling='whole', **kwargs):
    # Params defined for the whole populations (default)
    global_params = {
        'n_prefs': cardinal_params.n_prefs + oblique_params.n_prefs,
        'propn_cardinal': cardinal_params.n_prefs / (cardinal_params.n_prefs + oblique_params.n_prefs),
        'noise_type': 'poisson',
        'pref_distrib': 'cos',
        'param_distrib': 'cos',
        'tuning_type': 'von_mises',
        'pooling_q': cardinal_params.pooling_q,
    }
    if kwargs.get('global_params') is not None:  # update default global_params
        for _key in kwargs['global_params']:
            global_params[_key] = kwargs['global_params'][_key]

    anisotropy_info = {'param_labels': ['n_prefs', 'fwhm', 'pref_distrib', 'param_distrib', 'rmax', 'spont',
                                        'noise_scale', 'crf_exponent', 'crf_c50'],
                       'cardinal_params': cardinal_params.__dict__, 'oblique_params': oblique_params.__dict__}
    population_code = PopulationCode(ori_space=ori_space, anisotropy=anisotropy_info, debug=kwargs.get('debug', False),
                                     global_params=global_params, pref_sampling='whole')
    population_code.init_population_code()

    return population_code


class Params:
    def __init__(self, dicty=None, fwhm=None, sigma=None, kappa=None, deg_per_pref=None, n_prefs=None,
                 pref_distrib='flat',
                 param_distrib='flat', pref_range=None, rmax=None, spont=None, crf_exponent=None, crf_c50=None,
                 pooling=None, noise_scale=1, noise_type='poisson', correlated_noise=None,
                 baseline_params=None, pooling_exponent=None, pooling_extent=None, run_n_prefs=None, run_density=None,
                 run_sampling_rate=None, boundaries=None, std_ori=None, pooling_q=None,
                 ):
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
            self.n_prefs = n_prefs
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

        # Calculate sigma from FWHM
        if self.fwhm is not None:
            if self.sigma is None:
                self.sigma = self.fwhm / (2 * np.sqrt(2 * np.log(2)))  # s = fwhm / 2.3548
            if self.kappa is None:
                h = self.fwhm / 2
                self.kappa = np.log(0.5) / (np.cos(2 * h) - 1)  # Swindale (1998)


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


class PopulationCode:
    def __init__(self, ori_space, anisotropy=None, name=None, debug=True, **kwargs):
        self.pref_sampling = kwargs.get('pref_sampling')  # whether old version used (individual neural populations)
        self.params_co = None
        self.param_distribs = {}
        self.params = None  # callable with .get_params()
        self.debug = debug  # if True then error checks will occur
        self.ori_space = ori_space

        self.global_params = kwargs.get('global_params', None)
        if self.global_params is not None:
            self.global_params['n_samples'] = int(
                self.global_params['n_prefs'] * (self.ori_space.range / 180)
            )  # whole range (not per 180)

        # Anisotropic parameters
        self.anisotropy = anisotropy
        # the following will be appended in methods of class
        self.tunings = []
        self._tuning_normalisation = kwargs.get('_tuning_normalisation', 'peak')
        self.prefs = []
        self.prefs_idx = []
        self.stim_ori = None
        self.stim_ori_idx = None
        self.stim_con = None
        self.resp_tuned = None
        self.resp_noisy = None
        self.noise_type = 'poisson'
        self.decoded = None
        if name is None:
            name = 'populationcode'
        self.name = name
        self.saved_response = {}  # saves decoded vals in dict key=stim_ori

        self.window_idcs = None

        self.prior_ori = None

        self._corr_noise_cache_key = None
        self._corr_noise_sqrt = None
        self._corr_noise_cholesky = None

    def rolling_window(self, vector=None, window=None):
        # create rolling prefs window (where each col = pref) across any range of space
        # chosen prefs window centres on stim_ori to ensure circular tuning
        if vector is None:
            vector = self.prefs
        if window is None:
            # >= 0 to < 180 consists of all possible orientations in standard size feature space
            # Using 180 causes ML and PV estimates (esp low contrast) to be biased toward the centre
            # window = np.where(self.prefs < self.prefs[0] + 180)[0]  # len of window all that matters
            window = np.arange(0, 360)
        m = len(vector) - len(window)
        all_prefs_windows_idx = hankel(np.arange(0, m + 1), np.arange(m, len(vector)))
        all_prefs_windows = vector[all_prefs_windows_idx]
        self.all_prefs_windows_idx = all_prefs_windows_idx
        self.all_prefs_windows = all_prefs_windows  # cols = prefs
        return self.all_prefs_windows, self.all_prefs_windows_idx

    def gen_anisotropic_distribution(self, o_lim=None, c_lim=None):
        g_params = self.global_params
        tiling = self.ori_space.tiling
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
        # Define distribution curve for cardinal and oblique parameters
        self.gen_anisotropic_distribution()
        self.generate_prefs()  # sampled orientation preferences

        self.generate_params()  # parameters for each neuron sampled according to orientation preference
        self.generate_tunings()  # tuning curve for each pref

        # self.plot_tunings()

    def gen_prefs_distrib(self, scale, _fast=False):
        # Scale determines how frequently different regions are sampled from the curve
        if scale > 1:
            p_dist = normalise(self.anisotropy['pref_distrib'], 1, scale)
        else:
            p_dist = normalise(self.anisotropy['pref_distrib'], 1 / scale, 1)

        p_dist /= np.sum(p_dist)  # Probabilities sum to 1

        # Compute cumulative distribution function (CDF)
        cdf = np.cumsum(p_dist)

        # Inverse transform sampling: map uniform samples through inverse CDF
        n_samples = self.global_params['n_samples']
        uniform_samples = np.linspace(0, 1, n_samples, endpoint=False)
        idcs = np.searchsorted(cdf, uniform_samples)

        prefs = self.ori_space.tiling[idcs]

        # Fast mode only needs the cardinal count for optimisaiton
        if _fast:
            prefs_to_count = prefs[(prefs >= -90) & (prefs < 90)]
            n_cardinal = np.sum(
                (prefs_to_count % 90 < 22.5) |
                (prefs_to_count % 90 > 67.5)
            )
            return {'n_cardinal': n_cardinal}

        # Corresponding orientations
        prefs.sort()

        # Count cardinal orientations (within 22.5° of 0°, 90°, 180°, ...)
        prefs_to_count = prefs[(prefs >= -90) & (prefs < 90)]
        n_cardinal = np.sum((prefs_to_count % 90 < 22.5) | (prefs_to_count % 90 > 67.5))
        n_oblique = len(prefs_to_count) - n_cardinal

        propn_cardinal = np.round(n_cardinal / len(prefs_to_count), 3)

        output = {'prefs': prefs, 'n_cardinal': n_cardinal, 'n_oblique': n_oblique,
                  'propn_cardinal': propn_cardinal}

        return output

    def generate_prefs(self):
        # NEW MODEL - prefs from a single distribution #
        # ORIENTATION PREFERENCE DISTRIBUTIONS #
        g_params = self.global_params
        n_cardinal = self.anisotropy['cardinal_params']['n_prefs']
        tiling = self.ori_space.tiling

        # Generate preferences using defined distribution
        if np.round(g_params['propn_cardinal'], 2) == 0.50:
            chosen_scale = 1  # Flat = no need for minimisation routine as flat distribution
        else:
            # MINIMISE - scale curve to best reproduce the desired propn_cardinal
            if g_params['n_samples'] < 60:
                print(f"WARNING\n"
                      f"--- Prefs generated from scaled distribution with low N"
                      f"--- May cause differences between the desired and actual propn_cardinal")

            # bias = 5  # adding bias increases step sizes of minimisation routine (good for integer vals)
            # x0 = arr([3 + bias])

            bias = 50
            x0 = arr([0 + bias])  # 0 = flat (positive = cardinal bias, negative = oblique bias)

            min_fun = lambda scale: prefs_sampling_error(prefs_fun=self.gen_prefs_distrib, scale=scale - bias,
                                                         desired_n=n_cardinal, bias=bias, _print=False,
                                                         _fast=True)

            result = minimize(min_fun, x0=x0, method='Nelder-Mead', options={'maxiter': 200})

            chosen_scale = np.exp(result.x[0] - bias)

        # GENERATE PREFS #
        # Using chosen scale for cosine curve
        prefs_info = self.gen_prefs_distrib(scale=chosen_scale)

        # If actual propn_cardinal substantially deviates from desired, then print
        if abs(g_params['propn_cardinal'] - prefs_info['propn_cardinal']) >= 0.015:
            print(f"Desired propn_cardinal = {g_params['propn_cardinal']:.2f}"
                  f"\nActual propn_cardinal = {prefs_info['propn_cardinal']:.2f}")

        # Assign to model
        (prefs, prefs_idx) = find_nearest(tiling, prefs_info['prefs'])
        self.prefs = prefs
        self.prefs_idx = prefs_idx
        return prefs

    def generate_params(self):
        # NEW MODEL - prefs from a single distribution #
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
            i_param_distrib = i_param_distrib[self.prefs_idx]  # Index for each ori_pref
            param_distribs[i_param] = i_param_distrib

        self.param_distribs = param_distribs

        return param_distribs

    def generate_tunings(self):
        # Generate tuning curves from the selected distribution.
        g_params = self.global_params
        param_distribs = self.param_distribs
        tiling = self.ori_space.tiling
        fwhm = param_distribs['fwhm']

        if g_params['tuning_type'] == 'gaussian':
            sigma = (fwhm / 2.3548)[:, None]
            tunings = generate_gaussian(tiling, self.prefs, sigma)

        elif g_params['tuning_type'] == 'gaussian_wrapped':
            sigma = (fwhm / 2.3548)[:, None]
            tunings = generate_wrapped_gaussian(
                tiling, self.prefs, sigma, period=180, wrap_range=8
            )

        elif g_params['tuning_type'] == 'von_mises':
            rad_fwhm = np.radians(fwhm)
            kappa = np.log(0.5) / (np.cos(rad_fwhm) - 1)
            kappa = kappa[:, None]

            # if any(fwhm > 70):
            #     print(f"WARNING"
            #           f"\nvon Mises curve doesn't reach zero with fwhm={fwhm.max():.0f}"
            #           f"\nAfter normalising between 0 and 1, bandwidth may change")

            tunings = generate_vonmises(
                tiling, self.prefs, kappa, input_format='deg'
            )

        else:
            return None

        if self.debug is True:
            # Check the raw tuning-curve floor before min-max normalisation.
            # Raise an issue if any curve has a minimum significantly above zero.
            min_frac = tunings.min(axis=1) / tunings.max(axis=1)
            threshold = 0.05
            problematic = min_frac > threshold
            if np.any(problematic):
                print(
                    f"Issue: {problematic.sum()}/{len(problematic)} tuning curves "
                    f"have a minimum > {threshold:.0%} of their peak "
                    f"(maximum floor = {min_frac.max():.3f})."
                )

        if self._tuning_normalisation == 'peak':
            # Scale each curve so its peak equals its target maximum firing rate.
            curve_max = tunings.max(axis=1, keepdims=True)
            tunings = tunings / curve_max * param_distribs['rmax'][:, None]

        elif self._tuning_normalisation == 'minmax':
            # Normalise each curve to [0, 1], then scale its peak to rmax.
            curve_min = tunings.min(axis=1, keepdims=True)
            curve_max = tunings.max(axis=1, keepdims=True)
            tunings = (tunings - curve_min) / (curve_max - curve_min)
            tunings = tunings * param_distribs['rmax'][:, None]

        else:
            raise ValueError("_tuning_normalisation must be 'peak' or 'minmax'")

        self.tunings = tunings
        self.log_tunings = np.log(np.clip(tunings, 1e-10, None))

        if self.debug is True:
            plt.plot(tiling, tunings[::15].T)
            plt.xticks(np.linspace(tiling.min(), tiling.max(), 16))
            plt.grid()

        return tunings

    def contrast_response(self, pop_resp, contrast=100):
        # Scales population response according to contrast response function (CRF) - Naka-Rushton
        n = self.param_distribs['crf_exponent'][:, None, None]
        c50 = self.param_distribs['crf_c50'][:, None, None]
        r0 = self.param_distribs['spont'][:, None, None]  # set spont firing rate independently from rmax

        # Ensure contrast is an array of shape (:, n_contrasts, :)
        contrast = np.atleast_1d(contrast)[None, :, None]

        # Naka-Rushton scaling factor
        scale = (contrast ** n / (contrast ** n + c50 ** n))

        # Apply scaling and add spontaneous rate
        crf_resp = pop_resp * scale + r0

        return crf_resp

    def noisy_response(self, pop_resp):
        # Add either independent or orientation-correlated noise to population responses.
        if self.global_params.get('correlated_noise'):

            # Correlation structure is fixed; random noise is new for every trial.
            noise = self.gen_correlated_noise(pop_resp, corr_max=self.global_params['corr_max'],
                                              corr_width=self.global_params['corr_width'])

        elif self.global_params.get('noise_type'):
            if self.global_params['noise_type'] == 'poisson':
                noise = poissonlike_noise(pop_resp, self.param_distribs['noise_scale'])
            else:
                noise = np.random.normal(size=len(pop_resp))
        else:
            noise = np.random.normal(size=len(pop_resp))

        noisy_resp = pop_resp + noise
        noisy_resp[noisy_resp <= 0] = 0  # remove negative firing rates

        return noisy_resp

    def gen_correlated_noise(self, pop_resp, corr_max=0.1, corr_width=45):
        n_neurons, n_stim, n_reps = pop_resp.shape

        # Scale Gaussian noise to mimic Poisson-like variability.
        noise_sd = np.sqrt(np.maximum(pop_resp, 1e-10))
        noise_sd *= self.param_distribs['noise_scale'][:, None, None]

        # Build the orientation-dependent correlation matrix.
        prefs = np.asarray(self.prefs, dtype=np.float64)
        delta_pd = np.abs(prefs[:, None] - prefs[None, :])
        delta_pd = np.minimum(delta_pd, 180.0 - delta_pd)

        pd_max = 90
        Td = corr_width / pd_max

        corr_matrix = corr_max * np.exp(
            -(delta_pd ** 2) / (pd_max * Td) ** 2
        )
        np.fill_diagonal(corr_matrix, 1.0)

        # Try the cheaper Cholesky factorisation first.
        try:
            cache_key = (corr_max, corr_width)

            if getattr(self, '_corr_noise_cache_key', None) == cache_key:
                L = self._corr_noise_cholesky
            else:
                L = np.linalg.cholesky(
                    corr_matrix + 1e-8 * np.eye(n_neurons)
                )
                self._corr_noise_cholesky = L
                self._corr_noise_cache_key = cache_key

            # Generate independent standard-normal noise.
            z = np.random.standard_normal(
                size=(n_neurons, n_stim * n_reps)
            )

            # Impose the desired correlations.
            correlated_noise = (L @ z).reshape(
                n_neurons, n_stim, n_reps
            )

        except np.linalg.LinAlgError:
            # Fall back to the symmetric matrix square-root method.
            eigvals, eigvecs = np.linalg.eigh(corr_matrix)
            eigvals = np.maximum(eigvals, 0.0)

            Q = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
            Q = Q.astype(np.float32)

            z = np.random.standard_normal(
                size=(n_neurons, n_stim * n_reps)
            ).astype(np.float32)

            correlated_noise = (Q @ z).reshape(
                n_neurons, n_stim, n_reps
            )

        # Scale correlated noise by the response-dependent noise SD.
        scaled_correlated_noise = correlated_noise * noise_sd

        self.corr_noise_unscaled = correlated_noise
        self.corr_noise_scaled = scaled_correlated_noise

        return scaled_correlated_noise


    def gen_stim_response(self, stim_ori, stim_con, n_reps=1, window=False):
        """
        Generate population responses for one or more orientations and repetitions.
        INPUT: stim_ori with shape [n_oris] or [n_oris, n_reps]
        OUTPUT: resp_noisy with shape [n_neurons, n_oris, n_reps]
        """

        # Convert orientations into circular space.
        stim_ori = np.atleast_1d(stim_ori)
        stim_ori = convert_to_circular_space(stim_ori, self.ori_space.range)

        tunings = self.tunings
        tiling = self.ori_space.tiling

        # Find stimulus orientation indices within the selected tiling.
        _, stim_ori_idx = find_nearest(tiling, stim_ori)

        # Extract tuned responses: [neurons, ...stim_ori dimensions].
        resp_tuned = tunings[:, stim_ori_idx]

        if stim_ori.ndim == 1:
            # Input: [n_stim_oris]
            # Output: [neurons, n_reps, n_stim_oris]
            resp_tuned = np.repeat(resp_tuned[:, :, None], n_reps, axis=2)

        elif stim_ori.ndim != 2:
            raise ValueError(
                f"stim_ori must be 1D or 2D, got shape {stim_ori.shape}"
            )

        resp_tuned_contrast_scaled = self.contrast_response(resp_tuned, stim_con)
        resp_noisy = self.noisy_response(resp_tuned_contrast_scaled)

        self.stim_ori = stim_ori
        self.stim_ori_idx = stim_ori_idx
        self.resp_tuned = resp_tuned
        self.resp_noisy = resp_noisy

        return resp_noisy

    def detection(self, stim_ori, stim_con, method, n_reps=1):
        """ Return pooled population response for detection task """

        # If multiple contrasts entered for a single orientation, match up here
        if isinstance(stim_con, (list, tuple)) and not isinstance(stim_ori, (list, tuple)):
            stim_ori = [stim_ori] * len(stim_con)

        # Generate responses
        resp_noisy = self.gen_stim_response(stim_ori, stim_con, n_reps)

        if method == 'peak':
            pooled_resp = np.max(resp_noisy, 0)

        elif method == 'sum':
            pooled_resp = np.sum(resp_noisy, 0)  # sum across neurons for each trial

        elif method == 'mag':  # weighted by response magnitude
            pooled_resp = self.magnitude_weighted_pooling(resp_noisy, q=self.global_params['pooling_q'])

        elif method == 'distance':  # weighted by distance from stim
            pooled_resp = self.distance_sum()

        elif method == 'distance_v_mag':
            pooled_resp = self.distance_vectormag()

        elif method == 'PV':
            pooled_resp = self.popvector(_detection=True)  # estimates actual contrast [?]
        else:
            print(f'{method}\nNo valid detection pooling method chosen')
            pooled_resp = None

        return pooled_resp

    def magnitude_weighted_pooling(self, resp=None, q=None):
        # weighted sum of population response (Quick, 1974; also see Goris 2013)
        if resp is None:
            resp = self.resp_noisy
        if q is None:
            q = self.global_params['pooling_q']  # scalar (if diff for each nrn then reshape)

        pooled_val = np.sum(resp ** q, 0) ** (1 / q)  # sum and return to og units

        return pooled_val

    def distance_sum(self, pop_resp=None, pool_sigma=None):
        # scale response pooling based on distance from stim_ori
        if pop_resp is None:
            pop_resp = self.resp_noisy
        if pool_sigma is None:
            # pool_sigma = self.get_pooling_params(ori_std=self.stim_ori, label='exponent')
            pool_sigma = 10
        weights = generate_gaussian(x=self.prefs, mu=self.stim_ori, sigma=pool_sigma)
        weights = weights / np.max(weights)  # 0 to 1
        # pooled_val = my_asarray([weights]).T * pop_resp  # apply weights to population distance_weighted response
        pooled_val = arr([weights]).T * pop_resp  # apply weights to population distance_weighted response
        distance_val = np.sum(pooled_val, 0)
        return distance_val

    def distance_vectormag(self):
        # scale response pooling based on distance from stim_ori
        pop_resp = self.resp_noisy
        pool_sigma = self.get_pooling_params(ori_std=self.stim_ori, label='exponent')
        weights = generate_gaussian(x=self.prefs, mu=self.stim_ori, sigma=pool_sigma)
        weights = weights / np.max(weights)  # 0 to 1
        # w_pop_resp = my_asarray([weights]).T * pop_resp  # apply weights to population distance_weighted response
        w_pop_resp = arr([weights]).T * pop_resp  # apply weights to population distance_weighted response
        # VECTOR MAG POOLING
        q = self.get_pooling_params(ori_std=self.stim_ori, label='exponent')
        distance_vmag = np.sum(w_pop_resp ** q, 0) ** (1 / q)  # sum and return to og units
        return distance_vmag

    def decode_response(self, response_decoder='WTA', stim_ori=None, stim_con=None, n_reps=None, gen_resp=True,
                        save_decoded=False, load_decoded=False, window=False, prior=False):
        # If using stored decoded estimates
        if load_decoded is True and self.saved_response.get(stim_ori) is not None:
            decoded_val = self.saved_response[stim_ori]
            decoded_val = random.sample(list(decoded_val), n_reps)
            return decoded_val
        else:
            if all(i is not None for i in (stim_ori, stim_con, n_reps)):
                if gen_resp is True:
                    self.gen_stim_response(stim_ori, stim_con, n_reps)

            decoded_val = None
            if response_decoder == 'WTA':
                decoded_val = self.wta()
            elif response_decoder == 'PV':
                decoded_val = self.popvector()
            elif response_decoder == 'ML':
                decoded_val = self.maxlikelihood(None, window, prior)

            # plt.hist(decoded_val, alpha=0.8)
            # Add tiny amount of noise to separate shared estimates (for comparisons)
            decoded_val += np.random.uniform(-1e-4, 1e-4, size=decoded_val.shape)
            self.decoded = decoded_val

        # If storing decoded estimates for later use
        if save_decoded is True:
            self.saved_response[stim_ori] = self.decoded

        return decoded_val

    def wta(self, pop_resp=None, prefs=None):
        # Pref of the neurons which produced the strongest response each trial
        # pop_resp: 2D neuron(rows) x trial(cols) for resp(vals)
        # prefs: 1D prefs(vals)
        if pop_resp is None:
            pop_resp = self.resp_noisy
        if prefs is None:
            prefs = self.prefs
        max_resp = pop_resp.max(0)  # max response for each trial
        ismax = (max_resp == pop_resp).astype('int')  # bool 01 matrix of max for each trial
        ismax = ismax + np.random.uniform(-0.1, 0.1, size=ismax.shape)  # add random noise to distinguish shared vals
        wta_est = prefs[ismax.argmax(0)]  # get idx of pref with max response
        return wta_est

    def popvector(self, weights=None, pop_prefs=None, _detection=False):
        # pop_resp: 2D neuron(rows) x trial(cols) for resp(vals)
        # prefs: 1D prefs(vals)
        if weights is None:
            weights = self.resp_noisy
        if pop_prefs is None:
            pop_prefs = self.prefs
        pop_prefs = np.radians(pop_prefs)
        # Account for 180 deg of circular space
        if self.ori_space.range == 180:
            pop_prefs = pop_prefs * 2  # convert to circular space (i.e., 360 deg)

        # Circular population vector
        pop_prefs = pop_prefs[:, None, None]
        # 1. Response of each neurons represented in the complex plane
        response_vectors = weights * np.exp(1j * pop_prefs)
        # 2. Sum all vectors
        resultant_vector = np.sum(response_vectors, 0)
        # 3. Estimate angle of average vector in complex plane
        pv_est = np.angle(resultant_vector, deg=True)

        if self.ori_space.range == 180:  # Valid so long as ori_space is +- 0
            pv_est = pv_est / 2

        return pv_est

    def maxlikelihood(self, resp=None, window=False, prior=False):
        resp = self.resp_noisy if resp is None else resp

        tunings = self.tunings
        tiling = self.ori_space.tiling
        prior_ori = self.prior_ori

        # Apply orientation window if possible
        if window is True:
            if self.window_idcs:
                min_idx, max_idx = self.window_idcs
            else:
                min_idx, max_idx = self.get_tuning_window(buffer=20)

            tunings = self.tunings[:, min_idx:max_idx]
            tiling = self.ori_space.tiling[min_idx:max_idx]

            if prior and prior_ori:
                prior_ori = prior_ori[min_idx:max_idx]

        # Flatten trials/repeats.
        resp_flat = resp.reshape(resp.shape[0], -1)

        # Compute Poisson log-likelihood.
        log_tunings = np.log(np.clip(tunings, 1e-10, None))
        log_L = resp_flat.T @ log_tunings - np.sum(tunings, axis=0)

        # Select maximum-likelihood orientation.
        ml_est = tiling[log_L.argmax(axis=1)]

        # Reshape into multiple repetitions of each stimulus orientation
        ml_est = ml_est.reshape(resp.shape[1:])

        return ml_est

    def get_tuning_window(self, buffer=30, stim_ori=None):
        """Return tuning curves and corresponding orientation tiling."""

        tiling = self.ori_space.tiling
        stim_ori = self.stim_ori if stim_ori is None else stim_ori

        # Fall back to the full orientation range near circular boundaries
        if np.max(stim_ori) + buffer >= tiling.max() or np.min(stim_ori) - buffer <= tiling.min():
            print("\n!!! WINDOWING FAILED: using full tuned range !!!\n")
            min_idx = 0
            max_idx = len(tiling) - 1

        else:
            minmax_vals = convert_to_circular_space(
                [np.min(stim_ori) - buffer, np.max(stim_ori) + buffer],
                self.ori_space.range
            )
            _, min_idx = find_nearest(tiling, minmax_vals[0])
            _, max_idx = find_nearest(tiling, minmax_vals[1])

        return min_idx, max_idx


    def ori_reproduction_check(self, stim_oris=None, category_noise_sd=None, contrast=100, n_level_reps=200,
                               prior=True, show_plots=False, save_path=''):
        tiling = self.ori_space.tiling
        if stim_oris is None:
            stim_oris = np.linspace(-80, 80, 61)

        if category_noise_sd is None:
            category_noise_sd = self.global_params.get('category_noise_sd')

        decoded_oris = self.decode_response('ML', stim_oris, contrast, n_reps=n_level_reps,
                                            window=False, prior=prior)

        repro_error = decoded_oris.T - stim_oris
        mean_repro_error = np.mean(repro_error, axis=0)
        sd_repro_error = np.std(repro_error, axis=0)

        plt.plot(stim_oris, mean_repro_error, color='black')
        plt.fill_between(stim_oris, mean_repro_error - sd_repro_error,
                         mean_repro_error + sd_repro_error, alpha=0.2)
        plt.xlabel("Stimulus Orientation")
        plt.ylabel("Veridical error")

        if show_plots:
            plt.show()
        else:
            plt.close()

        return repro_error

    def plot_tunings(self, neuron_idx=None, n_neurons=None, prefs=None, ax=None,
                     alpha=0.5, linewidth=1, show_prefs=True, show=True):

        # Check that tuning curves have been generated.
        if self.tunings is None or len(self.tunings) == 0:
            raise ValueError("No tuning curves found. Run generate_tunings() first.")

        tiling = np.asarray(self.ori_space.tiling)
        tunings = np.asarray(self.tunings)

        # Select neurons by index, preferred orientation, or evenly across the population.
        if neuron_idx is not None:
            neuron_idx = np.atleast_1d(neuron_idx).astype(int)

        elif prefs is not None:
            prefs = np.atleast_1d(prefs)
            neuron_idx = np.asarray([
                np.argmin(np.abs(self.prefs - pref)) for pref in prefs
            ])

        else:
            if n_neurons is None:
                n_neurons = min(60, tunings.shape[0])

            neuron_idx = np.linspace(
                0, tunings.shape[0] - 1, n_neurons, dtype=int
            )

        if np.any(neuron_idx < 0) or np.any(neuron_idx >= tunings.shape[0]):
            raise IndexError("neuron_idx contains an invalid neuron index.")

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        # Plot each selected neuron's tuning curve.
        for idx in neuron_idx:
            label = f"Pref = {self.prefs[idx]:.1f}°" if show_prefs else None
            ax.plot(
                tiling,
                tunings[idx],
                alpha=alpha,
                linewidth=linewidth,
                label=label
            )

        ax.set_xlabel("Stimulus orientation (°)")
        ax.set_ylabel("Response")
        ax.set_xlim(tiling.min(), tiling.max())
        ax.grid(alpha=0.25)

        if show_prefs:
            ax.legend(title="Neuron", bbox_to_anchor=(1.02, 1), loc="upper left")

        if show:
            plt.show()

        return ax

