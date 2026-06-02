import numpy as np
from numpy import asarray as arr, newaxis as nax
from scipy.linalg import hankel
from scipy.optimize import minimize
import random
from .general_functions import get_attrib, save_attrib, find_nearest, generate_gaussian, generate_cosine, \
    generate_vonmises, convert_to_circular_space, \
    normalise, poissonlike_noise, generate_wrapped_gaussian
from components.run import set_random_seed
import matplotlib.pyplot as plt


def prefs_sampling_error(prefs_fun, scale, desired_n, **kwargs):
    """
    For a given scale, calculate error between the desired and actual n_cardinal
    """
    scale = np.exp(scale)  # Works with negative values
    output = prefs_fun(scale=scale, size=kwargs.get('size', 10000))
    n = output['n_cardinal']
    error = abs(n - desired_n)
    if kwargs.get('_print') is True:
        print(f"\nscale = {scale}"
              f"\nerror = {error}")
    return error


def gen_population_code(cardinal_params, oblique_params, orispace, pref_sampling='whole', **kwargs):
    # Params defined for the whole populations (default)
    global_params = {
        'n_prefs': cardinal_params.n_prefs + oblique_params.n_prefs,
        'propn_cardinal': cardinal_params.n_prefs / (cardinal_params.n_prefs + oblique_params.n_prefs),
        'noise_type': 'poisson',
        'pref_distrib': 'cos',
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
    population_code = PopulationCode(orispace=orispace, anisotropy=anisotropy_info, debug=kwargs.get('debug', False),
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


class PopulationCode:
    def __init__(self, orispace, anisotropy=None, name=None, debug=True, **kwargs):
        self.pref_sampling = kwargs.get('pref_sampling')  # whether old version used (individual neural populations)
        self.params_co = None
        self.param_distribs = None
        self.params = None  # callable with .get_params()
        self.debug = debug  # if True then error checks will occur
        self.orispace = orispace
        self.global_params = kwargs.get('global_params')
        if kwargs.get('global_params') is not None:
            self.global_params['n_samples'] = int(
                self.global_params['n_prefs'] * (self.orispace.range / 180))  # whole range (not per 180)
        # Anisotropic parameters
        self.anisotropy = anisotropy
        # the following will be appended in methods of class
        self.tunings = []
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
        # Define distribution curve for cardinal and oblique parameters
        self.gen_anisotropic_distribution()
        self.generate_prefs()  # sampled orientation preferences

        # Check sampling rates - difference between adjacent prefs
        # prefs = self.prefs
        # diffs = [abs(_i - prefs[_idx + 1]) for _idx, _i in enumerate(prefs) if _idx < len(prefs) - 1]
        # fig = plt.plot(prefs[:-1], diffs)
        # fig[0].axes.set(xlabel="Orientation Preference", xticks=np.linspace(-180, 180, 9),
        #                 ylabel="Spacing from adjacent pref [i+1]")
        # plt.close('all')

        self.generate_params()  # parameters for each neuron sampled according to orientation preference
        self.generate_tunings()  # tuning curve for each pref

    def gen_prefs_distrib(self, scale, size=100000):
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
        indices = np.searchsorted(cdf, uniform_samples)

        # Corresponding orientations
        prefs = self.orispace.tiling[indices]
        prefs.sort()

        # Count cardinal orientations (within 22.5° of 0°, 90°, 180°, ...)
        n_cardinal = np.sum((prefs % 90 < 22.5) | (prefs % 90 > 67.5))
        n_oblique = n_samples - n_cardinal

        output = {'prefs': prefs, 'n_cardinal': n_cardinal, 'n_oblique': n_oblique,
                  'propn_cardinal': np.round(n_cardinal / (self.global_params['n_samples']), 3)}

        return output

    def generate_prefs(self):
        # NEW MODEL - prefs from a single distribution #
        # ORIENTATION PREFERENCE DISTRIBUTIONS #
        g_params = self.global_params
        n_cardinal = self.anisotropy['cardinal_params']['n_prefs']
        tiling = self.orispace.tiling

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

            min_fun = lambda scale: prefs_sampling_error(prefs_fun=self.gen_prefs_distrib, scale=scale - bias, size=10000,
                                                         desired_n=n_cardinal, bias=bias, _print=False)

            np.random.seed(1)  # Ensures replicable simulations with shared parameters
            result = minimize(min_fun, x0=x0, method='Nelder-Mead', options={'maxiter': 200})

            chosen_scale = np.exp(result.x[0] - bias)

        # GENERATE PREFS #
        # Using chosen scale for cosine curve
        prefs_info = self.gen_prefs_distrib(scale=chosen_scale, size=100000)

        # If actual propn_cardinal substantially deviates from desired, then print
        _print_actual = False
        if _print_actual is True:
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
        # NEW MODEL - prefs from a single distribution #
        g_params = self.global_params
        param_distribs = self.param_distribs
        tiling = self.orispace.tiling
        fwhm = param_distribs['fwhm']
        # Gaussian tuning
        if g_params['tuning_dist'] == 'gaussian':
            sigma = fwhm / 2.3548
            sigma = arr([sigma]).transpose()  # 1 val per ori_pref
            tunings = generate_gaussian(tiling, self.prefs, sigma)  # gaussian tuning for each pref
        # Wrapped Gaussian tuning
        elif g_params['tuning_dist'] == 'gaussian_wrapped':
            sigma = fwhm / 2.3548
            sigma = arr([sigma]).transpose()  # 1 val per ori_pref
            tunings = generate_wrapped_gaussian(tiling, self.prefs, sigma, period=180, wrap_range=8)
        # Von Mises tuning
        elif g_params['tuning_dist'] == 'von_mises':
            rad_fwhm = np.radians(fwhm)  # must be in radians
            kappa = np.log(0.5) / (np.cos(rad_fwhm) - 1)  # Swindale (1998)
            kappa = kappa[:, nax]  # 1 val per ori_pref
            tunings = generate_vonmises(tiling, self.prefs, kappa, input_format='deg')
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
        self.tunings = tunings
        self.log_tunings = np.log(np.clip(tunings, 1e-10, None))

        if self.debug is True:
            plt.plot(tiling, tunings[::5].T)
            plt.yticks(np.linspace(tiling.min(), tiling.max(), 16))
            plt.grid()

        return tunings

    def contrast_response(self, pop_resp, contrast=100):
        # Scales population response according to contrast response function (CRF) - Naka-Rushton
        n = self.param_distribs['crf_exponent'][:, None, None]
        c50 = self.param_distribs['crf_c50'][:, None, None]
        r0 = self.param_distribs['spont'][:, None, None]  # set spont firing rate independently from rmax

        # Ensure contrast is an array of shape (1, 1, n_contrasts)
        contrast = np.atleast_1d(contrast)[None, None, :]

        # Naka-Rushton scaling factor
        scale = (contrast ** n / (contrast ** n + c50 ** n))

        # Apply scaling and add spontaneous rate
        crf_resp = pop_resp * scale + r0

        return crf_resp

    def noisy_response(self, pop_resp):
        # Noise added onto tuned response
        if self.global_params['noise_type'] == 'poisson':
            noise = poissonlike_noise(pop_resp, self.param_distribs['noise_scale'])
        else:
            noise = np.random.normal(size=len(pop_resp))
        noisy_resp = pop_resp + noise
        noisy_resp[noisy_resp <= 0] = 0  # remove negative firing rates
        return noisy_resp

    def gen_stim_response(self, stim_ori, stim_con, n_reps=1):
        """
        Generate population response given orientation(s) and contrast(s), repeated over n_reps
        """
        # Convert into orientation circular space (-90 to 90)
        stim_ori = convert_to_circular_space(stim_ori, self.orispace.range)

        _, stim_ori_idx = find_nearest(self.orispace.tiling, stim_ori)  # idx of stim_ori in orientation space

        # Extract tuned responses for all requested orientations
        stim_ori_idx = np.atleast_1d(stim_ori_idx)
        resp_tuned = self.tunings[:, stim_ori_idx]  # shape [neurons, n_oris]

        # Repeat each column for n_reps
        resp_tuned = np.repeat(resp_tuned[:, :, None], n_reps, axis=2)  # shape [neurons, n_oris, n_reps]
        resp_tuned = np.swapaxes(resp_tuned, 1, 2)  # (360, 1000, 2)

        resp_tuned = self.contrast_response(resp_tuned, stim_con)
        resp_noisy = self.noisy_response(resp_tuned)

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
            # q was 6 in thesis
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
                        save_decoded=False, load_decoded=False):
        # If using stored decoded estimates
        if load_decoded is True and self.saved_response.get(stim_ori) is not None:
            decoded_val = self.saved_response[stim_ori]
            decoded_val = random.sample(list(decoded_val), n_reps)
            return decoded_val
        else:
            if all(i or i == 0 for i in [stim_ori, stim_con, n_reps]):
                if gen_resp is True:
                    self.gen_stim_response(stim_ori, stim_con, n_reps)

            decoded_val = None
            if response_decoder == 'WTA':
                decoded_val = self.wta()
            elif response_decoder == 'PV':
                decoded_val = self.popvector()
            elif response_decoder == 'ML':
                decoded_val = self.maxlikelihood()

            # plt.hist(decoded_val, alpha=0.8)
            # Add tiny amount of noise to separate shared estimates (for comparisons)
            decoded_val += np.random.uniform(-1e-4, 1e-4, size=decoded_val.shape)
            self.decoded = decoded_val

        # If storing decoded estimates for later use
        if save_decoded is True:
            self.saved_response[stim_ori] = self.decoded

        return decoded_val

    def cramer_rao(self, stim_ori, stim_con=100, n_reps=1):
        self.gen_stim_response(stim_ori, stim_con, n_reps=n_reps)
        resp_noisy = self.resp_noisy  # each nrn noisy_resp across trials
        resp_tuned = self.resp_tuned  # each nrn tuned_resp across orispace
        # Define the parameter and likelihood function
        resp_tuned = resp_tuned + (1 ** -10)  # removes any vals=zero
        log_likelihood = resp_noisy.T @ np.log(resp_tuned)  # outputs: [trials x orispace]
        # # # ... in this matrix multiplication, summed signal (across neurons) each point of orispace
        # Calculate Fisher Information Matrix
        score_function = np.gradient(log_likelihood, axis=1)  # use axis 1 as only interested in each trial
        fisher_info = -np.gradient(score_function, axis=1)
        # Calculate Cramer-Rao bound (CRB) for given stim_ori by inverting Fisher Information Matrix
        crb = fisher_info ** -1
        crb_idx = crb[:, self.stim_ori_idx]
        crb_ave = np.mean(crb_idx)
        return crb_ave

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
        if self.orispace.range == 180:
            pop_prefs = pop_prefs * 2  # convert to circular space (i.e., 360 deg)

        # Circular population vector
        pop_prefs = pop_prefs[:, None, None]
        # 1. Response of each neurons represented in the complex plane
        response_vectors = weights * np.exp(1j * pop_prefs)
        # 2. Sum all vectors
        resultant_vector = np.sum(response_vectors, 0)
        # 3. Estimate angle of average vector in complex plane
        pv_est = np.angle(resultant_vector, deg=True)

        if self.orispace.range == 180:  # Valid so long as orispace is +- 0
            pv_est = pv_est / 2

        return pv_est

    def maxlikelihood(self, resp=None, tunings=None, tiling=None, window=True):
        resp = self.resp_noisy if resp is None else resp
        log_tunings = np.log(np.clip(tunings, 1e-10, None)) if tunings is not None else self.log_tunings
        tiling = self.orispace.tiling if tiling is None else tiling

        # Flatten trials/repeats
        trials = np.prod(resp.shape[1:])  # 2*2 = 4
        resp_flat = resp.reshape(resp.shape[0], trials)  # shape: (3,4)

        # Default is to include all tunings
        min_bound_idx = 0

        # Only include responses within an expected range given stimulus orientation (speed up)
        if window is True:
            # Valid when there is only one expected minima (buffer=5 gives same precision as full range)
            stim_ori = self.stim_ori
            buffer = 10  # TODO May become more of an issue at low contrasts but not if 100% contrast

            if np.max(stim_ori) + buffer > 90 or np.min(stim_ori) - buffer < -90:
                print(f"\n!!! MAX LIKELIHOOD SHORTCUT FAILED !!!\n---Applying ML to full tuned range\n")
            else:
                max_bound_val = convert_to_circular_space(np.max(stim_ori) + buffer, self.orispace.range)
                min_bound_val = convert_to_circular_space(np.min(stim_ori) - buffer, self.orispace.range)
                _, max_bound_idx = find_nearest(tiling, max_bound_val)
                _, min_bound_idx = find_nearest(tiling, min_bound_val)
                log_tunings = log_tunings[:, min_bound_idx:max_bound_idx]

        # Reduce memory size
        resp_flat = resp_flat.astype(np.float32)
        log_tunings = log_tunings.astype(np.float32)

        # Matrix multiplication: neuron axis aligns with log_tunings
        log_likelihood = resp_flat.T @ log_tunings  # shape: (2000, 3601)

        # Find orientation with highest likelihood
        ml_est = tiling[min_bound_idx + log_likelihood.argmax(axis=1)]

        # Optionally, reshape back to original trial layout (each stim is each col)
        ml_est = ml_est.reshape(resp.shape[1:])

        return ml_est

    def gen_template_ori(self, target_ori, n_reps):
        # Generate the decoded template / internal representation of the target orientation
        template = self.decoded_template[target_ori]  # template vals for this orientation
        set_random_seed()
        np.random.shuffle(template)
        template_vals = random.sample(list(template), n_reps)
        return template_vals

