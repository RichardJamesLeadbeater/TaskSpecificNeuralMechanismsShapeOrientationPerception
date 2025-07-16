import numpy as np
from numpy import asarray as arr, newaxis as nax
from scipy.linalg import hankel
from scipy.optimize import minimize
import pickle
import random
from .general_functions import find_nearest, generate_cosine, generate_vonmises, normalise, poissonlike_noise
from components.run import set_random_seed
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


def gen_population_code(cardinal_params, oblique_params, orispace, **kwargs):
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

    anisotropy_info = {'param_labels': ['n_prefs', 'hwhm', 'pref_distrib', 'param_distrib', 'rmax', 'spont',
                                        'noise_scale', 'crf_exponent', 'crf_c50'],
                       'cardinal_params': cardinal_params.__dict__, 'oblique_params': oblique_params.__dict__}
    population_code = PopulationCode(orispace=orispace, anisotropy=anisotropy_info, global_params=global_params,
                                     debug=kwargs.get('debug', False), pref_sampling='whole')
    population_code.init_population_code()

    return population_code

class Params:
    def __init__(self, dicty=None, hwhm=None, sigma=None, kappa=None, n_prefs=None,
                 rmax=None, spont=None, crf_exponent=None, crf_c50=None,
                 noise_scale=1, noise_type='poisson', pref_distrib='flat', param_distrib='flat',
                 pooling_q=None,
                 ):
        # Set attributes
        if not isinstance(dicty, dict):
            self.hwhm = hwhm
            self.sigma = sigma
            self.kappa = kappa
            self.n_prefs = n_prefs
            self.pref_distrib = pref_distrib
            self.param_distrib = param_distrib
            self.rmax = rmax
            self.noise_scale = noise_scale
            self.noise_type = noise_type
            self.spont = spont
            self.crf_exponent = crf_exponent
            self.crf_c50 = crf_c50
            self.pooling_q = pooling_q

        # Use dict to set attributes
        if isinstance(dicty, dict):
            for i_key in Params().__dict__:
                if dicty.get(i_key) is not None:
                    setattr(self, i_key, dicty[i_key])  # if exists then set value
                else:
                    setattr(self, i_key, Params().__dict__[i_key])  # default value if not in dicty

        # Calculate sigma and from hwhm
        if self.hwhm:
            if not self.sigma:
                self.sigma = self.hwhm / np.sqrt(2 * np.log(2))  # s = hwhm / 2.3548
            if self.kappa is None:
                self.kappa = np.log(0.5) / (np.cos(2 * self.hwhm) - 1)  # Swindale (1998)

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
    def __init__(self, orispace, anisotropy=None, name=None, debug=True, **kwargs):
        self.debug = debug  # if True then error checks will occur
        self.param_distribs = None
        self.params = None
        self.orispace = orispace
        # Global parameters
        self.global_params = kwargs.get('global_params')
        if kwargs.get('global_params') is not None:
            # Total N set for 180 deg - Scale N for larger deg range - # TODO can delete
            n_samples = int(self.global_params['n_prefs'] * (self.orispace.range / 180))
            self.global_params['n_samples'] = n_samples
        self.noise_type = 'poisson'
        # Anisotropic parameters
        self.anisotropy = anisotropy
        self.tunings = []
        self.prefs = []
        self.prefs_idx = []
        # Stimuli details
        self.stim_ori = None
        self.stim_ori_idx = None
        self.stim_contrast = None
        # Response
        self.resp_tuned = None
        self.resp_contrast = None
        self.resp_noisy = None
        # Decoder
        self.decoded = None

    def gen_anisotropic_distribution(self, o_lim=None, c_lim=None):
        # NEW MODEL - all_prefs from a single distribution #
        g_params = self.global_params
        tiling = self.orispace.tiling
        n_cycles = 4  # cycles per 360 deg, therefore 90 deg period required 4 cycles
        # peaks=cardinal  troughs=obliques
        # Set distribution for params and prefs
        for i_dist in ['pref_distrib', 'param_distrib']:
            distrib = self.global_params[i_dist]
            if any(_i == distrib for _i in ['cos']):
                curve = generate_cosine(tiling, n_cycles=n_cycles)
            elif g_params[i_dist] == 'flat':
                curve = generate_cosine(tiling, n_cycles=n_cycles)
                curve[curve <= 0.5] = 0  # oblique
                curve[curve > 0.5] = 1  # cardinal
            else:
                print(f"Warning: Invalid {i_dist}")
                curve = None
            self.anisotropy[i_dist] = curve

    def init_population_code(self):
        # Define distribution curve for cardinal and oblique parameters
        self.gen_anisotropic_distribution()
        self.gen_prefs()  # sampled orientation preferences
        self.gen_param_distributions()  # parameters for each neuron sampled according to orientation preference
        self.gen_tunings()  # tuning curve for each pref

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

    def gen_prefs(self):
        # NEW MODEL - all_prefs from a single distribution #
        # ORIENTATION PREFERENCE DISTRIBUTIONS #
        g_params = self.global_params
        n_cardinal = int(np.round(g_params['n_samples'] * g_params['propn_cardinal']))
        tiling = self.orispace.tiling

        # Generate preferences using pre-defined distribution
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

        # Assign to model
        (prefs, prefs_idx) = find_nearest(tiling, prefs_info['prefs'])
        self.prefs = prefs
        self.prefs_idx = prefs_idx

        return prefs

    def gen_param_distributions(self):
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

    def gen_tunings(self):
        param_distribs = self.param_distribs
        tiling = self.orispace.tiling
        hwhm = param_distribs['hwhm']

        # Von Mises tuning
        rad_hwhm = np.radians(hwhm)  # must be in radians
        kappa = np.log(0.5) / (np.cos(2 * rad_hwhm) - 1)  # Swindale (1998)
        tunings = generate_vonmises(tiling, self.prefs, kappa, input_format='deg')

        # Normalise so each tuning curve (row) has min=0 and max=1
        cols_idcs = np.argmax(tunings, axis=1)
        rows_idcs = np.arange(0, len(cols_idcs), 1)
        max_vals = tunings[rows_idcs, cols_idcs]
        tunings = tunings / arr([max_vals]).T
        # Set maximum firing rate to rmax
        tunings = (tunings.T * param_distribs['rmax'])
        self.tunings = tunings.T

        return tunings

    def contrast_response(self, pop_resp, stim_contrast=100):
        # Scales population response according to contrast response function (CRF) - Naka-Rushton
        n = self.param_distribs['crf_exponent']
        c50 = self.param_distribs['crf_c50']
        crf_resp = pop_resp.T * (stim_contrast ** n / (stim_contrast ** n + c50 ** n))
        # Spontaneous firing proportional to rmax (independent from contrast scaling)
        r0 = self.param_distribs['spont']  # set spont firing rate independently from rmax
        crf_resp = crf_resp + r0
        crf_resp = np.transpose(crf_resp)
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

    def gen_stim_response(self, stim_ori, stim_contrast=100, n_reps=1):
        # # # Generate a population response for a given orientation value # # #
        # Stimulus
        tiling = self.orispace.tiling
        _, stim_ori_idx = find_nearest(tiling, stim_ori)  # idx of stim_ori in orientation space

        # Response
        resp_tuned = self.tunings[:, stim_ori_idx]  # Tuned response to stim_ori
        resp_tuned = np.transpose(resp_tuned * np.ones([n_reps, 1]))  # Tuned response across trials / repeats
        resp_tuned = self.contrast_response(resp_tuned, stim_contrast)
        resp_noisy = self.noisy_response(resp_tuned)

        # Update attributes
        self.stim_ori = stim_ori
        self.stim_ori_idx = stim_ori_idx
        self.stim_contrast = stim_contrast
        self.resp_tuned = resp_tuned
        self.resp_noisy = resp_noisy

        return resp_noisy

    def detection(self, pooling_method, stim_ori=None, stim_con=None, n_reps=None):
        # return pooled population response for detection task #
        if all(i or i == 0 for i in [stim_ori, stim_con, n_reps]):
            self.gen_stim_response(stim_ori, stim_con, n_reps)
        if pooling_method == 'peak':
            pooled_response = np.max(self.resp_noisy, 0)
        elif pooling_method == 'sum':
            pooled_response = np.sum(self.resp_noisy, 0)  # sum across neurons for each trial
        elif pooling_method == 'mag':  # weighted by response magnitude
            pooled_response = self.vector_mag()
        else:
            print('No detection pooling method chosen')
            pooled_response = None

        return pooled_response

    def vector_mag(self, pop_resp=None, q=None):
        # weighted sum of population response (Quick, 1974; also see Goris 2013)
        if pop_resp is None:
            pop_resp = self.resp_noisy
        if q is None:
            q = self.global_params['pooling_q']
        vector_mag_val = np.sum(pop_resp ** q, 0) ** (1 / q)  # sum and return to og units

        return vector_mag_val

    def decode_response(self, response_decoder='WTA', stim_ori=None, stim_con=None, n_reps=None, gen_resp=True):
        # If using stored decoded estimates
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
        else:
            print('Warning: Response decoder not recognised')
        # plt.hist(decoded_val, alpha=0.8)
        # Add tiny amount of noise to separate shared estimates
        decoded_val = decoded_val + np.random.uniform(-0.0001, 0.0001, size=len(decoded_val))
        self.decoded = decoded_val

        return decoded_val

    def wta(self, pop_resp=None, prefs=None):
        # Pref of the neurons which produced the strongest response each trial
        # pop_resp: 2D neuron(rows) x trial(cols) for resp(vals)
        # prefs: 1D prefs(vals)
        if pop_resp is None:
            pop_resp = self.resp_noisy
        if prefs is None:
            prefs = self.prefs
        ismax = (pop_resp.max(0) == pop_resp).astype('int')  # bool 01 matrix of max for each trial
        ismax = ismax + np.random.uniform(-0.1, 0.1, size=ismax.shape)  # add random noise to distinguish shared vals
        wta_est = prefs[ismax.argmax(0)]  # get idx of pref with max response

        return wta_est

    def popvector(self, weights=None, pop_prefs=None, _detection=False):
        # pop_resp: 2D neuron(rows) x trial(cols) for resp(vals)
        # prefs: 1D prefs(vals)
        if weights is None:
            weights = self.resp_noisy
            # weights could also be set as self.resp_tuned
        if pop_prefs is None:
            pop_prefs = self.prefs
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
            pop_tunings = self.tunings
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
