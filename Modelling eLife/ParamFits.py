"""
Population Response Coding Model - Parameter Fitting

# TODO - Check if the below is true
Feature space: Extended from (0:+180) to (-90:+270) for circular tuning
Stimuli: Presented only within (0:+180) range
Preferences: 180-degree moving window that shifts with stimulus value

VERSION 2 - IMPROVED DATA SAVING OF ALL PARAMETERS AND MSE OF EACH MODEL FIT
"""
import copy
from copy import deepcopy
import itertools as itls
import os
from multiprocessing import Pool
import numpy as np
from numpy import asarray as arr
import pandas as pd
from scipy.optimize import differential_evolution
import time

# Custom modules
from mytools.dict_tools import append_dicty
from mytools.analysis import get_ave_data
from mytools.dframe_tools import (init_cols, custom_sort)

# Model components
from components.run import (mocs_task, run)
from components.population_code import Params, Tiling, gen_population_code
from components.model_simulation import (run_contrast_detection, run_orientation_discrimination, ModelSimulator,
                                         extract_output_measures)

# ===
# Global variables
# ===

# Stores the full model output for every candidate parameter set evaluated during
# optimisation. This is reset at key stages so that each stage only retains the
# fits relevant to that stage.
all_fits_list = []

def run_de(objH, param_names, bounds, popsize, maxiter, seed=None):
    """
    Run one differential-evolution optimisation.

    This function performs one global search over the parameter space using a
    specified random seed. It returns both the best-fitting parameter set found
    by this search and all model fits evaluated during the search.
    """

    # Differential evolution repeatedly proposes candidate parameter vectors.
    # Each vector is passed to run_objective_fn, which returns a single error
    # value. The optimiser then searches for the vector that minimises this error.
    de_result = differential_evolution(
        # lambda x: run_objective_fn(
        #     x, param_names, objH, n_eval=2, n_level_reps=100, append_all_fits=True
        # ),
        lambda x: run_objective_fn(
            x, param_names, objH, n_eval=1, n_level_reps=50, append_all_fits=True
        ),
        bounds=bounds,
        popsize=popsize,
        maxiter=maxiter,
        polish=False,
        seed=seed
    )

    # Convert the optimiser's best parameter vector into a labelled dictionary.
    # Fraction parameters were fitted on a log scale, so convert them back to
    # linear ratio values for reporting.
    opt_fit = {_k: _v for _k, _v in zip(param_names, de_result.x)}
    for k, v in opt_fit.items():
        v = np.exp(v) if k.endswith('_fraction') else v
        opt_fit[k] = np.round(v, 5)

    print(f"Best fitting parameters: {opt_fit}")
    print(f"Best fitting objective (i.e. error): {de_result.fun}")
    print(f"{'=' * 60}\n")

    # Label each saved fit with the seed that produced it. This makes it possible
    # to trace each candidate solution back to its independent DE run.
    global all_fits_list
    for sim_idx in range(len(all_fits_list)):
        all_fits_list[sim_idx]['seed'] = seed
    de_result = {'diff_evol': de_result, 'all_fits': all_fits_list, 'seed': seed}

    return de_result


def run_objective_fn(param_values, param_names, objH, n_eval, n_level_reps=None, append_all_fits=False,
                     _return_output=False):
    """
    Evaluate one candidate parameter vector.

    Flow:
    1. Optionally change the number of simulation repetitions.
    2. Print the candidate parameter values in readable form.
    3. Run the model and compute its error relative to behavioural targets.
    4. Optionally save the full model output.
    5. Return either the scalar error or the full output table.
    """

    # Early optimisation uses fewer repetitions for speed. Later re-evaluation
    # and final simulations use more repetitions for more stable estimates.
    if n_level_reps is not None:
        objH.config['n_level_reps'] = n_level_reps

    # Print each parameter value. Log-scaled fraction parameters are converted
    # back to the original ratio scale before printing.
    for k, v in zip(param_names, param_values):
        v = np.exp(v) if k.endswith(f"_fraction") else v
        print(k, f"{v:.3f}")

    # Run the simulation and scoring routine. The ObjectiveFunction object handles
    # model simulation, extraction of the requested measure, and comparison with
    # behavioural target values.
    model_output, var_params = objH.run(param_values, param_names, n_eval)

    # The optimiser needs one scalar error value for each candidate parameter set.
    total_error = model_output['total_error'].iloc[0]
    print(f"Error = {total_error:.4f}\n")

    # Keep the full output table for later pooling, filtering, and saving.
    if append_all_fits is True:
        global all_fits_list
        all_fits_list.append(model_output)

    # During optimisation, return the scalar error. During final evaluation, return
    # the full model output, so it can be saved.
    if _return_output is True:
        return model_output
    else:
        return total_error


class ObjectiveFunction:
    """Computes error between model output and target data."""

    def __init__(self, simulator, tasknames, decoders, measure, config, error_type='mse', task_weights=None,
                 PRINTS=True):
        self.simulator = simulator
        self.tasknames = [tasknames] if not isinstance(tasknames, (list, tuple)) else tasknames
        self.decoders = [decoders] if not isinstance(decoders, (list, tuple)) else decoders

        # self.task_names = task_names if isinstance(task_names, list) else [task_names]
        self.measure = measure
        self.config = config
        self.error_type = error_type
        self.task_weights = task_weights or {_t: 1.0 for _t in self.tasknames}

        self.PRINTS = PRINTS

    def run(self, param_values, param_names, n_eval=1):
        """
        Compute error between model predictions and target data.

        This is the central scoring routine used by the optimiser. For each
        task/decoder pair, it simulates model behaviour, extracts the requested
        behavioural measure, compares it with the target data, and accumulates a
        weighted error.
        """

        # Convert the optimiser's raw parameter array into a labelled dictionary.
        # This is the format expected by the simulator.
        var_params = {name: value for name, value in zip(param_names, param_values)}

        total_error = 0
        all_model_scores = {}
        all_target_scores = {}
        error_output = {}

        all_model_output = []

        # Each objective call can involve either one task/decoder pair or several
        # pairs if the model is being fitted jointly across tasks.
        for i_task, i_decoder in zip(self.tasknames, self.decoders):
            # Run simulation
            tic = time.time()

            # Package simulation arguments. The repeated-list structure makes it
            # easy to restore repeated simulations or multiprocessing later.
            n_repeats_placeholder = 1
            args_list = [
                (i_task, i_decoder, self.measure, var_params, self.config, n_eval)
                for _ in range(n_repeats_placeholder)
            ]

            _multiprocessing = False
            if _multiprocessing is True:
                if self.PRINTS is True:
                    print(f">>> Multiprocessing {i_task} over {n_repeats_placeholder} repeats")
                with Pool(processes=4) as pool:
                    results = pool.starmap(self.simulator.simulate, [(_args) for _args in args_list])
            else:
                if self.PRINTS is True:
                    print(f">>> Running {i_task} over {n_repeats_placeholder} repeats")
                # Debug-friendly version: runs sequentially.
                results = [self.simulator.simulate(*_args) for _args in args_list]

            # Combine simulator outputs into one tidy table for scoring and saving.
            model_output = pd.concat([pd.DataFrame(_r) for _r in results])

            # Record which parameters were free in this model variant.
            for fp_idx, free_param in enumerate(param_names):
                model_output[f"freeParam_{fp_idx}"] = free_param

            # model_output = pd.DataFrame(results)

            if self.PRINTS is True:
                print(f"\n####################"
                      f"\n{time.time() - tic:.3f}s"
                      f"\n####################")

            # Convert model output into the same vector format as the behavioural
            # target data before computing the error.
            if self.measure == 'thresholds':
                model_scores = []
                for c, o in zip(model_output['cardinalT'], model_output['obliqueT']):
                    model_scores = [*model_scores, c, o]
            else:
                model_scores = arr(model_output[self.measure])

            # Behavioural target values for the current task and measure.
            target_scores = np.hstack(self.config[self.measure][i_task])  # Behavioural data

            # Compute the task-level error. Log error compares proportional
            # differences; linear error compares raw differences.
            if self.error_type == 'log':
                task_error = np.mean((np.log(model_scores) - np.log(target_scores)) ** 2)
            else:
                # MSE
                task_error = np.mean((model_scores - target_scores) ** 2)

            # Apply task weighting before adding this task to the total objective.
            # Equal weights are used unless specified otherwise.
            weighted_error = task_error * self.task_weights[i_task]
            total_error += weighted_error

            if self.PRINTS is True:
                print(f"\n{i_task} & {i_decoder}")
                print(f"  Target: {np.round(target_scores, 3)}")
                print(f"  Model:  {np.round(model_scores, 3)}")

            all_model_scores[i_task] = model_scores
            all_target_scores[i_task] = target_scores
            error_output[i_task] = task_error

            # Store the error alongside the model output so each candidate fit can
            # later be ranked, filtered, and summarised.
            model_output['task_fit_error'] = task_error

            all_model_output.append(model_output)

        if self.PRINTS is True:
            print(f"\nTotal Error (log): {total_error:.4f}")

        all_model_output = pd.concat(all_model_output)

        # Store the actual objective value used for optimisation.
        all_model_output['total_error'] = total_error

        return all_model_output, var_params


def optimize_parameters(objH, free_params, log_fractions=True, _noise_test=False, cond_labels=None,
                        save_dir=''):
    """
    Optimise model parameters to minimise the objective function.

    High-level flow:
    1. Convert the free-parameter dictionary into arrays for scipy.
    2. Optionally run a noise test to estimate stochastic variability.
    3. Run several independent differential-evolution searches.
    4. Pool the best-performing candidate fits.
    5. Re-evaluate those candidates with more simulation repetitions.
    6. Filter unstable or outlying fits.
    7. Use the median of the surviving parameters for the final model fit.
    8. Save the final output, selected fits, and summary statistics.
    """

    # scipy expects ordered arrays, whereas the script defines parameters in a
    # labelled dictionary. These lists keep names, initial values, and bounds aligned.
    param_names = list(free_params.keys())
    initial_values = [free_params[k]['x0'] for k in param_names]
    bounds = [free_params[k]['bounds'] for k in param_names]


    def objective_noise_test(n_sims=10, anisotropy_vals=None, nz_n_level_reps=None):
        """
        Evaluate the objective function repeatedly at the same parameter values.

        This gives a sense of how much the stochastic simulation changes the MSE
        even when the parameters are fixed.
        """
        anisotropy_vals = anisotropy_vals if anisotropy_vals is not None else initial_values
        objH_copy = deepcopy(objH)
        objH_copy.config['n_level_reps'] = nz_n_level_reps

        n_sims = 10
        for i_n_level_reps in [20]:
            for i_n_eval in [5, 10, 20, 40, 80, 160, 320, 640, 1280]:
                print(f"n_level_reps: {i_n_level_reps}")
                print(f"n_eval: {i_n_eval}")
                results = [objH_copy.run(anisotropy_vals, param_names, n_eval=i_n_eval) for _ in range(n_sims)]
                MSE = np.asarray([_t[0] for _t in results])
                print(
                    f">>> NOISE TEST\n"
                    f"> {n_sims} simulations\n"
                    f"> Mean MSE = {np.mean(MSE):.5f}\n"
                    f"> 95th quantile = {np.quantile(MSE, 0.95):.5f}\n"
                    f"> Mean abs deviation from mean MSE = {np.mean(abs(MSE - np.mean(MSE))):.5f}\n"
                )

    _noise_test = False
    if _noise_test is True:
        objective_noise_test(n_sims=500, anisotropy_vals=np.log([1.71, .31, 1.52, 3.01]))  # How much model varies with fixed params

    if PRINTS is True:
        print(f"\n{'=' * 60}\nStarting optimization with differential evolution (DE)\n{'=' * 60}")

    # Build a filename that records which parameters were free in this fit. This
    # prevents outputs from different model variants overwriting each other.
    params_str = ''.join(f"--{_i}" for _i in free_params.keys()).lstrip('-')
    filename = f"{objH.tasknames}_{objH.decoders}--{params_str}"

    # Differential evolution is a global search method. It explores widely across
    # the parameter space and is less likely to become trapped in a local minimum.
    # Across generations, candidate parameter values are gradually refined.

    # ------------------------
    # 1. Run DE 3 times
    # ------------------------
    global all_fits_list
    all_fits_list = []  # Updated with each simulation within DE

    # ===================
    # Change N_runs, popsize and maxiter to vary depth of optimisation (and consequently vary run time)
    # ===================

    N_runs = 1
    # popsize = 12
    # maxiter = 10

    popsize = 2
    maxiter = 2

    # Independent seeds reduce the risk that a single unlucky DE run determines
    # the final parameter estimates.
    seeds = np.random.choice(1000, size=N_runs, replace=False)

    # Prepare arguments for multiprocessing. Each worker runs one independent DE search.
    args_list = [(objH, param_names, bounds, popsize, maxiter, seed) for seed in seeds]

    with Pool(processes=min(N_runs, 3)) as pool:
        all_de_results = pool.starmap(run_de, args_list)

    # Collect all candidate fits from all independent DE searches.
    all_fits = []
    for de_run in all_de_results:
        all_fits.extend(de_run['all_fits'])

    # ------------------
    # 2. Pool top 5% of fits
    # ------------------

    # Gather every candidate evaluation from all DE runs into one table.
    all_fits = pd.concat(all_fits)

    # Keep only the strongest candidate fits for the more expensive re-evaluation stage.
    n_candidate_sets = len(all_fits.drop_duplicates(subset=param_names))  # n_unique candidates
    quantile_cutoff = 0.05 if n_candidate_sets > 50 else 0.2

    top_fits = all_fits[all_fits['total_error'] <= all_fits['total_error'].quantile(quantile_cutoff)]

    # ---------------------
    # 3. Re-run top fits to obtain stable error estimates
    # ---------------------
    all_fits_list = []  # Reset so this list only stores re-evaluated top fits.
    rerun = []

    print(len(top_fits))

    # Re-evaluate each unique candidate parameter set. The first condition is used
    # as an anchor because each parameter set appears across multiple conditions.
    for idx, row in top_fits[top_fits['cond'] == top_fits['cond'].tolist()[0]].iterrows():

        # Extract parameter values from the selected row.
        vals = row[param_names].tolist()

        # Convert anisotropy fraction parameters back to the log scale expected by
        # the objective function.
        for _idx, _lbl in enumerate(param_names):
            vals[_idx] = np.log(vals[_idx]) if _lbl.endswith('_fraction') else vals[_idx]

        # Re-simulate selected candidates using more repetitions for a more stable
        # estimate of their error.
        result = run_objective_fn(
            vals, param_names, objH, n_eval=1, n_level_reps=500, append_all_fits=True
        )
        rerun.append(result)

    # ---------------------
    # 4. Filter out any outliers after re-evaluation
    # ---------------------

    # Replace the original top fits with their re-evaluated outputs.
    top_fits = pd.concat(all_fits_list) if isinstance(all_fits_list, list) else all_fits_list

    # Remove fits whose re-evaluated error is much larger than the typical error
    # for the same task, decoder, and condition. This guards against stochastic
    # outliers after re-simulation.
    top_fits = top_fits[
        top_fits['total_error'] <= top_fits.groupby(
            ['task', 'decoder', 'cond']
        )['total_error'].transform('median') * 2
        ].reset_index(drop=True)  # ungrouped, clean index


    # -----------------------
    # 5. Use median to obtain final parameter estimates
    # -----------------------

    # Use the median of robust, re-evaluated top fits as the final parameter
    # estimate. Fraction parameters are passed back on the log scale.
    final_param_vals = top_fits[param_names].median().to_list()

    for idx, param_name in enumerate(param_names):
        if param_name.endswith('_fraction'):
            final_param_vals[idx] = np.log(final_param_vals[idx])

    # Run a high-precision final simulation and save the resulting model output.
    final_output = run_objective_fn(
        final_param_vals,
        param_names,
        objH,
        n_eval=10,
        n_level_reps=1000,
        append_all_fits=True,
        _return_output=True
    )
    final_output.to_csv(os.path.join(save_dir, f'optFit-{filename}.csv'), index=False)

    # ------------------------
    # 6. Calculate summary statistics
    # ------------------------

    # Summarise the distribution of selected top fits by task, decoder, and condition.
    numeric_cols = top_fits.select_dtypes(include="number").columns
    measure_order = ['median', 'q25', 'q75', 'mean', 'std', 'sem']

    top_summary = get_ave_data(top_fits, ['task', 'decoder', 'cond'], numeric_cols)
    top_summary = custom_sort(
        top_summary,
        ['measure', 'task', 'decoder', 'cond'],
        [measure_order, ['con_det', 'ori_dis'], [], ['low', 'mid', 'high']]
    )

    # ----------------------
    # 7. Save
    # ----------------------

    # Save the robust top-fit table.
    top_fits = top_fits.copy().round(6)
    top_fits.to_csv(os.path.join(save_dir, f'topFits-{filename}.csv'))

    # Save summary statistics for the selected top fits.
    top_summary = top_summary.copy().round(5)
    top_summary.to_csv(os.path.join(save_dir, f'summaryTopFits_{filename}.csv'))

    return None


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':

    # The main block defines the fitting experiment: which tasks, decoders,
    # behavioural measures, free parameters, and output folders should be used.
    # It then loops through model variants and launches the optimisation pipeline.

    # Configuration
    DEBUG = False
    PRINTS = False

    DECODERS = {'con_det': ['sum', 'mag', 'peak'],
                'ori_dis': ['WTA', 'PV', 'ML']}
    TASK_NAMES = ['ori_dis', 'con_det']  # which tasks to optimise ['con_det', 'ori_dis']

    FIT_BOTH_TASKS = False  # Whether to fit simulations to both tasks at once
    FIT_AVERAGE = True  # Fit to average data
    FIT_INDIVIDUAL = False  # Fit to each participant

    MEASURE = 'OEI'  # 'OEI' or 'thresholds'
    CONTRAST = 100  # Contrast for orientation discrimination
    TOTAL_N = 180  # ---Note--- Switch to N=180 (or lower) for faster simulations

    ERROR_TYPE = 'lin'
    FREE_PARAM_IDS = {'n_fraction': 'Nf',
                      'fwhm_fraction': 'Hf',
                      'noise_scale_fraction': 'NZf',
                      'rmax_fraction': 'Rf'}

    # Generate every non-empty combination of anisotropy parameters. Each
    # combination defines a different model variant to be fitted.
    allFreeParams = ['n_fraction', 'fwhm_fraction', 'noise_scale_fraction', 'rmax_fraction']

    _all_combinations = True

    if _all_combinations is True:
        freeParamCombos = []

        for r in range(1, len(allFreeParams) + 1):
            freeParamCombos.extend(itls.combinations(allFreeParams, r))
    else:
        freeParamCombos = [allFreeParams]

    # Fit every model variant in turn.
    for FREE_PARAMS_CHOICE in freeParamCombos:

        # Create a compact label for the current set of free parameters.
        param_id_str = ''.join(f"{FREE_PARAM_IDS.get(p, p)}_" for p in FREE_PARAMS_CHOICE).rstrip('_')

        # This title is used to name the output folder for the current model variant.
        TITLE = f"n{TOTAL_N}_{MEASURE}__{param_id_str}__con{CONTRAST}"

        if FIT_BOTH_TASKS:
            TITLE += "__BOTHTASKS"

        log_anisotropies = True

        # --------------------------------------------------------------------
        # Paths
        # --------------------------------------------------------------------

        # Each model variant receives its own output directory. Existing completed
        # variants are skipped to avoid accidental overwrites.
        og_dir = os.getcwd()
        beh_data_dir = os.path.join(og_dir, 'beh_data')

        fits_dir = os.path.join(og_dir, 'customFitName', TITLE)

        if not os.path.exists(fits_dir):
            os.makedirs(fits_dir)
        else:
            print(f"Path already exists for {TITLE}. Skipping to next fit.")
            continue

        os.makedirs(fits_dir, exist_ok=True)

        # --------------------------------------------------------------------
        # Simulation and behavioural-data configuration
        # --------------------------------------------------------------------

        # config stores both simulation settings and the behavioural target values
        # against which model predictions are evaluated.
        config = {'run_type': 'loop',
                  'design': 'mocs',
                  'std_oris': [0, 45],
                  'contrasts': [CONTRAST],
                  'n_level_reps': 10,  # Set to 50 (10 for quick)
                  'step_size': 0.02,  # Orispace tiling resolution

                  # Behavioural Data
                  'OEI': {'con_det': [1.06, 1.39, 1.71], 'ori_dis': [2.45, 3.25, 3.36]},
                  'thresholds': {'con_det': [[0.459, 0.491], [1.123, 1.792], [5.154, 9.499]],
                                 'ori_dis': [[0.817, 2.033], [1.016, 3.261], [1.339, 4.049]]
                                 },
                  }
        print(f"Tiling step size = {config['step_size']}")

        # === SF conditions === #

        # Spatial-frequency conditions are labelled low, mid, and high. Each
        # condition has a different tuning width.
        cond_labels = ['low', 'mid', 'high']
        sf_params = {'fwhm': [56.2, 30.0, 22.0]}

        # --------------------------------------------------------------------
        # Free parameters
        # --------------------------------------------------------------------

        # Full menu of possible free parameters. For a given model variant, only
        # the subset in FREE_PARAMS_CHOICE is actually released.

        # Fraction parameters are cardinal:oblique ratios. They are later log
        # transformed so that reciprocal ratios are symmetric around zero.
        free_params_settings = {
            # Baseline parameters
            'total_n': {'x0': 270, 'bounds': (180, 540)},
            'rmax': {'x0': 50, 'bounds': (20, 100)},
            'noise_scale': {'x0': 1, 'bounds': (0.2, 4)},
            'crf_exponent': {'x0': 2, 'bounds': (1, 4)},
            'crf_c50': {'x0': 24, 'bounds': (10, 50)},
            'cardinal_crf_exponent': {'x0': 2, 'bounds': (1, 8)},
            'oblique_crf_exponent': {'x0': 2, 'bounds': (1, 8)},
            'cardinal_crf_c50': {'x0': 24, 'bounds': (5, 50)},
            'oblique_crf_c50': {'x0': 24, 'bounds': (5, 50)},

            # Anisotropies
            'n_fraction': {'x0': 1, 'bounds': (1/4, 4)},
            'fwhm_fraction': {'x0': 1, 'bounds': (1/4, 4)},
            'rmax_fraction': {'x0': 1, 'bounds': (1/4, 4)},
            'noise_scale_fraction': {'x0': 1, 'bounds': (1/4, 4)},

            'crf_exponent_fraction': {'x0': 1, 'bounds': (1/4, 4)},
            'crf_c50_fraction': {'x0': 1, 'bounds': (1/4, 4)},
        }

        # Select only the free parameters for the current model variant.
        free_params = {}
        for choice in FREE_PARAMS_CHOICE:
            free_params[choice] = free_params_settings[choice]

        # Optimise fraction parameters on a log scale. This makes multiplicative
        # effects easier to search and treats reciprocal ratios symmetrically.
        for param, fit_constraints in free_params.items():
            if not param.endswith(f"_fraction"):
                continue

            for k in fit_constraints:
                free_params[param][k] = np.log(free_params[param][k])

        # --------------------------------------------------------------------
        # Default model parameters
        # --------------------------------------------------------------------

        # These defaults define the baseline model. Any free parameters selected
        # above will be overwritten during optimisation.
        default_params = {
            # PARAMS
            'total_n': TOTAL_N, 'fwhm': 50, 'rmax': 60, 'noise_scale': 1,
            'spont': 1, 'crf_exponent': 2, 'crf_c50': 24,
            'pooling_q': 7, 'param_distrib': 'cos', 'pref_distrib': 'cos',
            # ANISOTROPIES
            'n_fraction': 1, 'fwhm_fraction': 1, 'rmax_fraction': 1, 'noise_scale_fraction': 1,
            'crf_exponent_fraction': 1, 'crf_c50_fraction': 1, 'spont_fraction': 1,
        }

        # --------------------------------------------------------------------
        # Average-data fits
        # --------------------------------------------------------------------

        # Fit the model to group-average behavioural data first.
        if FIT_AVERAGE is True:

            if FIT_BOTH_TASKS is False:
                print(f"\n{'=' * 60}\nBEGIN\n{'=' * 60}")

                # Separate-task mode: fit each task and decoder independently.
                for task in TASK_NAMES:
                    print(f">>>>> Parameter fits for {task}")

                    for decoder in DECODERS[task]:
                        print(f">>> Decoder = {decoder}")

                        # Create simulator for the current model variant.
                        simulator = ModelSimulator(default_params, cond_labels, sf_params, log_anisotropies=log_anisotropies)

                        # Create objective function for the current task and decoder.
                        config['decoder'] = decoder
                        objHandler = ObjectiveFunction(simulator=simulator, tasknames=task, decoders=decoder, measure=MEASURE,
                                                       config=config, error_type=ERROR_TYPE)

                        # Run the optimisation pipeline for this task/decoder pair.
                        opt_fit = optimize_parameters(objHandler, free_params, log_anisotropies, cond_labels=cond_labels,
                                                     save_dir=fits_dir)

                        z = 0
            else:
                # Joint-task mode: fit both tasks simultaneously for every pairing
                # of contrast-detection and orientation-discrimination decoders.
                all_decoder_pairs = list(itls.product(*[DECODERS[_t] for _t in TASK_NAMES]))  # Pairings of decoders for each task

                for decoder_pair in all_decoder_pairs:

                    # Create simulator for the current model variant.
                    simulator = ModelSimulator(default_params, cond_labels, sf_params, log_anisotropies=log_anisotropies)

                    # Create objective function spanning both tasks.
                    objHandler = ObjectiveFunction(simulator=simulator, tasknames=TASK_NAMES, decoders=decoder_pair,
                                                   measure=MEASURE, config=config, error_type=ERROR_TYPE)

                    # Run the optimisation pipeline for this decoder pairing.
                    optimize_parameters(objHandler, free_params, log_anisotropies, cond_labels=cond_labels,
                                        save_dir=fits_dir)

        # --------------------------------------------------------------------
        # Individual-participant fits
        # --------------------------------------------------------------------

        # Repeat the same fitting logic at participant level. The configuration is
        # copied and then updated with each participant's own behavioural scores.
        if FIT_INDIVIDUAL is True:

            # Load participant-level OEI data for each task.
            beh_oei_scores = []

            for task in TASK_NAMES:
                i_df = pd.read_csv(os.path.join(beh_data_dir, f'ratios_{task}.csv'))
                i_df['task'] = task  # Assign task label (reason for this weird bit)
                beh_oei_scores.append(i_df)

            beh_oei_scores = pd.concat(beh_oei_scores)
            beh_oei_scores = beh_oei_scores[beh_oei_scores['ratio'] == 'OEI']

            # Copy the group-level config so participant-specific target scores
            # can be inserted without modifying the original config object.
            indiv_config = copy.deepcopy(config)

            for pname in beh_oei_scores['participant'].unique():
                print(f">>>>> Parameter fits for {pname}")

                # Select behavioural data for this participant.
                p_oei_scores = beh_oei_scores[beh_oei_scores['participant'] == pname]

                # Insert this participant's OEI scores into the config.
                for task in TASK_NAMES:
                    indiv_config['OEI'][task] = p_oei_scores[p_oei_scores['task'] == task]['score'].to_list()

                # Deal with varying numbers of completed conditions across
                # participants. Only include the spatial-frequency conditions that
                # this participant actually completed.
                sf_to_idx_dict = {2: 0, 8: 1, 16: 2}

                idcs = [sf_to_idx_dict[_sf] for _sf in p_oei_scores['spatial frequency'].unique()]

                p_cond_labels = [cond_labels[_i] for _i in idcs]

                p_sf_params = sf_params.copy()
                for k in p_sf_params:
                    p_sf_params[k] = [p_sf_params[k][_i] for _i in idcs]

                # Create this participant's output folder.
                pname_fit_dir = os.path.join(fits_dir, 'indiv', pname)  # PATH

                if not os.path.exists(pname_fit_dir):
                    os.makedirs(pname_fit_dir)

                if FIT_BOTH_TASKS is False:

                    # Separate-task mode: fit each task and decoder independently
                    # for this participant.
                    for task in TASK_NAMES:
                        for decoder in DECODERS[task]:

                            # Create participant-specific simulator.
                            simulator = ModelSimulator(default_params, p_cond_labels, p_sf_params,
                                                       log_anisotropies=log_anisotropies)

                            # Create participant-specific objective function.
                            indiv_config['decoder'] = decoder
                            objHandler = ObjectiveFunction(simulator=simulator, tasknames=task, decoders=decoder,
                                                           measure=MEASURE,
                                                           config=indiv_config, error_type=ERROR_TYPE)

                            # Run optimisation for this participant, task, and decoder.
                            optimize_parameters(objHandler, free_params, log_anisotropies, cond_labels=p_cond_labels,
                                                save_dir=pname_fit_dir)

                else:
                    # Joint-task participant-level fitting. Each decoder pairing is
                    # fitted to both tasks simultaneously.
                    DECODERS = list(itls.product(*[DECODERS[_t] for _t in TASK_NAMES]))  # Pairings of decoders for each task

                    for decoder_pair in DECODERS:

                        # Create simulator
                        simulator = ModelSimulator(default_params, cond_labels, sf_params,
                                                   log_anisotropies=log_anisotropies)

                        # Create objective function spanning both tasks.
                        objHandler = ObjectiveFunction(simulator=simulator, tasknames=TASK_NAMES, decoders=decoder_pair,
                                                       measure=MEASURE, config=indiv_config, error_type=ERROR_TYPE)

                        # Run optimisation for this participant and decoder pairing.
                        optimize_parameters(objHandler, free_params, log_anisotropies, cond_labels=cond_labels,
                                            save_dir=pname_fit_dir)

