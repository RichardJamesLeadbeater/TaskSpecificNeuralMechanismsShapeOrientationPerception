"""
Parameter fitting for the population-response coding model.

The model is fitted to contrast-detection and orientation-discrimination
measurements using differential evolution. Candidate solutions are
re-evaluated with additional simulations before final parameter estimates
and model predictions are saved.

Fraction parameters describe cardinal-to-oblique ratios and are optimised
on a logarithmic scale.
"""

import copy
from copy import deepcopy
import itertools as itls
import os
from multiprocessing import Pool
import numpy as np
import random
from numpy import asarray as arr
import pandas as pd
from scipy.optimize import differential_evolution
import time
import uuid

# Project-specific analysis utilities
from mytools.analysis import get_ave_data
from mytools.dframe_tools import custom_sort

# Population-model simulator
from components.model_simulation import ModelSimulator

# ===
# Global variables
# ===
# Candidate model outputs collected during optimisation
# This is reset at each stage so that each step only retains relevant fits
all_fits_list = []


def run_de(objH, param_names, bounds, popsize, maxiter, n_level_reps, seed=None):
    """Run one differential-evolution search and return all evaluated fits."""
    de_result = differential_evolution(
        lambda x: run_objective_fn(
            x,
            param_names,
            objH,
            n_eval=1,
            n_level_reps=n_level_reps,
            append_all_fits=True,
            seed=seed,
        ),
        bounds=bounds,
        popsize=popsize,
        maxiter=maxiter,
        polish=False,
        seed=seed,
    )

    opt_fit = dict(zip(param_names, de_result.x))
    opt_fit = {
        name: np.round(
            np.exp(value) if name.endswith("_fraction") else value,
            5,
        )
        for name, value in opt_fit.items()
    }

    print(f"Best fitting parameters: {opt_fit}")
    print(f"Best fitting objective (i.e. error): {de_result.fun}")
    print(f"{'=' * 60}\n")

    global all_fits_list

    for fit in all_fits_list:
        fit["seed"] = seed

    return {
        "diff_evol": de_result,
        "all_fits": all_fits_list,
        "seed": seed,
    }


def run_objective_fn(
    param_values,
    param_names,
    objH,
    n_eval,
    n_level_reps=None,
    append_all_fits=False,
    _return_output=False,
    seed=None,
):
    """Evaluate a parameter vector and return its error or full model output."""
    if n_level_reps is not None:
        objH.config["n_level_reps"] = n_level_reps

    model_output, _ = objH.run(
        param_values,
        param_names,
        n_eval,
        seed,
    )

    total_error = model_output["total_error"].iloc[0]
    print(f"Error = {total_error:.4f}\n")

    model_output["fit_id"] = uuid.uuid4().hex

    if append_all_fits:
        global all_fits_list
        all_fits_list.append(model_output)

    if _return_output:
        return model_output

    return total_error


class ObjectiveFunction:
    """Evaluate model predictions against behavioural target data."""
    def __init__(self, simulator, tasknames, decoders, measure, config, error_type='mse', task_weights=None,
                 PRINTS=True):
        self.simulator = simulator
        self.tasknames = [tasknames] if not isinstance(tasknames, (list, tuple)) else tasknames
        self.decoders = [decoders] if not isinstance(decoders, (list, tuple)) else decoders
        self.measure = measure
        self.config = config
        self.error_type = error_type
        self.task_weights = task_weights or {_t: 1.0 for _t in self.tasknames}
        self.PRINTS = PRINTS

    def run(self, param_values, param_names, n_eval=1, seed=None):
        """
        Simulate each task and decoder, then calculate the weighted fitting error.

        Returns the combined model output and the evaluated parameter values.
        """

        # Map the optimiser vector onto named simulator parameters.
        var_params = {name: value for name, value in zip(param_names, param_values)}

        total_error = 0
        all_model_scores = {}
        all_target_scores = {}
        error_output = {}

        all_model_output = []

        if self.PRINTS is True:
            run_label = "\n".join([f"{_t} - {_d}" for _t, _d in zip(self.tasknames, self.decoders)])
            print(run_label)

        # Evaluate each task and decoder included in the fit.
        for i_task, i_decoder in zip(self.tasknames, self.decoders):
            # Run simulation
            tic = time.time()

            n_repeats_placeholder = 1
            args_list = [
                (i_task, i_decoder, self.measure, var_params, self.config, n_eval)
                for _ in range(n_repeats_placeholder)
            ]

            results = [self.simulator.simulate(*_args) for _args in args_list]

            # Combine repeated simulations into a single output table.
            model_output = pd.concat([pd.DataFrame(_r) for _r in results])

            # Record the free parameters defining this model variant.
            for fp_idx, free_param in enumerate(param_names):
                model_output[f"freeParam_{fp_idx}"] = free_param

            print(f"\n##### > {time.time() - tic:.3f}s < #####")

            # Convert model output into the same vector format as the behavioural
            # target data before computing the error.
            if self.measure == 'thresholds':
                model_scores = []
                for c, o in zip(model_output['cardinalT'], model_output['obliqueT']):
                    model_scores = [*model_scores, c, o]
            else:
                model_scores = arr(model_output[self.measure])

            # Target observations for the current task.
            target_scores = np.hstack(self.config[self.measure][i_task])  # Behavioural data

            # Calculate error in logarithmic or linear space.
            if self.error_type == 'log_mse':
                task_error = np.mean((np.log(model_scores) - np.log(target_scores)) ** 2)
            else:
                task_error = np.mean((model_scores - target_scores) ** 2)

            # Apply the task weight to the joint objective.
            weighted_error = task_error * self.task_weights[i_task]
            total_error += weighted_error

            if self.PRINTS is True:
                print(f"  Target: {np.round(target_scores, 3)}")
                print(f"  Model:  {np.round(model_scores, 3)}")

            all_model_scores[i_task] = model_scores
            all_target_scores[i_task] = target_scores
            error_output[i_task] = task_error

            # Retain task-level error for subsequent ranking and reporting.
            model_output['task_fit_error'] = task_error

            all_model_output.append(model_output)

        if self.PRINTS is True:
            if total_error > 50:
                print(f"{''*20}\nEXTREME ERROR\n{''*20}")
                z = 0

        all_model_output = pd.concat(all_model_output)

        # Store the objective value used during optimisation.
        all_model_output['total_error'] = total_error

        return all_model_output, var_params


def get_fit_ranking(df, param_names, error_col='total_error'):
    """Return rank and error mappings for unique parameter sets."""
    ranked = df.sort_values(error_col).drop_duplicates(param_names).reset_index(drop=True)
    keys = ranked[param_names].apply(tuple, axis=1)
    return dict(zip(keys, range(1, len(ranked) + 1))), dict(zip(keys, ranked[error_col]))


def add_pre_rerun_ranking(df, param_names, rank_pre, error_pre):
    """Attach ranks and errors obtained before candidate re-evaluation."""
    keys = df[param_names].apply(tuple, axis=1)
    df['rank_pre_rerun'] = keys.map(rank_pre)
    df['total_error_pre_rerun'] = keys.map(error_pre)
    return df


def add_post_rerun_ranking(df, param_names):
    """Attach post-evaluation ranks and changes in rank and error."""
    rank_after, _ = get_fit_ranking(df, param_names)
    keys = df[param_names].apply(tuple, axis=1)
    df['rank_post_rerun'] = keys.map(rank_after)
    df['rank_change'] = df['rank_post_rerun'] - df['rank_pre_rerun']
    df['total_error_pct_change'] = (
        (df['total_error'] - df['total_error_pre_rerun'])
        / df['total_error_pre_rerun']
    ) * 100
    return df


def rerun_fit(row, param_names, objH, n_level_reps_topFits, seed=None):
    """Run the final simulation using median parameters from selected fits."""
    # Transform fraction parameters back to the optimiser's logarithmic scale.
    vals = row[param_names].tolist()
    for i, label in enumerate(param_names):
        if label.endswith('_fraction'):
            vals[i] = np.log(vals[i])

    # Re-evaluate the candidate with additional simulation repetitions.
    _ = run_objective_fn(vals, param_names, objH, n_eval=1,
                         n_level_reps=n_level_reps_topFits, append_all_fits=True, seed=seed)

    global all_fits_list
    return all_fits_list


def run_final_fit(param_names, objH, de_seed, final_top_fits, n_eval, n_level_reps, master_seed):
    """Run the final simulation using median parameters from selected fits."""
    # Use median values across the retained candidate fits.
    final_param_vals = final_top_fits[param_names].median().to_list()

    # Transform fraction parameters to the optimiser's logarithmic scale.
    for idx, param_name in enumerate(param_names):
        if param_name.endswith('_fraction'):
            final_param_vals[idx] = np.log(final_param_vals[idx])

    # Generate the final, higher-precision model prediction.
    final_output = run_objective_fn(final_param_vals, param_names, objH, n_eval=n_eval,
                                    n_level_reps=n_level_reps,
                                    append_all_fits=False, _return_output=True, seed=master_seed)
    final_output['de_seed'] = de_seed

    return final_output


def optimize_parameters(objH, free_params, log_fractions=True, _noise_test=False, cond_labels=None,
                        save_dir=''):
    """
    Fit model parameters and save optimised candidates, save final model outputs.

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

    # Preserve a consistent ordering of parameter names, values and bounds.
    param_names = list(free_params.keys())
    initial_values = [free_params[k]['x0'] for k in param_names]
    bounds = [free_params[k]['bounds'] for k in param_names]

    if PRINTS is True:
        print(f"\n{'=' * 60}\nStarting optimization with differential evolution (DE)\n{'=' * 60}")

    # Build a filename that records which parameters were free in this fit
    param_id_str = ''.join(f"{FREE_PARAM_IDS.get(p, p)}_" for p in FREE_PARAMS_CHOICE).rstrip('_')
    filename = f"{objH.tasknames}_{objH.decoders}_{param_id_str}"

    # -------------------------------------------------------------------------
    # 1. Run independent differential-evolution searches
    # -------------------------------------------------------------------------
    # Reset the candidate-output store before optimisation.
    global all_fits_list
    all_fits_list = []

    # Optimisation settings
    de_config = objH.config['de_config']
    N_runs = de_config['N_SEEDS']
    popsize = de_config['popsize']
    maxiter = de_config['maxiter']
    n_level_reps = de_config['n_level_reps']

    # Each mp worker runs one independent differential-evolution search.
    USE_MULTIPROCESSING = False
    NPOOL = 4

    # Generate reproducible, non-overlapping seeds for independent searches.
    rng = np.random.default_rng(MASTER_SEED)
    de_seeds = rng.choice(1_000_000, size=N_runs, replace=False)

    print(f"============ DIFFERENTIAL EVOLUTION ============")
    print(f"DE seeds: {de_seeds.tolist()}")

    diffEvol_filepath = os.path.join(save_dir, f'diffEvol-{filename}.csv')
    if not os.path.exists(diffEvol_filepath):

        # Prepare arguments for multiprocessing. Each worker runs one independent DE search.
        args_list = [(objH, param_names, bounds, popsize, maxiter, n_level_reps, seed) for seed in de_seeds]

        if USE_MULTIPROCESSING and DE_CONFIG['N_SEEDS'] > 1:
            with Pool(NPOOL) as pool:
                all_de_results = pool.starmap(run_de, args_list)
        else:
            all_de_results = [run_de(*args) for args in args_list]

        # Combine candidate evaluations across all independent searches.
        all_fits = []
        for de_run in all_de_results:
            all_fits.extend(de_run['all_fits'])

        # Gather every candidate evaluation from all DE runs into one table.
        all_fits = pd.concat(all_fits).round(6)
        all_fits.to_csv(diffEvol_filepath, index=False)
    else:
        print(f'Found diffEvol file. Skipping.')
        all_fits = pd.read_csv(diffEvol_filepath)

    # Process each optimisation seed separately to preserve distinct solution sets.
    all_top_fits = []
    all_repro = []
    all_top_summary = []
    print(f"============ TOP FITS ============")
    topFits_filepath = os.path.join(save_dir, f'topFits-{filename}.csv')
    if not os.path.exists(topFits_filepath):

        for de_seed, seed_fits in all_fits.groupby('seed'):
            # -------------------------------------------------------------------------
            # 2. Select candidate fits for higher-precision evaluation
            # -------------------------------------------------------------------------
            print(f"{'=' * 30}\nSTEP 2. Pool top fits\n{'=' * 30}")

            # Select the strongest unique candidates from the current seed.
            n_rerun = 100 * len(cond_labels) * len(objH.tasknames)  # Conds repeated across 3 SF conditions & 2 tasks

            # Sort by error and keep the best unique candidates to rerun more accurately
            top_fits_pre = seed_fits.sort_values('total_error').head(n_rerun)
            top_fits_pre[param_names] = top_fits_pre[param_names].round(6)

            # Save the pre-rerun ranking and error for each unique parameter set.
            rank_pre, error_pre = get_fit_ranking(top_fits_pre, param_names)

            # -------------------------------------------------------------------------
            # 3. Re-evaluate selected candidates
            # -------------------------------------------------------------------------
            # Use additional simulation repetitions to obtain more stable error estimates.
            print(f"{'=' * 30}\nSTEP 3. Re-run top 5%\n{'=' * 30}")

            # Reset the store so that it contains only the re-evaluated candidates.
            all_fits_list = []

            n_level_reps_topFits = de_config['n_level_reps_topFits']

            rows = [row for _, row in top_fits_pre.drop_duplicates(param_names).iterrows()]

            if USE_MULTIPROCESSING and DE_CONFIG['N_SEEDS'] > 1:
                with Pool(NPOOL) as pool:
                    rerun_fits = pool.starmap(
                        rerun_fit,
                        [(row, param_names, objH, n_level_reps_topFits, MASTER_SEED) for row in rows]
                    )
            else:
                rerun_fits = [rerun_fit(row, param_names, objH, n_level_reps_topFits, MASTER_SEED) for row in rows]

            # Combine results returned by the worker processes.
            all_rerun_fits = []
            for i_fits in rerun_fits:
                all_rerun_fits.extend(i_fits)

            top_fits_rerun = pd.concat(all_rerun_fits)
            top_fits_rerun = top_fits_rerun.drop_duplicates(subset=[*param_names, 'cond'])

            # -------------------------------------------------------------------------
            # 4. Retain the strongest re-evaluated candidates
            # -------------------------------------------------------------------------
            print(f"{'=' * 30}\nSTEP 4. Keep best 50% of re-evaluated parameter sets\n{'=' * 30}")
            top_fits_rerun[param_names] = top_fits_rerun[param_names].round(6)

            # Compare candidate rankings before and after re-evaluation.
            top_fits_rerun = add_pre_rerun_ranking(top_fits_rerun, param_names, rank_pre, error_pre)
            top_fits_rerun = add_post_rerun_ranking(top_fits_rerun, param_names)

            # Retain the best half of the re-evaluated parameter sets.
            n_keep = max(
                len(cond_labels) * len(objH.tasknames),
                int(np.ceil(0.5 * len(top_fits_rerun)))
            )
            very_top_fits = top_fits_rerun.copy().sort_values('total_error').head(n_keep)
            very_top_fits['de_seed'] = de_seed
            all_top_fits.append(very_top_fits)

            # Summarise retained fits by task, decoder and condition.
            numeric_cols = very_top_fits.select_dtypes(include="number").columns
            measure_order = ['median', 'q25', 'q75', 'mean', 'std', 'sem']

            top_summary = get_ave_data(very_top_fits, ['task', 'decoder', 'cond'], numeric_cols)
            top_summary = custom_sort(
                top_summary,
                ['measure', 'task', 'decoder', 'cond'],
                [measure_order, ['con_det', 'ori_dis'], [], ['low', 'mid', 'high']]
            )
            top_summary['de_seed'] = de_seed
            all_top_summary.append(top_summary)

        # Save summary statistics for the selected top fits.
        all_top_summary = pd.concat(all_top_summary).round(5)
        all_top_summary.to_csv(os.path.join(save_dir, f'summaryTopFits-{filename}.csv'), index=False)

        # Save
        all_top_fits = pd.concat(all_top_fits)
        all_top_fits.to_csv(topFits_filepath, index=False)
    else:
        print(f'Found topFits file. Skipping.')
        all_top_fits = pd.read_csv(topFits_filepath)

    # -------------------------------------------------------------------------
    # 5. Generate final parameter estimates and model predictions
    # -------------------------------------------------------------------------
    print(f"============ OPT FIT ============")
    # Run a final fit separately for each optimisation seed.
    optFit_filepath = os.path.join(save_dir, f'optFit-{filename}.csv')
    if not os.path.exists(optFit_filepath):

        args = []
        n_eval_final = 10
        n_level_reps_final = 500

        for de_seed, seed_fits in all_top_fits.groupby('de_seed'):
            args.append((param_names, objH, de_seed, seed_fits, n_eval_final, n_level_reps_final, MASTER_SEED))

        if USE_MULTIPROCESSING and DE_CONFIG['N_SEEDS'] > 1:
            with Pool(NPOOL) as pool:
                all_opt_fits = pool.starmap(run_final_fit, args)
        else:
            all_opt_fits = [run_final_fit(*i_args) for i_args in args]

        # Generate model predictions and orientation-reproduction diagnostics.
        for i in all_opt_fits:
            # Initialise the model using the fitted parameter values.
            i_param_vals = i.iloc[0][param_names].drop_duplicates().tolist()

            for idx, param_name in enumerate(param_names):
                if param_name.endswith('_fraction'):
                    i_param_vals[idx] = np.log(i_param_vals[idx])

            run_objective_fn(
                i_param_vals, param_names, objH, n_eval=1, n_level_reps=10,
                append_all_fits=False, _return_output=True, seed=MASTER_SEED
            )

            # Evaluate orientation reproduction across the sampled orientations.
            ori_x = np.linspace(0, 45, 11)
            ori_repro = objH.simulator.model.ori_reproduction_check(
                stim_oris=ori_x, show_plots=False, prior=False, save_path=None
            )

            # Summarise reproduction responses for each stimulus orientation.
            ori_repro_df = pd.DataFrame({
                'x': ori_x,
                'mean': np.mean(ori_repro, axis=0),
                'std': np.std(ori_repro, axis=0),
                'de_seed': i['de_seed'].iloc[0],
            })

            all_repro.append(ori_repro_df)

        all_repro = pd.concat(all_repro)
        all_repro.to_csv(os.path.join(save_dir, f'repro-{filename}.csv'), index=False)

        # Save final fits and diagnostic outputs.
        all_opt_fits = pd.concat(all_opt_fits)
        all_opt_fits['n_eval'] = n_eval_final
        all_opt_fits['n_level_reps_final'] = n_level_reps_final
        all_opt_fits.to_csv(optFit_filepath, index=False)

    else:
        print(f'Found optFit file. Skipping.')

    return None


def fit_already_completed(save_dir, tasknames, decoders, free_params):
    """Return whether the final output file exists for this model variant."""
    param_id_str = ''.join(f"{FREE_PARAM_IDS.get(p, p)}_" for p in free_params).rstrip('_')

    task_label = f"{[str(x) for x in np.atleast_1d(tasknames)]}"

    if len(np.atleast_1d(decoders)) == 1:
        decoder_label = f"{[str(x) for x in np.atleast_1d(decoders)]}"
    else:
        decoder_label = f"{tuple(map(str, np.atleast_1d(decoders)))}"

    filename = f"{task_label}_{decoder_label}_{param_id_str}"

    file_exists = False
    if os.path.isfile(os.path.join(save_dir, f"optFit-{filename}.csv")):
        file_exists = True

    return file_exists

# =========================================================================
# Main analysis
# =========================================================================

if __name__ == '__main__':

    # Reduced settings for testing the analysis pipeline.
    DE_CONFIG = {
        # Computationally reduced configuration
        'N_SEEDS': 4,
        'popsize': 2,
        'maxiter': 5,
        'n_level_reps': 5,
        'n_level_reps_topFits': 5,

        # Configuration used for the reported model fits
        # 'N_SEEDS': 50,
        # 'popsize': 12,
        # 'maxiter': 10,
        # 'n_level_reps': 50,
        # 'n_level_reps_topFits': 250,
        # 'master_seed': 101,
    }

    MASTER_SEED = DE_CONFIG.get('master_seed', 789)

    repro_config = {
        'master_seed': MASTER_SEED,
        'numpy_version': np.__version__,
    }

    # Model variants defined by fitting mode and correlated-noise parameters.
    sim_conds = [
        # Single-task fits to average data only without correlated noise
        {'fit_ave': True, 'fit_indiv': False, 'fit_both': False, 'corrNz': True, 'corrMax': 0.3, 'corrWidth': 22.5},
        # Shared both-task fits to average data only without correlated noise
        {'fit_ave': True, 'fit_indiv': False, 'fit_both': True, 'corrNz': True, 'corrMax': 0.3, 'corrWidth': 22.5},

        # Single-task fits to average data only with defined correlated noise
        {'fit_ave': True, 'fit_indiv': False, 'fit_both': False, 'corrNz': False},
        # Shared both-task fits to average data only with defined correlated noise
        {'fit_ave': True, 'fit_indiv': False, 'fit_both': True, 'corrNz': False},

        # Single-task fits to each participant and average data without correlated noise
        {'fit_ave': True, 'fit_indiv': True, 'fit_both': False, 'corrNz': False},

        # ...
    ]

    for sim_cond in sim_conds:

        print(sim_cond)

        # Configuration
        DEBUG = False
        PRINTS = False

        DECODERS = {'con_det': ['sum', 'mag', 'peak'],
                    'ori_dis': ['PV', 'ML', 'WTA']}
        TASK_NAMES = ['con_det', 'ori_dis']  # which tasks to optimise ['con_det', 'ori_dis']

        FIT_AVERAGE = sim_cond.get('fit_ave', True)
        FIT_BOTH_TASKS = sim_cond['fit_both']
        FIT_INDIVIDUAL = sim_cond['fit_indiv']

        MEASURE = 'OEI'  # 'OEI' or 'thresholds'

        # Contrast used for orientation-discrimination simulations
        CONTRAST = 100

        TOTAL_N = 90  # 270 for paper replication; 90 for demonstrations to run quickly
        TUNING_TYPE = 'von_mises'  # 'von_mises' 'wrapped_gaussian'

        CORRELATED_NOISE = sim_cond['corrNz']
        CORR_MAX = sim_cond.get('corrMax')
        CORR_WIDTH = sim_cond.get('corrWidth')  # 45 should drop to 0 correlation at 90 deg; [HWHM = corr_width * np.sqrt(np.log(2))]

        ERROR_TYPE = 'log_mse'

        FREE_PARAM_IDS = {
            'n_fraction': 'Nf',
            'fwhm_fraction': 'Hf',
            'noise_scale_fraction': 'NZf',
            'rmax_fraction': 'Rf'
        }

        # Define the anisotropy parameters included in each fitted model.
        allFreeParams = ['n_fraction', 'fwhm_fraction', 'noise_scale_fraction', 'rmax_fraction']

        # Optionally generate every combination subset of anisotropy parameters.
        ALL_COMBINATIONS_FREE_PARAMS = False
        if ALL_COMBINATIONS_FREE_PARAMS is True:
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
            TITLE = f"n{TOTAL_N}_{MEASURE[:2]}_{param_id_str}_c{CONTRAST}"

            if FIT_BOTH_TASKS:
                TITLE += "_BOTH"

            if CORRELATED_NOISE is True:
                TITLE += f"_cr{CORR_MAX}x{CORR_WIDTH}"

            log_anisotropies = True  # Fractions will be log-transformed

            # Output paths
            og_dir = os.getcwd()
            beh_data_dir = os.path.join(og_dir, 'beh_data')

            fits_dir = os.path.join(
                og_dir,
                'JoV_fits',
                f"deN{DE_CONFIG['N_SEEDS']}_p{DE_CONFIG['popsize']}_i{DE_CONFIG['maxiter']}_MS{MASTER_SEED}",
                TITLE
            )

            if not os.path.exists(fits_dir):
                os.makedirs(fits_dir)

            # Simulation settings and behavioural target values
            config = {
                'run_type': 'loop',
                'design': 'mocs',
                'std_oris': [0, 45],
                'contrasts': [CONTRAST],

                # Orientation space tiling
                'min_tile': -90,
                'max_tile': 90,
                'step_size': 0.025,  # 0.0333333

                # Whether to constrain window during decoding (speeds up ML estimation)
                'window': True,

                # Diff evol settings
                'de_config': DE_CONFIG,
                'n_level_reps': DE_CONFIG['n_level_reps'],

                # Behavioural Data
                'OEI': {
                    'con_det': [1.06, 1.39, 1.71],
                    'ori_dis': [2.45, 3.25, 3.36]
                },
                'thresholds': {
                    'con_det': [[0.459, 0.491], [1.123, 1.792], [5.154, 9.499]],
                    'ori_dis': [[0.817, 2.033], [1.016, 3.261], [1.339, 4.049]]
                },
            }
            print(f"Tiling step size = {config['step_size']}")

            # Tuning widths for the three spatial-frequency conditions.
            cond_labels = ['low', 'mid', 'high']
            sf_params = {'fwhm': [56.2, 30.0, 22.0]}

            # --------------------------------------------------------------------
            # Free parameters
            # --------------------------------------------------------------------
            # Candidate parameter values and optimisation bounds

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

                # Correlated noise
                'corr_max': {'x0': 0.15, 'bounds': (0, 1)},
                'corr_width': {'x0': 30, 'bounds': (5, 50)},
            }

            # Retain only parameters included in the current model variant.
            free_params = {}
            for choice in FREE_PARAMS_CHOICE:
                free_params[choice] = free_params_settings[choice]

            # Optimise fraction parameters on a log scale.
            for param, fit_constraints in free_params.items():
                if not param.endswith(f"_fraction"):
                    continue

                for k in fit_constraints:
                    free_params[param][k] = np.log(free_params[param][k])

            # Default model parameters are overwritten by free params.
            default_params = {
                # PARAMS
                'total_n': TOTAL_N,
                'fwhm': 56.2,
                'rmax': 60,
                'noise_scale': 1,
                'spont': 1,
                'crf_exponent': 2,
                'crf_c50': 24,
                'pooling_q': 7,
                'param_distrib': 'cos',
                'pref_distrib': 'cos',

                # ANISOTROPIES
                'n_fraction': 1,
                'fwhm_fraction': 1,
                'rmax_fraction': 1,
                'noise_scale_fraction': 1,
                'crf_exponent_fraction': 1,
                'crf_c50_fraction': 1,
                'spont_fraction': 1,

                # Correlated noise
                'correlated_noise': CORRELATED_NOISE,
                'corr_max': CORR_MAX,
                'corr_width': CORR_WIDTH
            }

            # -------------------------------------------------------------------------
            # Group-average fits
            # -------------------------------------------------------------------------
            if FIT_AVERAGE is True:

                if FIT_BOTH_TASKS is False:
                    print(f"\n{'=' * 60}\nBEGIN\n{'=' * 60}")

                    # Separate-task mode: fit each task and decoder independently.
                    for task in TASK_NAMES:
                        print(f">>>>> Parameter fits for {task}")

                        for decoder in DECODERS[task]:
                            print(f">>> Decoder = {decoder}")

                            i_config = copy.deepcopy(config)

                            # Create simulator for the current model variant.
                            simulator = ModelSimulator(default_params, cond_labels, sf_params,
                                                       log_anisotropies=log_anisotropies)

                            # Create objective function for the current task and decoder.
                            i_config['decoder'] = decoder
                            objHandler = ObjectiveFunction(simulator=simulator, tasknames=task, decoders=decoder,
                                                           measure=MEASURE,
                                                           config=i_config, error_type=ERROR_TYPE)

                            # Run the optimisation pipeline for this task/decoder pair.
                            optimize_parameters(objHandler, free_params, log_anisotropies, cond_labels=cond_labels,
                                                save_dir=fits_dir)

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

            # -------------------------------------------------------------------------
            # Individual-participant fits
            # -------------------------------------------------------------------------
            if FIT_INDIVIDUAL is True:

                # Load participant-level behavioural measurements.
                beh_oei_scores = []

                for task in TASK_NAMES:
                    i_df = pd.read_csv(os.path.join(beh_data_dir, f'ratios_{task}.csv'))
                    i_df['task'] = task  # Assign task label (reason for this weird bit)
                    beh_oei_scores.append(i_df)

                beh_oei_scores = pd.concat(beh_oei_scores)
                beh_oei_scores = beh_oei_scores[beh_oei_scores['ratio'] == 'OEI']

                indiv_config = copy.deepcopy(config)

                for pname in beh_oei_scores['participant'].unique():
                    print(f">>>>> Parameter fits for {pname}")

                    # Select behavioural data for this participant.
                    p_oei_scores = beh_oei_scores[beh_oei_scores['participant'] == pname]

                    # Insert this participant's OEI scores into the config.
                    for task in TASK_NAMES:
                        indiv_config['OEI'][task] = p_oei_scores[p_oei_scores['task'] == task]['score'].to_list()

                    # Retain only conditions completed by the current participant.
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

                                i_indiv_config = copy.deepcopy(indiv_config)

                                # Create participant-specific simulator.
                                simulator = ModelSimulator(default_params, p_cond_labels, p_sf_params,
                                                           log_anisotropies=log_anisotropies)

                                # Create participant-specific objective function.
                                i_indiv_config['decoder'] = decoder
                                objHandler = ObjectiveFunction(simulator=simulator, tasknames=task, decoders=decoder,
                                                               measure=MEASURE,
                                                               config=i_indiv_config, error_type=ERROR_TYPE)

                                # Run optimisation for this participant, task, and decoder.
                                optimize_parameters(objHandler, free_params, log_anisotropies, cond_labels=p_cond_labels,
                                                    save_dir=pname_fit_dir)

                    else:
                        # Each decoder pairing fitted to both tasks simultaneously.
                        all_decoder_pairs = list(
                            itls.product(*[DECODERS[_t] for _t in TASK_NAMES])
                        )

                        for decoder_pair in all_decoder_pairs:

                            # Create simulator
                            simulator = ModelSimulator(default_params, p_cond_labels, p_sf_params,
                                                       log_anisotropies=log_anisotropies,
                                                       )

                            # Create objective function spanning both tasks.
                            objHandler = ObjectiveFunction(simulator=simulator, tasknames=TASK_NAMES, decoders=decoder_pair,
                                                           measure=MEASURE, config=indiv_config, error_type=ERROR_TYPE)

                            # Run optimisation for this participant and decoder pairing.
                            optimize_parameters(objHandler, free_params, log_anisotropies, cond_labels=p_cond_labels,
                                                save_dir=pname_fit_dir)

