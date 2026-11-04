import copy
import numpy as np
import pandas as pd
from mytools.dict_tools import append_dicty

# Import model components
from components.run import mocs_task, run
from components.population_code import Params, Tiling, gen_population_code
from mytools.dframe_tools import (to_csv_pkl, index_dframe, init_cols, custom_sort)


# ============================================================================
# TASK EXECUTION - Run experiments with the model
# ============================================================================
def run_orientation_discrimination(model, std_oris, decoder, contrasts,
                                   run_type, n_level_reps, design='mocs', **kwargs):
    """Run orientation discrimination task."""
    # Define levels of orientation difference
    if decoder == 'WTA':
        # Less accurate & very quick compared to others
        levels = np.hstack([
            np.linspace(0.1, 1, 5),
            np.linspace(1, 10, 6),  # Crucial range
            np.linspace(10, 40, 3)
        ]
        )
    else:
        # More accurate decoders
        levels = np.hstack([
            np.linspace(0.01, 0.1, 6),
            np.linspace(0.1, 1, 6),  # Crucial range
            np.linspace(1, 10, 6),
            np.linspace(10, 40, 3)
        ]
        )

    levels = np.unique(levels)  # Remove repeats
    levels = np.hstack([[0, np.min(levels) / 4, np.min(levels) / 2], levels])

    if design == 'mocs':
        df = run(
            experiment=mocs_task, model=model, decoders=decoder, stdoris=std_oris, contrasts=contrasts,
            run_type_=run_type, n_level_reps=n_level_reps, levels=levels, task='ori_dis', **kwargs
        )
    else:
        raise ValueError(f"Unknown design: {design}")

    return df


def run_contrast_detection(model, std_oris, decoder, run_type, n_level_reps, **kwargs):
    """Run contrast detection task."""
    levels = np.hstack([
        np.linspace(0.1, 1, 7),
        np.linspace(1, 10, 7),
        np.linspace(10, 30, 3),
    ])
    levels = np.unique(levels)  # Remove repeats

    levels = np.hstack([[0], levels])

    df = run(
        experiment=mocs_task, model=model, decoders=decoder,
        stdoris=std_oris, contrasts=[None],
        run_type_=run_type, n_level_reps=n_level_reps,
        levels=levels, task='con_det', **kwargs
    )
    return df


# ============================================================================
# PARAMETER MANAGEMENT - Handle model parameter configurations
# ============================================================================

class ParameterManager:
    """Manages parameter transformations and validations."""

    @staticmethod
    def apply_shared_params(params, shared_param_names):
        """Apply shared parameters to both cardinal and oblique populations."""
        for param_name in shared_param_names:
            if param_name in params:
                params[f"cardinal_{param_name}"] = params[param_name]
                params[f"oblique_{param_name}"] = params[param_name]
        return params

    @staticmethod
    def apply_anisotropy_fractions(params):
        """Apply anisotropies."""
        # === Number of neurons === #
        n_oblique = params['total_n'] / (params['n_fraction'] + 1)
        n_cardinal = params['total_n'] - n_oblique
        params['oblique_n_prefs'] = int(np.round(n_oblique))
        params['cardinal_n_prefs'] = int(np.round(n_cardinal))

        # If fraction is defined then define parameter for cardinal and obliques
        for fraction_str in [_k for _k in params.keys() if _k != 'n_fraction']:
            if fraction_str.endswith('_fraction'):
                lbl = fraction_str.split('_fraction')[0]
                oblique_param_val = params[lbl]
                params[f"oblique_{lbl}"] = oblique_param_val
                if params.get(f"cardinal_{lbl}"):  # If already defined
                    continue
                params[f"cardinal_{lbl}"] = oblique_param_val * params[fraction_str]

        return params


class ModelSimulator:
    """Handles model creation and simulation."""

    def __init__(self, default_params, sf_labels=None, sf_params=None, log_anisotropies=True, PRINTS=False):
        self.model = None
        self.default_params = default_params
        self.sf_labels = sf_labels
        self.sf_params = sf_params
        self.log_anisotropies = log_anisotropies
        self.param_manager = ParameterManager()
        self.PRINTS = PRINTS

    def create_cardinal_oblique_params(self, sim_params):
        ori_params = []
        for ori_category in ['cardinal', 'oblique']:
            prefix = f"{ori_category}_"
            ps = self.default_params.copy()

            for sim_param, sim_val in sim_params.items():
                if sim_param.startswith(prefix):
                    p_name = sim_param[len(prefix):]  # Remove ori label
                    ps[p_name] = sim_val
                elif not sim_param.startswith(('cardinal_', 'oblique_')):
                    ps[sim_param] = sim_val

            ori_params.append(Params(ps))

        return ori_params  # [Cardinal, Oblique]

    def create_model(self, sim_params, **kwargs):
        """Create population coding model with given parameters."""
        # ori_space = Tiling(min_tile=-90, max_tile=90, step_size=kwargs.get('step_size', 0.1))
        ori_space = Tiling(min_tile=kwargs['min_tile'],
                           max_tile=kwargs['max_tile'],
                           step_size=kwargs['step_size'])

        if sim_params.get('total_n') and sim_params['total_n'] % 1 > 0:
            og_val = sim_params['total_n']
            sim_params['total_n'] = int(sim_params['total_n'] + 0.5)
            print(f"TOTAL N from {og_val:.2f} to {sim_params['total_n']}")

        # Global parameters
        g_params = {'noise_type': 'poisson', 'tuning_type': sim_params.get('tuning_type', 'von_mises'), 'pooling_q': 7,
                    'save_prefs': False}
        for param in ['category_noise_sd', 'category_cardinal_ratio',
                      'correlated_noise', 'corr_max', 'corr_width']:
            g_params[param] = sim_params.get(param, None)

        # Model
        cardinal_params, oblique_params = self.create_cardinal_oblique_params(sim_params)
        model = gen_population_code(cardinal_params, oblique_params, ori_space, pref_sampling="whole",
                                    global_params=g_params)

        return model

    def run_task(self, model, taskname, std_oris, decoder, run_type, n_level_reps, contrasts=(100), **kwargs):
        """Execute specified task with the model."""
        if taskname == 'con_det':
            return run_contrast_detection(model, std_oris, decoder, run_type, n_level_reps, **kwargs)
        elif taskname == 'ori_dis':
            return run_orientation_discrimination(model, std_oris, decoder, contrasts, run_type, n_level_reps, **kwargs)
        else:
            raise ValueError(f"Unknown task: {taskname}")

    def simulate(self, *args):
        """
        Run full simulation for specified task and measure.
        """
        task, decoder, measure, sim_variable_params, config, n_eval = args

        # tasks = [tasks] if not isinstance(tasks, (list, tuple)) else tasks
        # decoders = [decoders] if not isinstance(decoders, (list, tuple)) else decoders

        # === Prepare parameters === #
        sim_params = {**self.default_params}  # Default params

        sim_variable_params = sim_variable_params.copy()  # <-- make local copy
        if self.log_anisotropies is True:
            # Convert back to ratio for consistency with model code
            for param, val in sim_variable_params.items():
                # variable_params[param] = np.exp(val) if param.endswith('_fraction') else val
                sim_variable_params[param] = np.exp(val) if param.endswith('_fraction') else val

        # Overwrite with variable (free) parameters
        sim_params.update(sim_variable_params)

        formatted_params = {k: str((round(v, 2)) if isinstance(v, float) else v)
                            for k, v in sim_variable_params.items()}
        if self.PRINTS is True:
            print(f"\nSimulation params: {formatted_params}")

        # Run simulation for each condition
        data_dict = {}
        all_data = []
        sf_labels = [str(_i) for _i in self.sf_labels] if self.sf_labels else ['cond']
        for idx, cond_label in enumerate(sf_labels):
            # print(f">>> SF {cond_label}")
            cond_params = sim_params.copy()  # init

            if self.sf_params:
                # Cannot run with clashes between free parameters and sf_params
                if any(item in sim_variable_params for item in self.sf_params):
                    raise ValueError("Parameter shared between sf_params and variable_params")

                for param in self.sf_params:
                    cond_params[param] = self.sf_params[param][idx]  # Overwrite with specific condition

            # Base model
            base_params = {k: v for k, v in cond_params.items()}

            # Apply anisotropies
            base_params = self.param_manager.apply_anisotropy_fractions(base_params)

            # Create and run model
            base_model = self.create_model(base_params, **config)
            self.model = base_model

            # Average over multiple evaluations to reduce impact of noise
            run_config = copy.deepcopy(config)
            run_config.update({'model': base_model, 'taskname': task, 'decoder': decoder})

            # ========== Run ========== #
            result_df = pd.concat([self.run_task(**run_config) for _ in range(n_eval)])

            result_df['cond'] = cond_label
            for i_param in sim_params:
                result_df[i_param] = sim_params[i_param]

            cardinalT = result_df[result_df['ori_std'] == 0]['threshold'].mean()
            obliqueT = result_df[result_df['ori_std'] == 45]['threshold'].mean()
            oei = obliqueT / cardinalT

            result_df['cardinalT'] = cardinalT
            result_df['obliqueT'] = obliqueT
            result_df['OEI'] = oei

            # Filter to single line DF with summary results (OEI, Cthresh, Othresh)
            summary_result_df = result_df.iloc[[0]]  # Get copy of single line DF for shared columns
            cols_filtered = [_col for _col in summary_result_df.columns if not _col in ['ori_std', 'threshold', 't_std', 'slope', 's_std', 'design']]
            summary_result_df = summary_result_df[cols_filtered]

            # data_dict[f"{idx + 1}_{cond_label}"] = result_df
            # data_dict[f"{idx + 1}_{cond_label}"] = summary_result_df

            all_data.append(summary_result_df)

        output = pd.concat(all_data)

        # Extract output measures
        # output = extract_output_measures(data_dict, measure)

        return output

