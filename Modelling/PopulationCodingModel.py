#
# Population Coding Model
#
import os
import time
from copy import deepcopy
from datetime import datetime
import numpy as np
import pandas as pd
# import own modules
from mytools.analysis import get_mean_data, get_oei_data
from mytools.dframe_tools import to_csv_pkl, change_dframe_colnames
from mytools.os_tools import make_paths
# import model components
from components.general_functions import join_dicts, get_condcombos
from components.run import mocs_task, run
from components.population_code import Params, Tiling, gen_population_code


"""
Population Coding Model that performs an odrientation discrimination two-alternative forced-choice task
"""

if __name__ == '__main__':

    '''Population Coding Model that can perform a variety of behavioural tasks'''

    # PATHS #
    og_path = os.getcwd()
    thesis_figures_path = os.path.join(og_path, 'Thesis', 'figures', 'tunings')
    for i_path in [thesis_figures_path]:
        if not os.path.exists(i_path):
            os.makedirs(i_path)

    # SETTINGS
    run_type = 'loop'  # 'single' or 'loop' or 'multi_proc'

    # Run information
    run_tasks = ['con_det', 'ori_dis']  # ['con_det', 'ori_dis']
    run_ori_stds = [0, 45]
    run_contrasts = [100]  # Applies to orientation discrimination
    n_level_reps = 1000

    # Pooling methods used in contrast-detection task
    decoders_condet = ['peak', 'sum', 'mag']  # # ['peak', 'sum', 'v_mag', 'distance', 'PV']

    # Orientation decoding methods used in orientation-discrimination task
    decoders_ori = ['ML', 'PV', 'WTA']  # ['WTA', 'PV', 'ML']

    # # # SIMULATION PARAMETERS # # #
    run_label = 'Simulation_1'  # Name of dataset

    # Orientation space
    orispace = Tiling(-90, 90, step_size=0.05)

    # Fixed response parameters
    fixed_params = {
        'noise_type': 'poisson',  # Poisson-like noise
        'pref_distrib': 'cos',  # Prefs drawn from cosine distribution
        'param_distrib': 'cos',  # Param vals drawn from cosine distribution
        'crf_exponent': 2,  # Parameter of Naka-Rushton CRF
        'crf_c50': 24,  # Parameter of Naka-Rushton CRF
        'spont': 3,  # Spontaneous firing rate (sp/s)
    }

    # Bandwidths corresponding to 2, 8, 16 cpd
    sf_hwhm = [28.1, 15.0, 11.0]

    # Variable response parameters for each simulation: [[Cardinal, Oblique], ...]
    variable_CvsO_params = {
        'n_prefs': [
            [135, 135],  # Number of prefs per 45 degrees for Cardinal and Oblique populations
            [180, 90],  # Cardinal overrepresentation
        ],
        'hwhm': [
            *[[i, i] for i in sf_hwhm],  # Uniformly decreasing bandwidths with increasing SF
            *[[i * 0.5, i] for i in sf_hwhm]  # Anistropic bandwidths that decrease with increasing SF
        ],
        'noise_scale': [
            [1, 1],  # Isotropic levels of response variability
            [0.5, 1]  # Anisotropic levels of response variability
        ],
        'rmax': [
            [60, 60],  # Uniform rmax
            [120, 60]  # Anisotropic maximum firing rates
        ]
    }

    # # Simpler parameters used for debugging
    # variable_CvsO_params = {
    #     'n_prefs': [[15, 15]],
    #     'hwhm': [*[[i, i] for i in sf_hwhm]],
    #     'noise_scale': [[1, 1]],
    #     'rmax': [[60, 60]]
    # }

    # Get list of all simulation parameter combinations
    simulation_conditions = get_condcombos(variable_CvsO_params)
    print(f"{run_label}\nN conds = {len(simulation_conditions)}")

    # Loop through simulation conditions
    all_data = []
    for idx_conds, i_conds in enumerate(simulation_conditions):
        # Update condition number
        cond_num = idx_conds + 1
        print(f"\nSimulation {cond_num}/{len(simulation_conditions)}")

        # Set cardinal and oblique parameters
        cardinal = Params({
            **{k: v[0] for k, v in i_conds.items()},
            **fixed_params
        })
        oblique = Params({
            **{k: v[1] for k, v in i_conds.items()},
            **fixed_params
        })

        # Initialise model with these parameters
        PopCode = gen_population_code(cardinal, oblique, orispace,
                                      g_params={'pooling_q': 6}  # Applies if using weighted pooling
                                      )

        # Update conditions file information
        i_conds['cond_num'] = cond_num
        i_conds['tiling_step'] = orispace.step_size
        run_info = i_conds
        i_filename = f"{cond_num}_data"

        # Update params
        datasets = {}

        # Contrast Detection
        if any(i == 'con_det' for i in run_tasks):
            print(f"\t- Starting contrast detection simulation")
            # Step 1 - Define the contrast level of each simulation
            levels_contrasts = np.hstack([
                np.linspace(0.01, 0.1, 3),
                np.linspace(0.1, 1, 9),
                np.linspace(1, 10, 9),
                np.linspace(10, 100, 5)]
            )  # Cover a wide range of possible performance
            levels_contrasts = np.asarray(list(set(levels_contrasts)))  # Remove repeated valuestic = time.time()

            # Step 2 - Run simulation for each standard orientation
            tic = time.time()
            condet_data = run(
                experiment=mocs_task,  task='con_det', levels=levels_contrasts, n_level_reps=n_level_reps,
                model=PopCode, decoders=decoders_condet, stdoris=run_ori_stds, contrasts=[None],
                run_type_=run_type, design='mocs',
            )
            print(f'\t\t- Contrast detection took {time.time() - tic:.2f} sec')

            # Add condition information to dataset
            for k, v in i_conds.items():
                condet_data[k] = f"{v}"

            # Add to full dataset
            datasets['con_det'] = condet_data

        # Orientation Discrimination
        if any(_t == 'ori_dis' for _t in run_tasks):
            print('\t- Starting orientation discrimination simulation')
            # Step 1 - Define the orientation differences between standard and comparison stimuli
            levels_ori_offsets = np.hstack([
                np.linspace(0.01, 0.1, 3),
                np.linspace(0.1, 1, 7),
                np.linspace(1, 10, 7),
                np.linspace(10, 45, 3),
            ])
            levels_ori_offsets = np.asarray(list(set(levels_ori_offsets)))  # remove repeats

            # Step 2 - Run simulation for each standard orientation & contrast
            tic = time.time()
            oridis_data = run(
                experiment=mocs_task, task='ori_dis', levels=levels_ori_offsets, n_level_reps=n_level_reps,
                model=PopCode, decoders=decoders_ori, stdoris=run_ori_stds, contrasts=run_contrasts,
                run_type_=run_type, design='mocs',
            )
            print(f'\t\t- Orientation discrimination took {time.time() - tic:.2f} sec')

            # Add condition information to dataset
            for k, v in i_conds.items():
                oridis_data[k] = f"{v}"

            # Add to full dataset
            datasets['ori_dis'] = oridis_data

        # Store datasets completed for this parameter set
        datasets['combined'] = pd.concat(datasets.values())
        # Include run info
        for k, v in run_info.items():
            datasets['combined'][k] = f"{v}"

        # Store combined dataset
        all_data.append(datasets['combined'])

        # Update progress through simulations
        print(f"{(cond_num / len(simulation_conditions)) * 100:.1f}% completed\n"
              f"{cond_num} / {len(simulation_conditions)}")

    # Once all simulations are complete, set save path
    tstamp = datetime.now().strftime("%y_%m_%d_%H%M")
    data_path = os.path.join(og_path, 'data', f"{run_label}_{tstamp}".lstrip('_'))
    cond_path = os.path.join(data_path, 'conds_data')
    make_paths([data_path, cond_path])

    # Save simulation conditions separately
    to_csv_pkl(pd.DataFrame(simulation_conditions), data_path, 'cond_list', rnd=4, _csv=True, _pkl=False)

    # Combine data from all simulations
    all_data = pd.concat(all_data)
    all_data['label'] = run_label
    all_data = all_data.sort_values(by=['task', 'decoder', 'ori_std', 'contrast'])

    to_csv_pkl(all_data, data_path, f"thresholds", rnd=4, _csv=True, _pkl=False)

    # Calculate OEI scores
    oei_data = get_oei_data(all_data, cond_list=['task', 'decoder', 'contrast', 'cond_num'],
                              ori_label='ori_std', dv='threshold')
    to_csv_pkl(oei_data, data_path, f"OEI_scores", rnd=4, _csv=True, _pkl=False)

