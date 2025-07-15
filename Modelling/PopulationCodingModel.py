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


def set_params(cond_dicty, cardinal_idx, oblique_idx):
    # sets dictys of params for cardinal and oblique populations
    c_dicty = {}
    o_dicty = {}
    for i_param in cond_dicty:
        if not isinstance(cond_dicty[i_param], list):
            continue
        c_dicty[i_param] = cond_dicty[i_param][cardinal_idx]
        o_dicty[i_param] = cond_dicty[i_param][oblique_idx]

    return Params(c_dicty), Params(o_dicty)


"""Population Coding Model that performs an odrientation discrimination two-alternative forced-choice task"""

if __name__ == '__main__':

    '''Population Coding Model that can perform a variety of behavioural tasks'''

    # PATHS #
    og_path = os.getcwd()
    thesis_figures_path = os.path.join(og_path, 'Thesis', 'figures', 'tunings')
    for i_path in [thesis_figures_path]:
        if not os.path.exists(i_path):
            os.makedirs(i_path)

    # SETTINGS
    _debug = False
    _thesis_plots = False
    show_errors = True  # Whether to print errors inside functions
    sampling = 'whole'  # 'whole'=all prefs drawn from distribution, 'indiv'=prefs for each ori popn
    run_type = 'multi_proc'  # 'single' or 'loop' or 'multi_proc'

    # Run information
    run_tasks = ['con_det', 'ori_dis']  # ['con_det', 'ori_dis']
    run_ori_stds = [0, 45]
    run_contrasts = [100]
    n_level_reps = 1000
    tuning_dist = 'von_mises'
    decoders_ori = ['ML', 'PV', 'WTA']  # ['WTA', 'PV', 'ML']
    decoders_condet = ['peak', 'sum', 'mag']  # # ['peak', 'sum', 'v_mag', 'distance', 'PV']
    _identical_template = True  # whether template population is identical to the encoding population

    # Checks
    print(run_tasks)
    design = ''  # ['mocs', 'median']  # median only works with OriDis
    if len(run_tasks) < 3:
        print(f"Only {len(run_tasks)} task chosen")

    attribs_dir = os.path.join(os.getcwd(), 'components', 'attribs')
    make_paths([attribs_dir])

    # # # SIMULATION PARAMETERS # # #
    title = 'DisDet_N180_30rmax'
    run_label = ''
    _rate_or_n = 'n'  # choose whether to define population numbers with rate or number (number is clearly superior no?)
    params = {'cardinal': {'deg_per_pref': [None],  # IGNORE # deg per pref
                           # 'n_prefs': [89],  # [45, 54, 63, 72, 80]  # old_model: per 45 deg
                           # 'n_prefs': [135, 181],
                           # 'n_prefs': [270, 360],
                           # 'n_prefs': [45, 60],
                           'n_prefs': [12],
                           'hwhm': [
                               np.round(_i * _j, 2) for _i in [28.1, 15.0, 22.2]
                               # np.round(_i * _j, 2) for _i in [22.2]
                               for _j in [1, 0.5]
                           ],  # full-width half-max of gaussian tuning mod_orispace
                           'pref_distrib': [f"cos"],  # ['flat', 'cos', f'gauss_{sigma}']
                           'param_distrib': [f"cos"],  # 0['flat', 'cos', f'gauss_{sigma}']
                           'rmax': [60],  # max firing rate (Shen et al., 2014)
                           'spont': [3],  # spontaneous firing rate (used to be 0.03 * rmax, now a constant)
                           'noise_scale': [1],  # scalar of poisson-like noise
                           'crf_exponent': [2],  # 2
                           'crf_c50': [24],  # 24
                           'pooling_extent': [10],  # used if distance-weighted pooling implemented (units=sigma)
                           },
              'oblique': {'deg_per_pref': [None],  # IGNORE # deg per pref
                          'n_prefs': [12],  # [45, 36, 27, 18, 10]
                          # 'n_prefs': [135, 89],
                          # 'n_prefs': [270, 180],
                          'hwhm': [
                              _i for _i in [28.1, 15.0, 11.0]
                              for _ in range(2)
                          ],
                          'pref_distrib': ["cos"],
                          'param_distrib': ["cos"],  # distrib of deg / pref across space
                          'rmax': [60],
                          'spont': [3],  # spontaneous firing rate (used to be 0.03 * rmax)
                          'noise_scale': [1],  # scalar
                          'crf_exponent': [2],  # 2
                          'crf_c50': [24],  # 24
                          'pooling_extent': [10],  # used if distance-weighted pooling implemented (units = sigma)
                          },
              'pooling': {'method': ['sum'],  # todo check functionality (may not be used)
                          'exponent': [1],
                          'q': [6]}
              }

    # # Quickly undo all dict and make all params equal
    _single_param_debug = False
    if _single_param_debug is True:
        for i_ori in params:
            for i_param in params[i_ori]:
                params[i_ori][i_param] = [params[i_ori][i_param][0]]

    # Check for odd number of prefs in each neural population (skips if 0)
    if any(_n % 2 != 1 for _n in [*params['cardinal']['n_prefs'], *params['oblique']['n_prefs']] if _n % 2 != 0):
        print(f'Only use odd numbers of neurons, to ensure central sampling of stimulus orientation')
        quit()

    # Stuff for bounds
    for ori_categ in params:
        if _rate_or_n == 'n':
            # set so bounds can be adjusted to prevent pref overlap
            params[ori_categ]['deg_per_pref'] = [None]
        elif _rate_or_n == 'rate':
            params[ori_categ]['n_prefs'] = [None]


    # Stop extra conditions being looped through when distance-weighted pooling not used
    if not any(i == 'distance' for i in params['pooling']['method']):
        params['cardinal']['pooling_extent'] = [None]
        params['oblique']['pooling_extent'] = [None]
    # fixed_ivs = [key for key in params['cardinal']]   # all ivs
    # interaction_ivs = ['param_distrib']  # does flat vs cos matter?
    fixed_ivs = None
    interaction_ivs = None
    # join card/oblq vals together to facilitate looping through cond combos
    conds = join_dicts(params['cardinal'], params['oblique'])

    # Pooling information (not currently utilised - meant for consistency between detection and discrimination encoding)
    conds['pooling_method'] = params['pooling']['method']
    conds['pooling_exponent'] = params['pooling']['exponent']
    conds['pooling_q'] = params['pooling']['q']
    # Get list of all cond combinations
    cond_combos = get_condcombos(conds, fixed_ivs, interaction_ivs)  # isolated & interaction effects of interest

    counter = 0
    # init list of all datasets (idx corresponds to cond in condlist)
    datasets_all_conds = []
    # get all possible combos of idcs for each paramater
    for idx_conds, i_conds in enumerate(cond_combos):
        i_conds['cond_num'] = idx_conds
        i_conds['tuning_dist'] = tuning_dist
        print(f"{title}\nN conds = {len(cond_combos)}")
        i_filename = f"{idx_conds}_data"  # Set filename
        # Set different parameters for cardinals and obliques
        cardinal, oblique = set_params(i_conds, cardinal_idx=0, oblique_idx=1)
        # Set shared params that are always equal in both cardinal and oblique populations
        shared_params = {'pref_range': 45, 'noise_type': 'poisson', 'correlated_noise': None,
                         'pooling': {'method': i_conds['pooling_method'],
                                     'exponent': i_conds['pooling_exponent'],
                                     'q': i_conds['pooling_q']},
                         }
        for p in shared_params:
            cardinal.__setattr__(p, shared_params[p])
            oblique.__setattr__(p, shared_params[p])
        total_n = (cardinal.n_prefs + oblique.n_prefs) * 2  # record total N
        tiling_step_size = 0.05
        i_conds['tiling_step'] = tiling_step_size
        # Define orientation space
        orispace = Tiling(-90, 90, tiling_step_size)
        # OriSpace = Tiling(min_tile=-180, max_tile=180, step_size=tiling_step_size)  # set to 0.05 for multi_proc
        # Init population code
        PopCode = gen_population_code(cardinal, oblique, orispace, pref_sampling=sampling,
                                      g_params={'tuning_dist': tuning_dist,
                                                'pooling_q': i_conds['pooling_q']
                                                })

        # Update params
        run_info = i_conds
        datasets = {}
        # # # METHOD OF CONSTANT STIMULI # # #
        print(f"\nMocs")
        datasets['mocs'] = {}
        # Contrast Detection
        if any(i == 'con_det' for i in run_tasks):
            levels_contrasts = np.hstack([
                np.linspace(0.01, 0.1, 3),
                np.linspace(0.1, 1, 9),
                np.linspace(1, 10, 9),
                np.linspace(10, 100, 3)]
            )
            levels_contrast = np.asarray(list(set(levels_contrasts)))  # remove repeats
            tic = time.time()
            condet_data = run(experiment=mocs_task, model=PopCode, decoders=decoders_condet,
                              stdoris=run_ori_stds, contrasts=[None], design='mocs',
                              run_type_=run_type, n_level_reps=n_level_reps,
                              _plot_psychometric=False,
                              levels=levels_contrasts, task='con_det')
            datasets['mocs']['con_det'] = condet_data
            print(f'MOCS: ConDet took {time.time() - tic:.2f} sec')

        # Orientation Discrimination
        if any(_t == 'ori_dis' for _t in run_tasks):
            print('oridis')
            levels_ori_offsets = np.hstack([
                np.linspace(0.01, 0.1, 3),
                np.linspace(0.1, 1, 7),
                np.linspace(1, 10, 7),
                np.linspace(10, 100, 3),
            ])
            levels_ori_offsets = np.asarray(list(set(levels_ori_offsets)))  # remove repeats
            tic = time.time()
            datasets['mocs']['ori_dis'] = run(experiment=mocs_task, model=PopCode, decoders=decoders_ori,
                                              stdoris=run_ori_stds, contrasts=run_contrasts,
                                              design='mocs', _plot_psychometric=_thesis_plots,
                                              run_type_=run_type, n_level_reps=n_level_reps,
                                              levels=levels_ori_offsets, task='ori_dis')
            print(f'MOCS: OriDis took {time.time() - tic:.2f} sec')

        # DATASETS #
        # add current cond_combo info and design
        for i_design in datasets:
            if datasets[i_design] == {}:
                continue
            # combine results from which used same design
            datasets[i_design]['combined'] = pd.concat([datasets[i_design][i] for i in datasets[i_design]])
            datasets[i_design]['combined']['cond_num'] = idx_conds  # add cond_number
            for i_data in datasets[i_design]:
                datasets[i_design][i_data]['cond_num'] = idx_conds
                # Run info
                for p in run_info:
                    datasets[i_design][i_data][p] = str(run_info[p])
                # Design
                datasets[i_design][i_data]['design'] = i_design
        # add to database of all results from all conditions
        datasets_all_conds.append(datasets)
        # update % completed
        print(f"{((idx_conds + 1) / len(cond_combos)) * 100:.2f}% completed\n"
              f"{idx_conds + 1} / {len(cond_combos)}")

    # set save path
    tstamp = datetime.now().strftime("%y_%m_%d_%H%M")
    title = f"{title}_{tstamp}".lstrip('_')
    data_path = os.path.join(og_path, 'data', title)
    cond_path = os.path.join(data_path, 'conds_data')
    make_paths([data_path, cond_path])

    # Save cond_info
    to_csv_pkl(pd.DataFrame(cond_combos), data_path, 'cond_list', rnd=4, _csv=True, _pkl=False)

    # Save each condcombo in separate folder (seems unnecessary)
    _save_separate_conds = False
    if _save_separate_conds is True:
        for idx, _ in enumerate(datasets_all_conds):
            i_dir = os.path.join(cond_path, f"{idx}", 'data')
            if not os.path.exists(i_dir):
                os.makedirs(i_dir)
            for i_design in datasets_all_conds[idx]:
                for i_task in datasets_all_conds[idx][i_design]:
                    dframe = datasets_all_conds[idx][i_design][i_task]
                    dframe['label'] = run_label
                    to_csv_pkl(dframe, i_dir, f"{idx}_{i_design}_{i_task}", rnd=4, _csv=True, _pkl=False)

    # Pull out mocs data
    for i_design in ['mocs']:
        i_thresh_data = []
        for idx_cond, _ in enumerate(datasets_all_conds):
            i_cond_data = datasets_all_conds[idx_cond]
            if i_cond_data.get(i_design):
                i_thresh_data.append(i_cond_data[i_design]['combined'])
        if len(i_thresh_data) == 0:
            continue
        i_thresh_data = pd.concat(i_thresh_data)  # Combine all conditions
        to_csv_pkl(i_thresh_data, data_path, f"thresholds_{i_design}", rnd=4, _csv=True, _pkl=False)
        # OEI data for each design
        i_oei_data = get_oei_data(i_thresh_data, cond_list=['design', 'task', 'decoder', 'contrast', 'cond_num'],
                                  ori_label='ori_std', dv='threshold')
        to_csv_pkl(i_oei_data, data_path, f"oei_{i_design}", rnd=4, _csv=True, _pkl=False)

    # Combine datasets from each design
    combined_data = []
    for idx, _ in enumerate(datasets_all_conds):
        for i_design in datasets_all_conds[idx]:
            if datasets_all_conds[idx][i_design] == {}:
                continue
            combined_data.append(datasets_all_conds[idx][i_design]['combined'])
    combined_data = pd.concat(combined_data)
    combined_data = combined_data.sort_values(by=['task', 'decoder', 'ori_std', 'contrast'])
    # Save condcombo data
    combined_data['label'] = run_label
    to_csv_pkl(combined_data, data_path, f"combined_datasets", rnd=4, _csv=True, _pkl=True)

    # Calculate oblique effect ratio index (OEI)
    oei_data = get_oei_data(combined_data, cond_list=['design', 'task', 'decoder', 'contrast', 'cond_num'],
                            ori_label='ori_std', dv='threshold')
    oei_data['label'] = run_label
    to_csv_pkl(oei_data, data_path, 'combined_ratios', rnd=4, _pkl=True)

    z = 0
