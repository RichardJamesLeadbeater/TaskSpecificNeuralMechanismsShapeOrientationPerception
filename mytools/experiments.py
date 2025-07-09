import itertools as itls
import numpy as np
from numpy.random import choice
from numpy import asarray as arr
import os
import pandas as pd
from psychopy.misc import fromFile
import scipy.stats as stats
from mytools.dframe_tools import init_cols, index_dframe
from mytools.list_tools import get_element_combos, shuffle_order, flatten


class BayesianStaircase:
    def __init__(self, startVal, minVal, maxVal, nUp, nDown, nReversals):
        """Initialise the Bayesian Staircase with necessary parameters."""
        self.startVal = startVal  # Initial threshold estimate
        self.minVal = minVal  # Minimum contrast level
        self.maxVal = maxVal  # Maximum contrast level
        self.nUp = nUp  # Number of correct responses before going up
        self.nDown = nDown  # Number of incorrect responses before going down
        self.nReversals = nReversals  # Number of reversals to stop after
        # Initialise the prior (uniform distribution over the contrast range)
        self.prior = stats.uniform(self.minVal, self.maxVal - self.minVal)
        # Initialisation of staircase parameters
        self.current_value = self.startVal  # Current contrast level
        self.reversalIntensities = []  # Track reversals in intensity
        # Trial counters
        self.up_count = 0
        self.down_count = 0
        self.reversals = 0

    def __iter__(self):
        """ Makes the class iterable """
        return self

    def __next__(self):
        """ Returns the next stimulus intensity. """
        if self.reversals >= self.nReversals:
            raise StopIteration  # Stop iteration after the required number of reversals
        return self.current_value

    def update_posterior(self, response):
        """Update the posterior distribution based on the response."""
        # Likelihood function (using a simple Gaussian model for the psychometric function)
        likelihood = stats.norm.pdf(response, loc=self.current_value, scale=0.1)
        # Adjust likelihood based on whether the response was correct or incorrect
        if response == 1:  # Correct response
            likelihood = 1 - likelihood  # Flip likelihood for correct response
        # No adjustment needed for incorrect responses (they align with the likelihood)
        # Update the posterior by multiplying prior with likelihood (Bayes' rule)
        posterior = self.prior.pdf(self.current_value) * likelihood
        # Normalize the posterior to make it a valid probability distribution
        posterior /= np.sum(posterior)
        # Update the current belief about the contrast level based on the posterior
        self.current_value = np.argmax(posterior)
        return posterior

    def get_next_stimulus(self):
        """Return the next stimulus intensity based on the updated posterior."""
        return self.current_value

    def add_response(self, response):
        """
        Run the staircase by updating the posterior and adjusting the stimulus.
        If incorrect, response=0; if correct, response=1.
        """
        # Update the posterior based on the response
        self.update_posterior(response)
        # Update counters for up and down based on the response
        if response == 1:
            self.up_count += 1
        else:
            self.down_count += 1
        # Check if the reversal condition has been met
        if self.up_count >= self.nUp:
            self.reversals += 1
            self.reversalIntensities.append(self.current_value)
            self.up_count = 0
            self.down_count = 0  # Reset counters
        if self.down_count >= self.nDown:
            self.reversals += 1
            self.reversalIntensities.append(self.current_value)
            self.up_count = 0
            self.down_count = 0  # Reset counters
        # Exit when enough reversals have been made
        if self.reversals >= self.nReversals:
            return True  # Staircase complete
        return False  # Continue the staircase


class PseudorandomOrder:
    # randomly chooses item from list whilst ensuring each item shows equal n times
    def __init__(self, items, block_size=5):
        self.items = items  # e.g. [['f','j'], ['j','f']
        self.block_size = block_size  # n repeats before resetting
        self.current_order = []
        self.idx_keys = [f"{i}" for i in range(len(items))]
        self.item_counter = {}
        for i in self.idx_keys:
            self.item_counter[i] = 0  # init counter for each order

    def get_current_order(self, update=False):
        if update:
            # randomly choose idx_key as long as counter is < block_size
            idx_choice = choice([i for i in self.idx_keys if self.item_counter[i] < self.block_size])
            self.current_order = self.items[int(idx_choice)]  # set current_order by indexing into self.order_list
            self.item_counter[idx_choice] += 1  # update counter
            # if all reached block_size reset counters
            if all(self.item_counter[i] == self.block_size for i in self.idx_keys):
                for i in self.idx_keys:
                    self.item_counter[i] = 0
        else:
            self.current_order = choice(self.items)  # if not updating random item in list
        return self.current_order


def gen_tally(data_folder, conds, run_info=None, _update=False):
    # conds should be a dict with multiple lists
    # create list of all possible cond combos (keys for dict)
    tally_labels = []
    cond_keys = list(conds.keys())
    cond_lists = list(conds.values())
    cond_combos = get_element_combos(cond_lists)
    for i_combo in cond_combos:
        i_combo = [f"{_c}" for _c in i_combo]
        tally_labels.append(f"{i_combo}")
    runcount_tally = init_cols(cols=tally_labels, val=0)  # init tally at 0
    # Count n files with each condcombo then append present condition
    filetype = "<class 'psychopy.data.experiment.ExperimentHandler'>"  # Desired file type
    for i_file in os.listdir(data_folder):
        if i_file.endswith('.psydat'):  # only count one filetype
            datafile = fromFile(os.path.join(data_folder, i_file))
            if str(type(datafile)) != filetype:  # If not desired filetype then continue
                continue
            info = datafile.extraInfo
            if info.get('results') is None:  # if file has no results skip from tally
                continue
            tally_label = f"{[info.get(_k) for _k in cond_keys]}"
            if runcount_tally.get(tally_label) is not None:
                runcount_tally[tally_label] += 1  # add to tally
    if _update is True:  # update tally with current run info
        # UPDATE TALLY WITH CURRENT COND INFO #
        run_label = f"{[run_info.get(_k) for _k in cond_keys]}"
        runcount_tally[run_label] += 1  # update tally with current run information
    # save tally as dframe
    runcount_tally = pd.DataFrame(runcount_tally, index=['count']).transpose()
    runcount_tally.to_csv(os.path.join(data_folder, f"_runcount_tally.csv"), index=True)
    return runcount_tally


def auto_conds(tally, max_nruns=4, extra_run_conds=None, extra_n_runs=0):
    # Automatic randomised selection of conditions based on current tally
    # Assumes tally values are strings "['val1', 'val2'...]"
    if tally['count'].min() < max_nruns:  # if desired trial limit reached, inform user
        # Randomise from conditionss with fewer runs
        counts = sorted(list(tally['count'].unique()))
        if len(counts) > 1:
            next_count = counts[1]
        else:
            next_count = np.max(counts)
        conds_todo = tally[tally['count'] < next_count].index
        if conds_todo.empty:  # If counts are equal, pick from all conds
            conds_todo = tally.index
    else:
        if extra_run_conds is None:
            return None
        # Index DataFrame to pick only from extraconds
        index_labels = []
        for _o in extra_run_conds:
            for _n in extra_run_conds[_o]:
                index_labels.append(f"['{_o}', '{_n}']")
        extra_tally = tally.loc[index_labels]
        if extra_tally['count'].min() < max_nruns + extra_n_runs:
            conds_todo = []
            for _o in extra_run_conds:
                for _n in extra_run_conds[_o]:
                    xtra_label = f"['{_o}', '{_n}']"  # convert to match tally format
                    if tally.loc[xtra_label]['count'] <  max_nruns + extra_n_runs:
                        conds_todo.append(xtra_label)  # which extra runs are suitable
        else:
            return None
    # Automate trial cond choice
    conds_choice = choice(conds_todo).split("'")  # assumes all values are strings "['val1', 'val2'...]"
    conds_choice = [conds_choice[_idx] for _idx in range(len(conds_choice)) if _idx % 2 != 0]  # every odd idx (split)
    return conds_choice


def get_completed_cond_list(data_folder, cond_keys):
    # Retrieve a list of all completed conditions
    completed_list = []
    filetype = "<class 'psychopy.data.experiment.ExperimentHandler'>"  # Desired file type
    for i_file in os.listdir(data_folder):
        if i_file.endswith('.psydat'):  # Only use this filetype
            datafile = fromFile(os.path.join(data_folder, i_file))  # Load data from filetype
            if str(type(datafile)) != filetype:  # If not desired filetype then continue
                continue
            info = datafile.extraInfo  # Check file information
            if info.get('results') is None:  # If file has no results then skip from count
                continue
            run_conds = [info.get(_k) for _k in cond_keys]  # Pull relevant information from file
            completed_list.append(run_conds)  # Add completed run to list

    return completed_list


def gen_cond_combos(*conds, block_conds=None, n_runs=1):
    # Generate a list of all condition combinations - Can be split into blocks for each element in block_conds
    # All variables except for n_runs should be lists
    cond_combos = []
    if block_conds is None:
        for r in range(n_runs):
            # Generate list of all possible condition combinations
            cond_combos = [*cond_combos, *list(itls.product(*conds))]
            cond_combos = shuffle_order(cond_combos)
    else:
        for r in range(n_runs):
            # Split into blocks - Generate all combinations for each block
            blocks = [list(itls.product(*conds, [_i])) for _i in block_conds]
            # Randomise order of blocks and individual conditions within blocks
            blocks = shuffle_order([shuffle_order(_i) for _i in blocks])
            cond_combos = [*cond_combos, *blocks]  # Join lists
        cond_combos = flatten(cond_combos)  # Flatten list into single dimension
    # Convert individual condition groups into lists
    cond_combos = [list(_i) for _i in cond_combos]
    return cond_combos


def rms_contrast(array, bg_lum=None):
    if bg_lum is not None:
        if bg_lum == '[0, 0]':
            bg_lum = array[0, 0]
        array = array[array != bg_lum]  # filter out background vals (image only)
    # rms as stdev assumes lum vals from -1 to 1
    rms = np.std(arr(array), ddof=0)  # sanity check: std([-1, 1]) == 1 and std([0, .5, 1, .5] * 500) == 0.707
    # Note: dividing by the mean lum (as in Pelli, 2013) gives a different contrast dependent on the value of mean
    #       rms = stdev(lum) / mean(lum) : Pelli (2013)
    return rms


def michelson_contrast(array):
    array = (arr(array) + 1) / 2  # convert to rgb01
    con = (np.max(array) - np.min(array)) / (np.max(array) + np.min(array))
    return con

