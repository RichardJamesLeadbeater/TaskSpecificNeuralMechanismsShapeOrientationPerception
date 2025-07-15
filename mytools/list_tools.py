import itertools as itls
import random
import copy


def force_to_list(x):
    # Convert any data type into list
    if isinstance(x, str) or not hasattr(x, '__iter__'):  # Avoid splitting string
        return [x]
    else:
        return list(x)

def all_combinations(listy):
    # Input should be a list of lists
    # Output is all combinations of each original list element (order maintained)
    combos = itls.product(*listy)
    combos = list(combos)

    return combos

def shuffle(listy):
    listy = list(copy.copy(listy))  # accounts for other datatypes
    random.shuffle(listy)
    return listy


def flatten(listy):
    flattened_listy = list(itls.chain(*listy))
    return flattened_listy
