import itertools as itls
import random
import copy


def force_to_list(x):
    # Convert any data type into list
    if isinstance(x, str) or not hasattr(x, '__iter__'):  # Avoid splitting string
        return [x]
    else:
        return list(x)


def shuffle_order(listy):
    listy = list(copy.copy(listy))  # accounts for other datatypes
    random.shuffle(listy)
    return listy


def unique_values(listy):
    listy = list(copy.copy(listy))
    unique_vals = []
    for i in listy:
        if any(j == i for j in unique_vals):
            continue
        else:
            unique_vals.append(i)
    return sorted(unique_vals)


def get_element_combos(list_of_lists):
    combos = itls.product(*list_of_lists)
    combos = list(combos)
    return combos


def flatten(listy):
    flattened_listy = list(itls.chain(*listy))
    return flattened_listy


def remove_duplicates(listy):
    listy = [_i for _i in listy]  # copy
    listy = list(dict.fromkeys(listy))
    return listy

