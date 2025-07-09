from copy import copy


def init_cols(cols, val=None):
    if val is None:
        val = []
    output = {}
    for i_col in cols:
        output[i_col] = copy(val)  # copy used to keep vals independent
    return output


def combine_dicts(*args):
    # Initialize an empty dictionary to store the combined result
    combined = {}

    # Iterate over each dictionary passed as argument
    for d in args:
        # Update the combined dictionary with each dictionary's items
        combined.update(d)

    # Return the combined dictionary
    return combined



def append_dicty(dicty, values):
    # Alters original dict
    if len(dicty.keys()) != len(values):
        raise ValueError("Length of dict and values must be equal")
    for idx, key in enumerate(dicty):
        dicty[key].append(values[idx])


def copy_dicty(dicty):
    output_dict = {}
    for key in dicty:
        output_dict[key] = copy(dicty[key])
    return output_dict

