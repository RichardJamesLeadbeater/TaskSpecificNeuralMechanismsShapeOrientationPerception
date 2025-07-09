import pandas as pd
import os
import numpy as np
import itertools as itls
from .dict_tools import init_cols
from mytools.list_tools import force_to_list


def change_dframe_labels(dframe, col, old_labels=None, new_labels=None):
    # Make a copy of the original DataFrame to avoid modifying in place
    output_dframe = dframe.copy()
    # Create a mapping dictionary from old labels to new labels
    label_map = dict(zip(old_labels, new_labels))
    # Use pandas' vectorised replace method to relabel the specified column
    output_dframe[col] = output_dframe[col].replace(label_map)

    return output_dframe


def change_dframe_colnames(dframe, old_colnames, new_colnames):
    """ Change multiple column names in DataFrame """
    og_colnames = dframe.columns.to_list()
    updated_colnames = [col for col in og_colnames]  # Create copy of og_cols in og order
    output_dframe = dframe.copy()
    for idx in range(len(old_colnames)):
        old = old_colnames[idx]
        new = new_colnames[idx]
        # Update column names
        for col_idx, col in enumerate(og_colnames):
            if col == old:
                updated_colnames[col_idx] = new  # Replace label in list of colnames in order
                output_dframe[new] = dframe[old]  # Add copy of og_column under new colname
    output_dframe = output_dframe[updated_colnames]  # Use only these colnames
    return output_dframe


def index_dframe(dframe, columns, values, index_filter='and'):
    """
    Filters a DataFrame based on multiple conditions applied to specified columns.

    Args:
        dframe (pd.DataFrame): The DataFrame to filter.
        columns (list or str): A single column or a list of columns to apply the filter.
        values (list or any): A single value or a list of values corresponding to each column for filtering.
        index_filter (str): Specifies how to combine conditions. Options: 'and', 'or', 'not'.
                             'and' applies all conditions together, 'or' applies any condition,
                             'not' negates the condition.

    Returns:
        pd.DataFrame: The filtered DataFrame based on the conditions.
    """

    # Convert to list in case single value entered
    columns = force_to_list(columns)
    values = force_to_list(values)
    # indexes dframe where col == val
    if len(columns) == 1 and len(values) > 1:
        columns = [columns[0]] * len(values)  # Repeat if single column with multiple values

    # If there's only one column and multiple values, replicate the column to match the values length
    if len(columns) == 1 and len(values) > 1:
        columns = itls.repeat(columns[0], len(values))  # Repeat the column name to match the length of values

    # Filter columns to only include those that exist in the DataFrame
    columns = [col for col in columns if col in dframe.columns]

    if not columns:  # Return original DataFrame if no valid columns are found
        return dframe

    # Filter the DataFrame based on the conditions
    if index_filter == 'or':
        # Apply 'or' logic: each condition can be True independently
        output_dframe = []
        for col, val in zip(columns, values):
            output_dframe.append(dframe[dframe[col] == val])  # Add each condition's result
        output_dframe = pd.concat(output_dframe).drop_duplicates()  # Concatenate and remove duplicates

    elif index_filter == 'and':
        # Apply 'and' logic: all conditions must be True together
        output_dframe = dframe.copy()  # Work on a copy to preserve original DataFrame
        for col, val in zip(columns, values):
            output_dframe = output_dframe[output_dframe[col] == val]  # Apply each condition in sequence

    elif index_filter == 'not':
        # Apply 'not' logic: conditions must NOT be True
        output_dframe = dframe.copy()  # Work on a copy to preserve original DataFrame
        for col, val in zip(columns, values):
            output_dframe = output_dframe[output_dframe[col] != val]  # Apply negation condition

    else:
        raise ValueError("Invalid 'index_filter' value. Must be 'and', 'or', or 'not'.")

    return output_dframe


def to_csv_pkl(dframe, folder, title, rnd=2, _csv=True, _pkl=True):
    if _csv is True:
        dframe.round(rnd).to_csv(os.path.join(folder, f"{title}.csv"), index=False)
    if _pkl is True:
        dframe.to_pickle(os.path.join(folder, f"{title}.pkl"))


def cols2rows(dframe, change_cols, new_col):
    # useful to convert summary_data data tables for plotting
    keep_cols = [i for i in dframe.columns if all(j != i for j in change_cols)]
    output = init_cols(keep_cols + [new_col] + ['score'])
    for i_change in change_cols:
        output[new_col] += [i_change] * len(dframe)
        output['score'] += dframe[i_change].to_list()
        for i_keep in keep_cols:
            output[i_keep] += dframe[i_keep].to_list()
    return pd.DataFrame(output)


def remove_rows(dframe, col, vals):
    rows2remove = index_dframe(dframe, columns=[col], values=vals, index_filter='or').index
    dframe = dframe.drop(index=rows2remove)
    return dframe


def custom_sort(dframe, cols, ordered_labels):
    # input: cols as list, ordered_labels as list of lists
    # if single list is input for a single column
    if len(cols) == 1:
        if not any(isinstance(ordered_labels[0], _type) for _type in [list, tuple]):
            ordered_labels = [ordered_labels]
    # sort selected cols of dframe based on custom order
    _dframe = dframe.copy()
    order_cols = []
    numbered_orders = {}
    for col, order in zip(cols, ordered_labels):
        if all(_i != col for _i in _dframe.columns):
            continue  # skip if column does not exist
        else:
            order_cols.append(col)
        for jdx, cond in enumerate(order):
            numbered_orders[cond] = jdx
    _dframe = _dframe.sort_values(by=order_cols, key=lambda x: x.map(numbered_orders))
    return _dframe


def get_cond_combos(dframe, conds):
    # return all possible condition combinations for given cols
    combos = itls.product(*[dframe[i].unique() for i in conds])
    combos = list(combos)
    return combos


def spss_format(df, cond_list, id_label, dv_label):
    # Save dataframe in format for use in SPSS or Jamovi (repeated measures ANOVA)
    # Requires that input dataframe contains equal observations in all
    all_conds = [df[_cond].unique() for _cond in cond_list]
    cond_combos = list(itls.product(*all_conds))
    reformatted_df = pd.DataFrame({id_label: df[id_label].unique()})
    for i_combo in cond_combos:
        i_label = ''.join([f"{_i}_" for _i in i_combo]).rstrip("_")  # generate new label for combined condition
        i_data = index_dframe(df, cond_list, i_combo)
        reformatted_df[i_label] = i_data[dv_label].to_list()
    return reformatted_df

