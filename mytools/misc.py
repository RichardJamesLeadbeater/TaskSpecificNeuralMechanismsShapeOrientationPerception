import datetime
import numpy as np
import pandas as pd
from mytools.dframe_tools import index_dframe, change_dframe_labels


def kill_code(message=''):
    # end the code by dividing by zero with printed message
    print(message)
    exit(1)


def assign_participant_numbers(og_names):
    # Input list and gives out relevant numbers
    assigned = [f"P{idx + 1}" for idx, _ in enumerate(og_names)]
    return assigned


def add_cardinal_and_oblique(dframe, ori_label):
    ori_ = ori_label
    cardinals = dframe[(dframe[ori_] == 'horizontal') | (dframe[ori_] == 'vertical')].copy()
    cardinals[ori_] = 'cardinal'
    obliques = dframe[(dframe[ori_] == 'minus45') | (dframe[ori_] == 'plus45')].copy()
    obliques[ori_] = 'oblique'
    output_dframe = pd.concat([dframe, cardinals, obliques])
    return output_dframe


def calc_cardinal_and_oblique(input_df, ori_label, dv_label):
    # get the mean values of cardinals (H & V) and obliques (-45 & 45)
    df = input_df.copy()
    horiz = index_dframe(df, [ori_label], ['horizontal'])
    vert = index_dframe(df, [ori_label], ['vertical'])
    cardinal = horiz.copy()  # template with incorrect dv vals
    cardinal[ori_label] = 'cardinal'
    cardinal[dv_label] = np.mean([horiz[dv_label].to_numpy(), vert[dv_label].to_numpy()],
                                 axis=0)
    m45 = index_dframe(df, [ori_label], ['minus45'])
    p45 = index_dframe(df, [ori_label], ['plus45'])
    oblique = m45.copy()  # template with incorrect dv vals
    oblique[ori_label] = 'oblique'
    oblique[dv_label] = np.mean([m45[dv_label].to_numpy(), p45[dv_label].to_numpy()],
                                axis=0)
    output_dframe = pd.concat([df, cardinal, oblique])
    output_dframe = output_dframe.sort_values(by=[_i for _i in df.columns if _i != dv_label])
    return output_dframe


def get_tstamp():
    t_now = datetime.datetime.now()  # create timestamp
    t_stamp = [t_now.year, t_now.month, t_now.day, t_now.hour, t_now.minute]
    for idx, t in enumerate(t_stamp):
        if len(str(t)) == 1:
            t_stamp[idx] = f"0{t}"  # adds 0 to single digit nums (aids sorting)
    return ''.join([str(t) for t in t_stamp])


def get_orientation(ori_):
    if ori_ == 'vertical':
        ori_ = 0
    elif ori_ == 'plus45':
        ori_ = 45
    elif ori_ == 'horizontal':
        ori_ = 90
    elif ori_ == 'minus45':
        ori_ = -45
    return ori_


def rgb255_to_minus101(img_array):
    """Convert array 0 to 255 into -1 to 1"""
    output_img = (img_array / 127.5) - 1

    return output_img


def is_cardinal_or_oblique(my_ori):
    # return str of cardinal or oblique to lighten later code
    if any(i == my_ori for i in ['vertical', 'horizontal']):
        return 'cardinal'
    elif any(i == my_ori for i in ['minus45', 'plus45']):
        return 'oblique'


def visual_angle_calculator(size=None, distance=None, vis_angle=None):
    if vis_angle is None:
        vis_angle = 2 * np.arctan((size / 2) / distance)
        vis_angle = vis_angle * (180 / np.pi)  # rad to deg
        return vis_angle
    elif size is None:
        vis_angle = vis_angle / (180 / np.pi)  # deg to rad
        size = 2 * np.tan(vis_angle / 2) * distance
        return size
    elif vis_angle is None:
        vis_angle = vis_angle / (180 / np.pi)  # deg to rad
        distance = (size / 2) / np.tan(vis_angle / 2)
        return distance

