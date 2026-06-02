def lowercase_greek_alphabet():
    alphabet = {'alpha': '\u03B1',
                'beta': '\u03B2',
                'gamma': '\u03B3',
                'delta': '\u03B4',
                'epsilon': '\u03B5',
                'zeta': '\u03B6',
                'eta': '\u03B7',
                'theta': '\u03B8',
                'iota': '\u03B9',
                'kappa': '\u03BA',
                'lambda': '\u03BB',
                'mu': '\u03BC',
                'nu': '\u03BD',
                'xi': '\u03BE',
                'omicron': '\u03BF',
                'pi': '\u03C0',
                'rho': '\u03C1',
                'sigma': '\u03C3',
                'tau': '\u03C4',
                'upsilon': '\u03C5',
                'phi': '\u03C6',
                'chi': '\u03C7',
                'psi': '\u03C8',
                'omega': '\u03C9'
                }
    return alphabet


def uppercase_greek_alphabet():
    alphabet = {'Alpha': '\u0391',
                'Beta': '\u0392',
                'Gamma': '\u0393',
                'Delta': '\u0394',
                'Epsilon': '\u0395',
                'Zeta': '\u0396',
                'Eta': '\u0397',
                'Theta': '\u0398',
                'Iota': '\u0399',
                'Kappa': '\u039A',
                'Lambda': '\u039B',
                'Mu': '\u039C',
                'Nu': '\u039D',
                'Xi': '\u039E',
                'Omicron': '\u039F',
                'Pi': '\u03A0',
                'Rho': '\u03A1',
                'Sigma': '\u03A3',
                'Tau': '\u03A4',
                'Upsilon': '\u03A5',
                'Phi': '\u03A6',
                'Chi': '\u03A7',
                'Psi': '\u03A8',
                'Omega': '\u03A9'
                }
    return alphabet


def model_symbols():
    # Symbol list
    sym = lowercase_greek_alphabet()
    SYM = uppercase_greek_alphabet()

    # # Use a key in case vals need changing later
    # n = "N"
    # fwhm = "h"  # sym['sigma']
    # noise_scale = "F"  # sym['lambda']
    # rmax = "g"  # sub('r', 'max')

    param_labels = {
        # Each model parameter and associated symbols
        'N': italics("N"),
        # 'fwhm': sub(sym['sigma'], 'ori'),
        'fwhm': 'h',  # Used in paper as symbol for HWHM
        # 'noise_scale': sub(sym['lambda'], 'F'),
        'noise_scale': "F",
        # 'rmax': sub('r', 'max'),
        'rmax': 'g',
        'spont': sub('r', '0'),
        'crf_c50': sub('c', '50'),
        'crf_exponent': r"$\it{q}$",

        # Add cardinal and oblique versions of each
        'c_N': sub('N', 'C'),
        'o_N': sub('N', 'O'),
        # 'c_fwhm': sub(sym['sigma'], 'C'),
        # 'o_fwhm': sub(sym['sigma'], 'O'),
        'c_hwhm': sub('h', 'C'),
        'o_hwhm': sub('h', 'O'),
        # 'c_noise': sub(sym['lambda'], 'C'),
        # 'o_noise': sub(sym['lambda'], 'O'),
        'c_noise': sub("F", 'C'),
        'o_noise': sub("F", 'O'),
        # 'o_rmax': f"{sub('r', 'max  O')}",
        # 'c_rmax': f"{sub('r', 'max  C')}",
        'o_rmax': f"{sub('g', 'O')}",
        'c_rmax': f"{sub('g', 'C')}",

        # Relative parameters for figure legends
        'total_n': f"{sub('N', 'total')}",
        'propn_cardinal': f"{sub('N', 'total')} / {sub('N', 'C')}",
        'n_fraction': f"{sub('N', 'C')} / {sub('N', 'O')}",
        # 'fwhm_fraction': f"{sub(sym['sigma'], 'C')} / {sub(sym['sigma'], 'O')}",
        'fwhm_fraction': f"{sub('h', 'C')} / {sub('h', 'O')}",
        # 'noise_fraction': f"{sub(sym['lambda'], 'O')} / {sub(sym['lambda'], 'C')}",
        # 'noise_fraction': f"{sub('F', 'O')} / {sub('F', 'C')}",
        'noise_fraction': f"{sub('F', 'C')} / {sub('F', 'O')}",
        # 'rmax_fraction': f"{sub('r', 'max C')} / {sub('r', 'max O')}",
        'rmax_fraction': f"{sub('g', 'C')} / {sub('g', 'O')}",
    }
    return param_labels


def misc_symbols():
    symbols = {'deg': '\u00B0',
               '+-': '\u00B1',
               '**2': '\u00B2',
               '**3': '\u00B3',
               'micro': '\u00B5',
               'middle_dot': '\u00B7',
               'division': '\u00F7',
               'times': '\u00D7',
               'inf': '\u221E',
               'sqrt': '\u221A',
               '!=': '\u2260',
               '<=': '\u2264',
               '>=': '\u2265',
               '~': '\u2248',
               }
    return symbols


def sup(normal, superscript):
    # e.g. normal = sigma, superscript = 2
    subscript = "{" + superscript + "}"
    label = f"${normal}^{superscript}$"
    return label


def sub(normal, subscript):
    # e.g. normal = r, subscript = max
    subscript = "{" + subscript + "}"
    label = f"${normal}_{subscript}$"
    return label


def italics(s):
    italicised_string = r"$\it{" + s + "}$"
    return italicised_string


def bold(s):
    bold_string = r"$\mathbf{" + s + "}$"
    return bold_string

